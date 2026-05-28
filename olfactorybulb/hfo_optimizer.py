"""Batch-oriented HFO regime search helpers for the provisional EPLI model.

The search strategy is intentionally batch-first:

1. Seed with a Latin-hypercube design in transformed parameter space.
2. Evaluate each candidate in paired control / ketamine-block conditions.
3. Score candidates on differential HFO expression in the configured target HFO band.
4. Refine around elites with a truncated Gaussian proposal plus exploration.

This is a better fit for Phoenix than Nelder-Mead because the objective is
noisy, non-smooth, and expensive, and Phoenix throughput is highest when we
launch many independent runs concurrently inside one long-lived allocation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import math

import numpy as np
from scipy.stats import qmc

import obgpu_experiment_helpers as hlp


def _default_repo_root() -> Path:
    """Prefer the user's visible checkout path over a resolved symlink target."""
    home_checkout = Path.home() / "OlfactoryBulb"
    if home_checkout.exists():
        return home_checkout
    return Path(__file__).resolve().parents[1]


DEFAULT_CAMPAIGNS_BASE = _default_repo_root() / "results" / "notebook_runs" / "optimization"
DEFAULT_SCORE_BANDS = {
    "beta": (15.0, 35.0),
    "low_gamma": (35.0, 65.0),
    "high_gamma": (65.0, 100.0),
    "hfo_80_130": (80.0, 130.0),
    "hfo_130_160": (130.0, 160.0),
    "target_hfo": (160.0, 230.0),
    "supra_hfo": (230.0, 260.0),
}
PSD_TEMPLATE_FREQS_HZ = tuple(float(value) for value in np.arange(20.0, 301.0, 5.0))

PAIR_SCORE_VERSION = 6
ARCHIVE_FILTER_FILENAME = "objective_filter.json"
PLAUSIBILITY_SOFT_LIMITS = {
    "kar_mt_gmax": 0.05,
    "kar_gc_gmax": 0.02,
    "kar_osn_weight_scale": 2.0,
    "kar_gc_weight_scale": 4.0,
    "kar_mt_effective_drive": 0.10,
}


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    low: float
    high: float
    scale: str = "log"
    dtype: str = "float"
    description: str = ""
    default: float | None = None

    def clamp(self, value: float) -> float:
        return min(max(float(value), float(self.low)), float(self.high))

    def encode(self, value: float) -> float:
        value = self.clamp(value)
        if self.scale == "log":
            return math.log10(max(value, 1e-12))
        if self.scale != "linear":
            raise ValueError(f"Unsupported scale {self.scale!r}")
        return float(value)

    def decode(self, value: float) -> float:
        if self.scale == "log":
            decoded = 10.0 ** float(value)
        elif self.scale == "linear":
            decoded = float(value)
        else:
            raise ValueError(f"Unsupported scale {self.scale!r}")
        decoded = self.clamp(decoded)
        if self.dtype == "int":
            return int(round(decoded))
        return float(decoded)

    def low_encoded(self) -> float:
        return self.encode(self.low)

    def high_encoded(self) -> float:
        return self.encode(self.high)

    def default_value(self) -> float:
        if self.default is not None:
            return self.clamp(float(self.default))
        if self.scale == "log":
            return self.decode(0.5 * (self.low_encoded() + self.high_encoded()))
        return self.decode(0.5 * (float(self.low) + float(self.high)))


def default_hfo_search_space() -> list[ParameterSpec]:
    """Return the default conductance-focused search space.

    The ranges are intentionally wide enough to discover a viable regime while
    still remaining organized around interpretable conductance and coupling
    knobs rather than arbitrary timing rewrites.
    """
    return [
        ParameterSpec(
            path="kar_mt_gmax",
            low=0.01,
            high=0.08,
            scale="log",
            description="KAR conductance on M/T cells",
        ),
        ParameterSpec(
            path="kar_gc_gmax",
            low=0.001,
            high=0.025,
            scale="log",
            description="Optional KAR conductance on granule cells",
        ),
        ParameterSpec(
            path="gaba_gmax",
            low=0.25,
            high=8.0,
            scale="log",
            description="Fast inhibitory GABA-A max conductance",
        ),
        ParameterSpec(
            path="ampa_nmda_gmax",
            low=16.0,
            high=128.0,
            scale="log",
            description="Dendrodendritic AMPA/NMDA max conductance",
        ),
        ParameterSpec(
            path="epli_ampa_weight_scale",
            low=0.1,
            high=8.0,
            scale="log",
            default=1.0,
            description="MC/TC->EPLI reciprocal excitation weight scale",
        ),
        ParameterSpec(
            path="epli_gaba_weight_scale",
            low=0.1,
            high=8.0,
            scale="log",
            default=1.0,
            description="EPLI->MC/TC reciprocal inhibition weight scale",
        ),
        ParameterSpec(
            path="gap_tc",
            low=4.0,
            high=64.0,
            scale="log",
            description="TC gap-junction conductance",
        ),
        ParameterSpec(
            path="gap_mc",
            low=2.0,
            high=48.0,
            scale="log",
            description="MC gap-junction conductance",
        ),
        ParameterSpec(
            path="tc_input_weight",
            low=0.4,
            high=1.2,
            scale="linear",
            description="Feedforward TC input weight",
        ),
        ParameterSpec(
            path="mc_input_weight",
            low=0.05,
            high=0.35,
            scale="linear",
            description="Feedforward MC input weight",
        ),
        ParameterSpec(
            path="kar_osn_weight_scale",
            low=0.25,
            high=2.0,
            scale="log",
            default=1.0,
            description="OSN event weight multiplier for M/T KAR traces",
        ),
        ParameterSpec(
            path="kar_gc_weight_scale",
            low=0.25,
            high=4.0,
            scale="log",
            default=1.0,
            description="M/T event weight multiplier for GC KAR traces",
        ),
        ParameterSpec(
            path="gc_ka_gbar_scale",
            low=0.25,
            high=3.0,
            scale="log",
            default=1.0,
            description="Granule-cell A-type potassium conductance scale",
        ),
    ]


def search_space_rows(search_space: Sequence[ParameterSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in search_space]


def infer_remote_template_from_recent_runs(
    *,
    results_base: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return the newest remote run config we can recover from saved run_info."""
    results_base = Path(results_base or (_default_repo_root() / "results" / "notebook_runs"))
    candidates = sorted(results_base.glob("**/run_info.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        config = payload.get("config") or {}
        if str(config.get("runner_backend", "")).strip() not in {"sol_slurm", "slurm_remote"}:
            continue
        return dict(config)
    return None


def build_manual_allocation_remote_config(
    *,
    slurm_allocation_job_id: str,
    base_template: dict[str, Any] | None = None,
    total_tasks: int = 120,
) -> dict[str, Any]:
    """Build a remote config that reuses one explicit user-managed allocation."""
    template = dict(base_template or {})
    remote_host = str(template.get("remote_host") or "jmpaniag@localhost")
    remote_repo_root = str(template.get("remote_repo_root") or "/home/jmpaniag/OlfactoryBulb")
    remote_results_root = str(
        template.get("remote_results_root") or (Path(remote_repo_root) / "results" / "notebook_runs")
    )
    config = hlp.build_sol_remote_config(
        remote_host=remote_host,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        remote_conda_activate_cmd=str(template.get("remote_conda_activate_cmd") or "source tools/setup/activate_obgpu.sh"),
        remote_runtime_profiles=list(template.get("remote_runtime_profiles") or []),
        remote_fallback_conda_activate_cmd=template.get("remote_fallback_conda_activate_cmd"),
        remote_fast_node_feature=template.get("remote_fast_node_feature"),
        remote_mechanism_profile=str(template.get("remote_mechanism_profile") or "default"),
        remote_fallback_mechanism_profile=str(template.get("remote_fallback_mechanism_profile") or "portable"),
        remote_mpi_exec=str(template.get("remote_mpi_exec") or hlp.default_remote_mpi_exec()),
        slurm_partition=template.get("slurm_partition"),
        slurm_account=template.get("slurm_account"),
        slurm_time=template.get("slurm_time"),
        slurm_gpus=template.get("slurm_gpus"),
        slurm_cpus_per_task=template.get("slurm_cpus_per_task"),
        slurm_mem=template.get("slurm_mem"),
        sweep_sync_live=False,
        sweep_sync_soma_vs=False,
        sweep_sync_voltage_summary=False,
        remote_preserve_paramiko_session=True,
        remote_repo_mode=str(template.get("remote_repo_mode") or "shared"),
        remote_git_ref=template.get("remote_git_ref"),
        remote_git_fetch=bool(template.get("remote_git_fetch", False)),
        remote_git_remote=str(template.get("remote_git_remote") or "origin"),
        slurm_allocation_job_id=str(slurm_allocation_job_id),
        slurm_reuse_allocation=False,
        slurm_allocation_time=None,
        slurm_allocation_name=None,
        ssh_options=list(template.get("ssh_options") or []),
        slurm_extra_args=[],
    )
    config["optimizer_total_tasks"] = int(total_tasks)
    return config


def paramiko_auth_probe(
    remote_config: dict[str, Any],
    *,
    command: str = "printf 'paramiko-auth-ok host=%s cwd=%s\\n' \"$(hostname)\" \"$PWD\"",
) -> dict[str, Any]:
    """Perform a cheap Paramiko-authenticated remote command."""
    completed = hlp._run_ssh_shell(remote_config, command)
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def sustained_odor_schedule(
    tstop_ms: float,
    *,
    period_ms: float = 200.0,
    odor_name: str = "Apple",
    rel_conc: float = 0.2,
) -> dict[int, dict[str, Any]]:
    """Build a repeated odor schedule covering a long HFO optimization run."""
    tstop_ms = float(tstop_ms)
    period_ms = max(float(period_ms), 1e-9)
    onsets = np.arange(0.0, max(tstop_ms, 0.0), period_ms)
    return {
        int(round(float(onset))): {"name": str(odor_name), "rel_conc": float(rel_conc)}
        for onset in onsets
    }


def default_campaign_run_config(
    remote_config: dict[str, Any],
    *,
    paramset: str = "GammaSignature_EPLI_Provisional_TCOnly",
    nranks: int = 15,
    total_tasks: int = 120,
    tstop_ms: float = 9000.0,
    cell_permute: int = 0,
    odor_period_ms: float = 200.0,
    odor_rel_conc: float = 0.2,
    inhale_duration_ms: float = 125.0,
) -> dict[str, Any]:
    """Return the base notebook run config for one HFO search campaign."""
    config = hlp.build_run_config(
        mode="fast",
        paramset=paramset,
        label_prefix="hfo_optimizer",
        nranks=int(nranks),
        use_corenrn=True,
        use_gpu=False,
        cell_permute=int(cell_permute),
        tstop_ms=float(tstop_ms),
        input_odors=sustained_odor_schedule(
            tstop_ms,
            period_ms=odor_period_ms,
            rel_conc=odor_rel_conc,
        ),
        inhale_duration_ms=float(inhale_duration_ms),
        sim_dt_ms=0.1,
        recording_period_ms=0.1,
        analysis_dt_ms=0.1,
        spectrogram_signal="lfp",
        wavelet_signal="lfp",
        enable_lfp=True,
        enable_reciprocal_synapses=True,
        enable_epl_interneurons=True,
        max_epl_interneurons=24,
        epl_interneuron_cell_type="EPLI",
        enable_gc_kar=True,
        record_from_somas=["MC", "TC", "GC", "EPLI"],
        record_gc_output_events=False,
        save_soma_traces=False,
        save_voltage_summary=False,
        keep_native_lfp_debug_files=False,
        sweep_engine="remote_batch",
        sweep_sync_live=False,
        sweep_sync_soma_vs=False,
        sweep_sync_voltage_summary=False,
        sweep_parallelism=max(int(total_tasks) // max(int(nranks), 1), 1),
    )
    config.update(dict(remote_config))
    config["nranks"] = int(nranks)
    config["use_corenrn"] = True
    config["use_gpu"] = False
    config["cell_permute"] = int(cell_permute)
    config["tstop_ms"] = float(tstop_ms)
    config["input_odors"] = sustained_odor_schedule(
        tstop_ms,
        period_ms=odor_period_ms,
        rel_conc=odor_rel_conc,
    )
    config["inhale_duration_ms"] = float(inhale_duration_ms)
    config["enable_gc_kar"] = True
    config["sweep_parallelism"] = max(int(total_tasks) // max(int(nranks), 1), 1)
    return config


def lfp_source_diagnostic_configs(
    base_config: dict[str, Any],
    *,
    shifted_locations: Sequence[Sequence[float]] | None = None,
    non_gc_cell_types: Sequence[str] = ("MC", "TC", "EPLI", "PVCRH"),
) -> dict[str, dict[str, Any]]:
    """Build LFP-source diagnostic variants without changing circuit dynamics."""
    base = dict(base_config)
    base.setdefault("enable_lfp", True)
    variants: dict[str, dict[str, Any]] = {
        "all_sources": dict(base),
        "exclude_gc_lfp": {
            **base,
            "lfp_include_cell_types": None,
            "lfp_exclude_cell_types": ["GC"],
        },
        "non_gc_sources_lfp": {
            **base,
            "lfp_include_cell_types": list(non_gc_cell_types),
            "lfp_exclude_cell_types": None,
        },
    }

    for index, location in enumerate(shifted_locations or ()):
        variants[f"probe_shift_{index:02d}"] = {
            **base,
            "lfp_electrode_location": [float(value) for value in location],
            "lfp_include_cell_types": None,
            "lfp_exclude_cell_types": None,
        }

    return variants


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "campaign"


def ensure_campaign_dir(
    campaign_name: str,
    *,
    base_dir: str | Path = DEFAULT_CAMPAIGNS_BASE,
) -> Path:
    campaign_dir = Path(base_dir) / _safe_slug(campaign_name)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "batches").mkdir(exist_ok=True)
    return campaign_dir


def _state_path(campaign_dir: Path) -> Path:
    return campaign_dir / "state.json"


def _archive_path(campaign_dir: Path, *, kind: str) -> Path:
    return campaign_dir / f"{kind}_archive.jsonl"


def _batch_index_from_name(batch_name: Any) -> int | None:
    text = str(batch_name or "")
    if "_" not in text:
        return None
    tail = text.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def _archive_filter_path(campaign_dir: Path) -> Path:
    return campaign_dir / ARCHIVE_FILTER_FILENAME


def load_objective_filter(campaign_dir: str | Path) -> dict[str, Any]:
    """Load an optional campaign-local archive filter for objective pivots."""
    path = _archive_filter_path(Path(campaign_dir))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_objective_filter(campaign_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Write a campaign-local filter used when ranking/proposing candidates."""
    path = _archive_filter_path(Path(campaign_dir))
    path.write_text(json.dumps(hlp._json_ready(dict(payload)), indent=2, sort_keys=True) + "\n")
    return path


def _row_matches_objective_filter(row: dict[str, Any], objective_filter: dict[str, Any]) -> bool:
    try:
        min_batch_index = int(objective_filter.get("min_batch_index", 0))
    except (TypeError, ValueError):
        min_batch_index = 0
    if min_batch_index > 0:
        batch_index = _batch_index_from_name(row.get("batch_name"))
        if batch_index is None or batch_index < min_batch_index:
            return False
    return True


def _target_band_bounds(
    bands: dict[str, tuple[float, float]],
    *,
    target_hz: float,
    target_half_width_hz: float,
) -> tuple[float, float, float, float]:
    """Return authoritative target bounds.

    The named ``target_hfo`` band is the scoring objective. The target
    center/half-width arguments are retained for callers that provide custom
    bands without ``target_hfo``; stale notebook arguments should not silently
    narrow the objective after the default band changes.
    """
    target_band = bands.get("target_hfo")
    if target_band is not None:
        lo, hi = float(target_band[0]), float(target_band[1])
        if hi < lo:
            lo, hi = hi, lo
        center = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo)
        return lo, hi, center, half_width
    center = float(target_hz)
    half_width = float(target_half_width_hz)
    return center - half_width, center + half_width, center, half_width


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def initialize_campaign(
    campaign_dir: str | Path,
    *,
    base_config: dict[str, Any],
    search_space: Sequence[ParameterSpec],
    notes: str | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "batches").mkdir(exist_ok=True)
    config_payload = {
        "base_config": hlp._json_ready(base_config),
        "search_space": [asdict(spec) for spec in search_space],
        "notes": notes or "",
    }
    _write_json(campaign_dir / "campaign_config.json", config_payload)
    state = _read_json(
        _state_path(campaign_dir),
        {
            "next_batch_index": 0,
            "next_candidate_index": 0,
            "completed_batches": [],
        },
    )
    _write_json(_state_path(campaign_dir), state)
    return state


def load_campaign_state(campaign_dir: str | Path) -> dict[str, Any]:
    return _read_json(_state_path(Path(campaign_dir)), {})


def _write_campaign_state(campaign_dir: Path, state: dict[str, Any]) -> None:
    _write_json(_state_path(campaign_dir), state)


def _candidate_id(state: dict[str, Any]) -> str:
    return f"C{int(state['next_candidate_index']):05d}"


def _batch_name(state: dict[str, Any]) -> str:
    return f"batch_{int(state['next_batch_index']):04d}"


def _sample_unit_lhs(n: int, d: int, *, seed: int | None = None) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return sampler.random(n=n)


def _decode_unit_samples(samples: np.ndarray, search_space: Sequence[ParameterSpec]) -> list[dict[str, float]]:
    decoded = []
    for row in np.asarray(samples, dtype=float):
        params = {}
        for coord, spec in zip(row, search_space):
            lo = spec.low_encoded()
            hi = spec.high_encoded()
            value = lo + float(coord) * (hi - lo)
            params[spec.path] = spec.decode(value)
        decoded.append(params)
    return decoded


def _random_unit_samples(n: int, d: int, *, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def _candidate_vector(candidate: dict[str, Any], search_space: Sequence[ParameterSpec]) -> np.ndarray:
    values = []
    for spec in search_space:
        raw_value = candidate.get(spec.path, spec.default_value())
        values.append(spec.encode(float(raw_value)))
    return np.asarray(values, dtype=float)


def _candidate_metric(row: dict[str, Any], condition: str, field: str, default: float = 0.0) -> float:
    metrics = row.get(f"{condition}_metrics") or {}
    if field == "target_hfo":
        value = (metrics.get("relative_band_power") or {}).get("target_hfo", default)
    else:
        value = metrics.get(field, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _relative_band(metrics: dict[str, Any], band: str, default: float = 0.0) -> float:
    try:
        return float((metrics.get("relative_band_power") or {}).get(band, default))
    except (TypeError, ValueError):
        return float(default)


def _band_power(metrics: dict[str, Any], band: str, default: float = 0.0) -> float:
    try:
        return float((metrics.get("band_power") or {}).get(band, default))
    except (TypeError, ValueError):
        return float(default)


def _supra_hfo_relative(metrics: dict[str, Any]) -> float:
    """Return undesirable above-target HFO power, with fallback for old archives."""
    if "supra_hfo" in (metrics.get("relative_band_power") or {}):
        return _relative_band(metrics, "supra_hfo")
    # Old campaign rows used 160-200 Hz as target and 200-250 Hz as the
    # above-target side band. Treat that side band as an artifact when
    # re-ranking old archive rows under the broader 160-230 Hz objective.
    return _relative_band(metrics, "hfo_200_250")


def _target_density_ratio(metrics: dict[str, Any]) -> float:
    """Return total target-band density over background density."""
    for key in ("target_density_ratio", "peak_ratio"):
        try:
            value = float(metrics.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    return 0.0


def _target_clean_fraction(metrics: dict[str, Any]) -> float:
    target_power = _band_power(metrics, "target_hfo")
    lower_side_power = _band_power(metrics, "hfo_80_130") + _band_power(metrics, "hfo_130_160")
    supra_power = _band_power(metrics, "supra_hfo")
    if supra_power <= 0.0:
        supra_power = _band_power(metrics, "hfo_200_250")
    denom = target_power + lower_side_power + supra_power
    return target_power / denom if denom > 0.0 else 0.0


def _normalize_psd_shape(values: Sequence[float]) -> np.ndarray:
    shape = np.asarray(values, dtype=float)
    shape = np.where(np.isfinite(shape) & (shape > 0.0), shape, 0.0)
    total = float(np.sum(shape))
    if total <= 0.0:
        return np.zeros_like(shape, dtype=float)
    return shape / total


def _gaussian_on_psd_grid(center_hz: float, sigma_hz: float, weight: float) -> np.ndarray:
    grid = np.asarray(PSD_TEMPLATE_FREQS_HZ, dtype=float)
    return float(weight) * np.exp(-0.5 * ((grid - float(center_hz)) / max(float(sigma_hz), 1e-9)) ** 2)


def _theoretical_psd_template(kind: str) -> np.ndarray:
    """Return normalized PSD-shape targets for template-loss scoring."""
    grid = np.asarray(PSD_TEMPLATE_FREQS_HZ, dtype=float)
    baseline = np.full_like(grid, 0.010, dtype=float)
    if kind == "ketamine":
        shape = (
            baseline
            + _gaussian_on_psd_grid(24.0, 7.0, 0.055)
            + _gaussian_on_psd_grid(55.0, 13.0, 0.070)
            + _gaussian_on_psd_grid(85.0, 18.0, 0.100)
            + _gaussian_on_psd_grid(195.0, 18.0, 0.520)
        )
        shape = np.where(grid > 240.0, shape * 0.35, shape)
        return _normalize_psd_shape(shape)
    if kind == "control":
        shape = (
            baseline
            + _gaussian_on_psd_grid(24.0, 8.0, 0.110)
            + _gaussian_on_psd_grid(55.0, 16.0, 0.130)
            + _gaussian_on_psd_grid(90.0, 24.0, 0.120)
            + _gaussian_on_psd_grid(150.0, 55.0, 0.035)
        )
        shape = np.where((grid >= 160.0) & (grid <= 230.0), shape * 0.55, shape)
        return _normalize_psd_shape(shape)
    if kind == "contrast":
        shape = (
            _gaussian_on_psd_grid(195.0, 17.0, 1.0)
            + _gaussian_on_psd_grid(85.0, 22.0, 0.08)
        )
        shape = np.where(grid > 235.0, shape * 0.20, shape)
        return _normalize_psd_shape(shape)
    raise ValueError(f"Unknown PSD template kind {kind!r}")


def psd_template_curve(
    kind: str,
    freqs_hz: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a normalized theoretical PSD template for diagnostics.

    The returned curve is the same shape target used by the v6 objective.  It is
    normalized as a discrete probability vector, so callers plotting it on top
    of a measured PSD should scale it to the measured units first.
    """
    template_freqs = np.asarray(PSD_TEMPLATE_FREQS_HZ, dtype=float)
    template_power = _theoretical_psd_template(kind)
    if freqs_hz is None:
        return template_freqs.copy(), template_power.copy()

    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.ndim != 1:
        raise ValueError("freqs_hz must be one-dimensional")
    curve = np.interp(freqs, template_freqs, template_power, left=0.0, right=0.0)
    return freqs.copy(), _normalize_psd_shape(curve)


def scaled_psd_template_curve(
    kind: str,
    freqs_hz: Sequence[float] | np.ndarray,
    reference_psd: Sequence[float] | np.ndarray,
    *,
    fit_band_hz: tuple[float, float] = (20.0, 300.0),
    method: str = "area",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a PSD template scaled onto a measured PSD axis.

    `method="area"` matches integrated power over `fit_band_hz`; `method="peak"`
    matches the peak height in that band.  This is for visual diagnostics only;
    scoring still uses the normalized template vectors.
    """
    freqs, template = psd_template_curve(kind, freqs_hz)
    reference = np.asarray(reference_psd, dtype=float)
    if reference.shape != freqs.shape:
        raise ValueError("reference_psd must have the same shape as freqs_hz")

    lo_hz, hi_hz = fit_band_hz
    mask = (
        np.isfinite(freqs)
        & np.isfinite(template)
        & np.isfinite(reference)
        & (freqs >= float(lo_hz))
        & (freqs <= float(hi_hz))
    )
    if not np.any(mask):
        return freqs, np.zeros_like(template)

    if method == "area":
        template_scale = float(np.trapezoid(template[mask], freqs[mask]))
        reference_scale = float(np.trapezoid(np.maximum(reference[mask], 0.0), freqs[mask]))
    elif method == "peak":
        template_scale = float(np.max(template[mask]))
        reference_scale = float(np.max(np.maximum(reference[mask], 0.0)))
    else:
        raise ValueError(f"Unsupported template scaling method {method!r}")

    if template_scale <= 0.0 or reference_scale <= 0.0:
        return freqs, np.zeros_like(template)
    return freqs, template * (reference_scale / template_scale)


def _psd_shape_from_arrays(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(psd) & (psd > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.zeros(len(PSD_TEMPLATE_FREQS_HZ), dtype=float)
    grid = np.asarray(PSD_TEMPLATE_FREQS_HZ, dtype=float)
    interpolated = np.interp(grid, freqs[mask], psd[mask], left=0.0, right=0.0)
    return _normalize_psd_shape(np.sqrt(np.maximum(interpolated, 0.0)))


def _coarse_psd_shape_from_band_metrics(metrics: dict[str, Any]) -> np.ndarray:
    grid = np.asarray(PSD_TEMPLATE_FREQS_HZ, dtype=float)
    shape = np.zeros_like(grid, dtype=float)
    relative = metrics.get("relative_band_power") or {}
    band_bounds = dict(DEFAULT_SCORE_BANDS)
    target_band = metrics.get("target_band_hz")
    if isinstance(target_band, (list, tuple)) and len(target_band) == 2:
        try:
            target_lo = float(target_band[0])
            target_hi = float(target_band[1])
        except (TypeError, ValueError):
            target_lo, target_hi = DEFAULT_SCORE_BANDS["target_hfo"]
        band_bounds["target_hfo"] = (target_lo, target_hi)
    for band_name, (lo_hz, hi_hz) in band_bounds.items():
        try:
            power = float(relative.get(band_name, 0.0))
        except (TypeError, ValueError):
            power = 0.0
        if power <= 0.0 or hi_hz <= lo_hz:
            continue
        mask = (grid >= lo_hz) & (grid <= hi_hz)
        if np.any(mask):
            shape[mask] += power / max(float(np.count_nonzero(mask)), 1.0)
    return _normalize_psd_shape(shape)


def _psd_shape_from_metrics(metrics: dict[str, Any]) -> np.ndarray:
    raw = metrics.get("psd_shape_power")
    if isinstance(raw, (list, tuple)) and len(raw) == len(PSD_TEMPLATE_FREQS_HZ):
        return _normalize_psd_shape(raw)
    return _coarse_psd_shape_from_band_metrics(metrics)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-18:
        return 0.0
    value = float(np.dot(left, right) / denom)
    return min(max(value, 0.0), 1.0)


def _psd_template_pair_metrics(
    control_metrics: dict[str, Any],
    ketamine_metrics: dict[str, Any],
) -> dict[str, float]:
    control_shape = _psd_shape_from_metrics(control_metrics)
    ketamine_shape = _psd_shape_from_metrics(ketamine_metrics)
    ketamine_template = _theoretical_psd_template("ketamine")
    control_template = _theoretical_psd_template("control")
    contrast_template = _theoretical_psd_template("contrast")
    positive_contrast_shape = _normalize_psd_shape(np.maximum(ketamine_shape - control_shape, 0.0))

    ketamine_similarity = _cosine_similarity(ketamine_shape, ketamine_template)
    control_similarity = _cosine_similarity(control_shape, control_template)
    contrast_similarity = _cosine_similarity(positive_contrast_shape, contrast_template)
    control_hfo_similarity = _cosine_similarity(control_shape, ketamine_template)

    ketamine_loss = 1.0 - ketamine_similarity
    control_loss = 1.0 - control_similarity
    contrast_loss = 1.0 - contrast_similarity
    template_loss = (
        1.20 * ketamine_loss
        + 0.75 * control_loss
        + 1.60 * contrast_loss
        + 0.80 * control_hfo_similarity
    )
    template_score = (
        1.20 * ketamine_similarity
        + 0.75 * control_similarity
        + 1.60 * contrast_similarity
        - 0.80 * control_hfo_similarity
    )
    return {
        "psd_template_loss": float(template_loss),
        "psd_template_score": float(template_score),
        "ketamine_psd_template_loss": float(ketamine_loss),
        "control_psd_template_loss": float(control_loss),
        "psd_contrast_template_loss": float(contrast_loss),
        "ketamine_psd_template_similarity": float(ketamine_similarity),
        "control_psd_template_similarity": float(control_similarity),
        "psd_contrast_template_similarity": float(contrast_similarity),
        "control_hfo_template_similarity": float(control_hfo_similarity),
    }


def parameter_plausibility_penalty(parameters: dict[str, Any] | None) -> tuple[float, dict[str, float]]:
    """Return a soft penalty for KAR settings outside the working plausible range."""
    parameters = dict(parameters or {})
    components: dict[str, float] = {}

    def value(name: str, default: float) -> float:
        try:
            raw = float(parameters.get(name, default))
        except (TypeError, ValueError):
            return float(default)
        return raw if math.isfinite(raw) else float(default)

    def add_log_excess(name: str, soft_limit: float, weight: float) -> None:
        raw = value(name, soft_limit)
        if raw <= soft_limit:
            return
        excess = math.log10(max(raw / soft_limit, 1.0))
        components[name] = float(weight * excess * excess)

    add_log_excess("kar_mt_gmax", PLAUSIBILITY_SOFT_LIMITS["kar_mt_gmax"], 20.0)
    add_log_excess("kar_gc_gmax", PLAUSIBILITY_SOFT_LIMITS["kar_gc_gmax"], 15.0)
    add_log_excess("kar_osn_weight_scale", PLAUSIBILITY_SOFT_LIMITS["kar_osn_weight_scale"], 8.0)
    add_log_excess("kar_gc_weight_scale", PLAUSIBILITY_SOFT_LIMITS["kar_gc_weight_scale"], 5.0)

    mt_drive = value("kar_mt_gmax", 0.0) * value("kar_osn_weight_scale", 1.0)
    mt_drive_limit = PLAUSIBILITY_SOFT_LIMITS["kar_mt_effective_drive"]
    if mt_drive > mt_drive_limit:
        excess = math.log10(max(mt_drive / mt_drive_limit, 1.0))
        components["kar_mt_effective_drive"] = float(25.0 * excess * excess)

    return float(sum(components.values())), components


def _apply_parameter_plausibility_penalty(
    pair_metrics: dict[str, Any],
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    """Subtract the KAR plausibility penalty from one pair score payload."""
    adjusted = dict(pair_metrics)
    base_score = float(adjusted.get("pair_score", float("-inf")))
    penalty, components = parameter_plausibility_penalty(parameters)
    adjusted["unpenalized_pair_score"] = float(base_score)
    adjusted["parameter_plausibility_penalty"] = float(penalty)
    adjusted["parameter_plausibility_components"] = components
    if math.isfinite(base_score):
        adjusted["pair_score"] = float(base_score - penalty)
    return adjusted


def _target_band_for_pair(
    control_metrics: dict[str, Any],
    ketamine_metrics: dict[str, Any],
) -> tuple[float, float]:
    """Return the active target band recorded in condition metrics."""
    for metrics in (ketamine_metrics, control_metrics):
        raw_band = metrics.get("target_band_hz")
        if not isinstance(raw_band, (list, tuple)) or len(raw_band) != 2:
            continue
        try:
            lo = float(raw_band[0])
            hi = float(raw_band[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi
    lo, hi = DEFAULT_SCORE_BANDS["target_hfo"]
    return float(lo), float(hi)


def rescore_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute pair-level ranking fields for one archive row."""
    control_metrics = row.get("control_metrics")
    ketamine_metrics = row.get("ketamine_metrics")
    if control_metrics is None or ketamine_metrics is None:
        return row
    rescored = dict(row)
    pair_metrics = score_candidate_pair(
        control_metrics=control_metrics,
        ketamine_metrics=ketamine_metrics,
    )
    rescored.update(_apply_parameter_plausibility_penalty(pair_metrics, row.get("parameters")))
    return rescored


def _sample_truncated_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    search_space: Sequence[ParameterSpec],
    n: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = np.asarray([spec.low_encoded() for spec in search_space], dtype=float)
    hi = np.asarray([spec.high_encoded() for spec in search_space], dtype=float)
    rows = []
    attempts = 0
    max_attempts = max(1000, 100 * n)
    while len(rows) < n and attempts < max_attempts:
        candidate = rng.multivariate_normal(mean=mean, cov=cov)
        attempts += 1
        if np.all(candidate >= lo) and np.all(candidate <= hi):
            rows.append(candidate)
    if len(rows) < n:
        fallback = _random_unit_samples(n - len(rows), len(search_space), seed=seed)
        for row in _decode_unit_samples(fallback, search_space):
            rows.append(_candidate_vector(row, search_space))
    return np.asarray(rows, dtype=float)


def _targeted_elite_probe_rows(
    elite_vectors: np.ndarray,
    search_space: Sequence[ParameterSpec],
    n: int,
    *,
    mode: str = "line",
    seed: int | None = None,
    archive_rows: Sequence[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Return deterministic local probes around the top two elite candidates."""
    n = int(n)
    if n <= 0 or len(elite_vectors) < 2:
        return np.empty((0, len(search_space)), dtype=float)

    rng = np.random.default_rng(seed)
    encoded_lo = np.asarray([spec.low_encoded() for spec in search_space], dtype=float)
    encoded_hi = np.asarray([spec.high_encoded() for spec in search_space], dtype=float)
    encoded_span = encoded_hi - encoded_lo
    top = np.asarray(elite_vectors[0], dtype=float)
    second = np.asarray(elite_vectors[1], dtype=float)
    rows: list[np.ndarray] = []

    priority_paths = [
        "gaba_gmax",
        "epli_ampa_weight_scale",
        "epli_gaba_weight_scale",
        "kar_gc_gmax",
        "kar_mt_gmax",
        "tc_input_weight",
        "gap_tc",
        "ampa_nmda_gmax",
        "kar_gc_weight_scale",
        "gc_ka_gbar_scale",
        "gap_mc",
        "mc_input_weight",
        "kar_osn_weight_scale",
    ]
    priority_indices = [
        index
        for path in priority_paths
        for index, spec in enumerate(search_space)
        if spec.path == path
    ]
    if not priority_indices:
        priority_indices = list(range(len(search_space)))

    if mode == "line":
        # First walk the line between the current best and the strongest near miss.
        for alpha in (0.25, 0.50, 0.75):
            if len(rows) >= n:
                break
            rows.append(np.clip((1.0 - alpha) * top + alpha * second, encoded_lo, encoded_hi))

    if mode == "stencil":
        move_plan = [
            ("gaba_gmax", 0.030),
            ("gaba_gmax", -0.030),
            ("kar_gc_gmax", 0.030),
            ("kar_gc_gmax", -0.030),
            ("kar_mt_gmax", 0.030),
            ("kar_mt_gmax", -0.030),
            ("tc_input_weight", -0.030),
            ("ampa_nmda_gmax", -0.020),
            ("gap_tc", -0.020),
            ("kar_gc_weight_scale", 0.020),
            ("gc_ka_gbar_scale", 0.020),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for path, step_fraction in move_plan:
            if len(rows) >= n:
                break
            if path not in path_to_index:
                continue
            row = np.array(top, copy=True)
            dim = path_to_index[path]
            row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    if mode == "combo":
        combo_plan = [
            (("tc_input_weight", -0.030), ("ampa_nmda_gmax", -0.020)),
            (("tc_input_weight", -0.030), ("gaba_gmax", 0.030)),
            (("kar_gc_gmax", 0.020), ("gaba_gmax", 0.020)),
            (("kar_gc_gmax", -0.020), ("kar_mt_gmax", 0.020)),
            (("kar_mt_gmax", 0.020), ("gaba_gmax", 0.020)),
            (("gap_tc", -0.030), ("gaba_gmax", 0.020)),
            (("gap_tc", -0.030), ("tc_input_weight", -0.020)),
            (("kar_gc_weight_scale", 0.025), ("kar_gc_gmax", -0.015)),
            (("gc_ka_gbar_scale", 0.025), ("gaba_gmax", 0.015)),
            (("ampa_nmda_gmax", 0.015), ("gaba_gmax", 0.025)),
            (("kar_gc_weight_scale", 0.025), ("kar_mt_gmax", -0.015), ("gaba_gmax", 0.015)),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for moves in combo_plan:
            if len(rows) >= n:
                break
            row = np.array(top, copy=True)
            for path, step_fraction in moves:
                if path not in path_to_index:
                    continue
                dim = path_to_index[path]
                row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    if mode == "micro":
        micro_plan = [
            (("gaba_gmax", 0.012),),
            (("gaba_gmax", -0.012),),
            (("gap_tc", -0.012),),
            (("gap_tc", 0.012),),
            (("gaba_gmax", 0.012), ("gap_tc", -0.012)),
            (("gaba_gmax", -0.012), ("gap_tc", 0.012)),
            (("ampa_nmda_gmax", -0.012),),
            (("ampa_nmda_gmax", -0.012), ("gaba_gmax", 0.010)),
            (("ampa_nmda_gmax", -0.012), ("gap_tc", -0.010)),
            (("tc_input_weight", -0.010), ("gaba_gmax", 0.010)),
            (("kar_gc_weight_scale", 0.010), ("kar_gc_gmax", -0.010)),
            (("gc_ka_gbar_scale", 0.010), ("gaba_gmax", 0.010)),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for moves in micro_plan:
            if len(rows) >= n:
                break
            row = np.array(top, copy=True)
            for path, step_fraction in moves:
                if path not in path_to_index:
                    continue
                dim = path_to_index[path]
                row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    if mode == "ridge":
        ridge_centers = {
            "top": top,
            "second": second,
            "third": elite_vectors[2] if len(elite_vectors) > 2 else second,
            "power": elite_vectors[3] if len(elite_vectors) > 3 else second,
            "leak": elite_vectors[4] if len(elite_vectors) > 4 else top,
        }
        ridge_plan = [
            ("top", (("gaba_gmax", 0.006),)),
            ("top", (("gaba_gmax", 0.018), ("gap_tc", -0.006))),
            ("top", (("gap_tc", -0.018),)),
            ("top", (("ampa_nmda_gmax", -0.018), ("gaba_gmax", 0.006), ("gap_tc", -0.006))),
            ("top", (("kar_gc_weight_scale", 0.012), ("kar_gc_gmax", -0.008), ("gaba_gmax", 0.006))),
            ("top", (("tc_input_weight", -0.008), ("gaba_gmax", 0.006), ("gap_tc", -0.006))),
            ("second", (("gaba_gmax", 0.030), ("gap_tc", -0.030), ("ampa_nmda_gmax", -0.010))),
            ("power", (("gaba_gmax", 0.030), ("gap_tc", -0.030))),
            ("power", (("gaba_gmax", 0.030), ("gap_tc", -0.030), ("ampa_nmda_gmax", 0.010))),
            ("leak", (("ampa_nmda_gmax", 0.006), ("gaba_gmax", 0.006))),
            ("leak", (("ampa_nmda_gmax", 0.012), ("gap_tc", -0.006))),
            ("third", (("gaba_gmax", 0.020), ("gap_tc", -0.020), ("kar_gc_gmax", 0.006))),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for center_name, moves in ridge_plan:
            if len(rows) >= n:
                break
            row = np.array(ridge_centers[center_name], copy=True)
            for path, step_fraction in moves:
                if path not in path_to_index:
                    continue
                dim = path_to_index[path]
                row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    if mode == "needle":
        needle_centers = {
            "top": top,
            "second": second,
            "third": elite_vectors[2] if len(elite_vectors) > 2 else second,
            "fourth": elite_vectors[3] if len(elite_vectors) > 3 else second,
        }
        needle_plan = [
            ("top", (("gaba_gmax", -0.003),)),
            ("top", (("gaba_gmax", -0.006),)),
            ("top", (("gaba_gmax", -0.003), ("ampa_nmda_gmax", -0.004))),
            ("top", (("gaba_gmax", -0.006), ("gap_tc", 0.004))),
            ("top", (("gap_tc", 0.004),)),
            ("top", (("ampa_nmda_gmax", -0.004),)),
            ("top", (("ampa_nmda_gmax", -0.004), ("gap_tc", 0.004))),
            ("top", (("tc_input_weight", 0.004), ("gaba_gmax", -0.003))),
            ("second", (("gaba_gmax", 0.010), ("ampa_nmda_gmax", -0.010))),
            ("second", (("gaba_gmax", 0.014), ("gap_tc", 0.004), ("ampa_nmda_gmax", -0.014))),
            ("third", (("gaba_gmax", 0.006), ("gap_tc", -0.006), ("ampa_nmda_gmax", -0.010))),
            ("fourth", (("gaba_gmax", 0.006), ("gap_tc", -0.006), ("ampa_nmda_gmax", -0.006))),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for center_name, moves in needle_plan:
            if len(rows) >= n:
                break
            row = np.array(needle_centers[center_name], copy=True)
            for path, step_fraction in moves:
                if path not in path_to_index:
                    continue
                dim = path_to_index[path]
                row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    if mode == "basin":
        archive = list(archive_rows or [])

        def row_vector(row: dict[str, Any] | None) -> np.ndarray | None:
            if row is None:
                return None
            try:
                return _candidate_vector(row["parameters"], search_space)
            except (KeyError, TypeError, ValueError):
                return None

        def in_ketamine_target(row: dict[str, Any]) -> bool:
            peak = _candidate_metric(row, "ketamine", "peak_hz", math.nan)
            return math.isfinite(peak) and 160.0 <= peak <= 230.0

        target_rows = [row for row in archive if in_ketamine_target(row)]
        power_pool = [
            row for row in target_rows
            if _candidate_metric(row, "control", "target_hfo") <= 0.105
        ]
        leak_pool = [
            row for row in target_rows
            if _candidate_metric(row, "ketamine", "target_hfo") >= 0.120
        ]
        distance_pool: list[tuple[float, dict[str, Any]]] = []
        for row in target_rows:
            vector = row_vector(row)
            if vector is None:
                continue
            distance = float(np.linalg.norm((vector - top) / np.maximum(encoded_span, 1e-9)))
            if (
                distance >= 0.18
                and _candidate_metric(row, "ketamine", "target_hfo") >= 0.090
                and _candidate_metric(row, "control", "target_hfo") <= 0.140
            ):
                distance_pool.append((distance, row))
        distant_ranked = [
            row for _distance, row in sorted(
                distance_pool,
                key=lambda item: float(item[1].get("pair_score", float("-inf"))),
                reverse=True,
            )
        ]

        power_row = max(power_pool, key=lambda row: _candidate_metric(row, "ketamine", "target_hfo"), default=None)
        leak_row = min(leak_pool, key=lambda row: _candidate_metric(row, "control", "target_hfo"), default=None)
        distant_one = distant_ranked[0] if len(distant_ranked) > 0 else None
        distant_two = distant_ranked[1] if len(distant_ranked) > 1 else None

        basin_centers = {
            "top": top,
            "second": second,
            "power": row_vector(power_row) if row_vector(power_row) is not None else (elite_vectors[2] if len(elite_vectors) > 2 else top),
            "leak": row_vector(leak_row) if row_vector(leak_row) is not None else second,
            "distant_one": row_vector(distant_one) if row_vector(distant_one) is not None else (elite_vectors[3] if len(elite_vectors) > 3 else second),
            "distant_two": row_vector(distant_two) if row_vector(distant_two) is not None else (elite_vectors[4] if len(elite_vectors) > 4 else top),
        }
        basin_plan = [
            ("top", (("gaba_gmax", -0.003), ("ampa_nmda_gmax", -0.004))),
            ("top", (("gap_tc", 0.004), ("ampa_nmda_gmax", -0.004))),
            ("power", (("gaba_gmax", 0.008), ("ampa_nmda_gmax", -0.006))),
            ("power", (("gaba_gmax", 0.012), ("gap_tc", -0.004))),
            ("power", (("gaba_gmax", 0.008), ("gap_tc", 0.004), ("ampa_nmda_gmax", -0.006))),
            ("leak", (("gaba_gmax", -0.006), ("ampa_nmda_gmax", -0.006))),
            ("leak", (("gap_tc", 0.006), ("kar_gc_gmax", 0.006))),
            ("second", (("gaba_gmax", 0.008), ("ampa_nmda_gmax", -0.012))),
            ("distant_one", (("gaba_gmax", 0.012), ("ampa_nmda_gmax", -0.010))),
            ("distant_one", (("kar_gc_gmax", 0.012), ("gaba_gmax", 0.006))),
            ("distant_two", (("gaba_gmax", 0.010), ("gap_tc", -0.008))),
            ("distant_two", (("ampa_nmda_gmax", -0.010), ("kar_gc_gmax", 0.010))),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for center_name, moves in basin_plan:
            if len(rows) >= n:
                break
            row = np.array(basin_centers[center_name], copy=True)
            for path, step_fraction in moves:
                if path not in path_to_index:
                    continue
                dim = path_to_index[path]
                row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    if mode == "frontier":
        archive = list(archive_rows or [])

        def row_vector(row: dict[str, Any] | None) -> np.ndarray | None:
            if row is None:
                return None
            try:
                return _candidate_vector(row["parameters"], search_space)
            except (KeyError, TypeError, ValueError):
                return None

        def target_rel(row: dict[str, Any], condition: str) -> float:
            return _candidate_metric(row, condition, "target_hfo")

        def target_delta(row: dict[str, Any]) -> float:
            return target_rel(row, "ketamine") - target_rel(row, "control")

        def peak_contrast(row: dict[str, Any], condition: str) -> float:
            value = row.get(f"{condition}_peak_contrast")
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                value_float = math.nan
            if math.isfinite(value_float):
                return value_float
            metrics = row.get(f"{condition}_metrics")
            if not isinstance(metrics, dict):
                return 0.0
            fallback = metrics.get("target_peak_contrast", metrics.get("peak_ratio", 0.0))
            try:
                fallback_float = float(fallback)
            except (TypeError, ValueError):
                return 0.0
            return fallback_float if math.isfinite(fallback_float) else 0.0

        def epli_rate(row: dict[str, Any], condition: str) -> float:
            value = row.get(f"{condition}_epli_rate_hz")
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                value_float = math.nan
            if math.isfinite(value_float):
                return value_float
            metrics = row.get(f"{condition}_metrics")
            if not isinstance(metrics, dict):
                return 0.0
            rates = metrics.get("mean_firing_rate_by_type")
            if not isinstance(rates, dict):
                return 0.0
            try:
                rate_float = float(rates.get("EPLI", 0.0))
            except (TypeError, ValueError):
                return 0.0
            return rate_float if math.isfinite(rate_float) else 0.0

        def contrast_support_score(row: dict[str, Any]) -> float:
            ketamine_peak = peak_contrast(row, "ketamine")
            control_peak = peak_contrast(row, "control")
            ketamine_epli = epli_rate(row, "ketamine")
            return (
                2.0 * math.log10(1.0 + ketamine_peak)
                - 1.0 * math.log10(1.0 + control_peak)
                + 2.5 * target_delta(row)
                + 0.2 * min(ketamine_epli, 8.0) / 8.0
            )

        def ketamine_peak(row: dict[str, Any]) -> float:
            return _candidate_metric(row, "ketamine", "peak_hz", math.nan)

        def in_target(row: dict[str, Any]) -> bool:
            peak = ketamine_peak(row)
            return math.isfinite(peak) and 160.0 <= peak <= 230.0

        target_rows = [row for row in archive if in_target(row)]
        exact_rows = [
            row for row in target_rows
            if target_rel(row, "ketamine") >= 0.12 and target_rel(row, "control") <= 0.11
        ]
        contrast_rows = [
            row for row in target_rows
            if target_rel(row, "ketamine") >= 0.12 and target_rel(row, "control") <= 0.11
        ]
        power_rows = [
            row for row in target_rows
            if target_rel(row, "control") <= 0.20
        ]
        low_control_rows = [
            row for row in target_rows
            if target_rel(row, "ketamine") >= 0.12
        ]
        contrast_support_rows = [
            row for row in target_rows
            if (
                epli_rate(row, "ketamine") >= 2.0
                and peak_contrast(row, "ketamine") >= peak_contrast(row, "control")
                and target_rel(row, "ketamine") >= 0.06
                and target_rel(row, "control") <= 0.14
            )
        ]

        exact_row = max(exact_rows, key=lambda row: target_rel(row, "ketamine"), default=None)
        contrast_row = max(contrast_rows, key=target_delta, default=None)
        power_row = max(power_rows, key=lambda row: target_rel(row, "ketamine"), default=None)
        low_control_row = min(low_control_rows, key=lambda row: target_rel(row, "control"), default=None)
        contrast_support_row = max(contrast_support_rows, key=contrast_support_score, default=None)

        exact_vector = row_vector(exact_row)
        contrast_vector = row_vector(contrast_row)
        power_vector = row_vector(power_row)
        low_control_vector = row_vector(low_control_row)
        contrast_support_vector = row_vector(contrast_support_row)
        exact = exact_vector if exact_vector is not None else top
        contrast = contrast_vector if contrast_vector is not None else second
        power = power_vector if power_vector is not None else (elite_vectors[2] if len(elite_vectors) > 2 else second)
        low_control = (
            low_control_vector
            if low_control_vector is not None
            else (elite_vectors[3] if len(elite_vectors) > 3 else top)
        )
        contrast_support = (
            contrast_support_vector
            if contrast_support_vector is not None
            else contrast
        )

        frontier_centers = {
            "top": top,
            "exact": exact,
            "contrast25": np.clip(0.75 * top + 0.25 * contrast, encoded_lo, encoded_hi),
            "contrast50": np.clip(0.50 * top + 0.50 * contrast, encoded_lo, encoded_hi),
            "contrast_support": contrast_support,
            "power": power,
            "low_control": low_control,
        }
        frontier_plan = [
            ("top", (("epli_ampa_weight_scale", 0.16),)),
            ("top", (("epli_gaba_weight_scale", 0.16),)),
            ("top", (("epli_ampa_weight_scale", -0.16),)),
            ("top", (("epli_gaba_weight_scale", -0.16),)),
            ("top", (("epli_ampa_weight_scale", -0.16), ("epli_gaba_weight_scale", -0.16))),
            ("top", (("epli_ampa_weight_scale", 0.16), ("epli_gaba_weight_scale", -0.16))),
            ("top", (("epli_ampa_weight_scale", -0.16), ("epli_gaba_weight_scale", 0.16))),
            ("exact", (("epli_ampa_weight_scale", -0.16), ("epli_gaba_weight_scale", -0.16), ("gaba_gmax", 0.004))),
            ("low_control", (("epli_ampa_weight_scale", -0.24), ("epli_gaba_weight_scale", -0.24), ("ampa_nmda_gmax", -0.004))),
            ("power", (("epli_ampa_weight_scale", 0.16), ("epli_gaba_weight_scale", 0.08), ("gaba_gmax", 0.006))),
            ("contrast25", (("epli_ampa_weight_scale", -0.16), ("epli_gaba_weight_scale", -0.16))),
            ("contrast50", (("epli_ampa_weight_scale", -0.24), ("epli_gaba_weight_scale", -0.16), ("gap_tc", -0.006))),
            ("contrast_support", (("epli_ampa_weight_scale", 0.12), ("epli_gaba_weight_scale", -0.08), ("gaba_gmax", 0.006))),
            ("contrast_support", (("epli_ampa_weight_scale", 0.18), ("kar_gc_gmax", 0.006))),
            ("contrast_support", (("gap_tc", -0.006), ("ampa_nmda_gmax", -0.006))),
        ]
        path_to_index = {spec.path: index for index, spec in enumerate(search_space)}
        for center_name, moves in frontier_plan:
            if len(rows) >= n:
                break
            row = np.array(frontier_centers[center_name], copy=True)
            for path, step_fraction in moves:
                if path not in path_to_index:
                    continue
                dim = path_to_index[path]
                row[dim] += float(step_fraction) * encoded_span[dim]
            rows.append(np.clip(row, encoded_lo, encoded_hi))

    # Fill any remaining slots with small one-coordinate probes around the top two points.
    while len(rows) < n:
        center = top if (mode in {"stencil", "ridge"} or len(rows) % 2 == 0) else second
        row = np.array(center, copy=True)
        dim = priority_indices[int(rng.integers(0, len(priority_indices)))]
        step_fraction = 0.018 if len(rows) % 3 else 0.035
        sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        row[dim] += sign * step_fraction * encoded_span[dim]
        rows.append(np.clip(row, encoded_lo, encoded_hi))

    return np.asarray(rows, dtype=float)


def _rows_to_candidates(
    rows: np.ndarray,
    search_space: Sequence[ParameterSpec],
    *,
    candidate_ids: Sequence[str],
    method: str,
    stage: str,
    batch_name: str,
) -> list[dict[str, Any]]:
    candidates = []
    for candidate_id, row in zip(candidate_ids, rows):
        params = {}
        for coord, spec in zip(np.asarray(row, dtype=float), search_space):
            params[spec.path] = spec.decode(float(coord))
        params["optimizer_candidate_id"] = str(candidate_id)
        params["optimizer_method"] = str(method)
        params["optimizer_stage"] = str(stage)
        params["optimizer_batch_name"] = str(batch_name)
        candidates.append(params)
    return candidates


def _encoded_row_signature(row: Sequence[float], *, digits: int = 10) -> tuple[float, ...]:
    return tuple(round(float(value), int(digits)) for value in np.asarray(row, dtype=float))


def _archive_encoded_signatures(
    archive_rows: Sequence[dict[str, Any]],
    search_space: Sequence[ParameterSpec],
) -> set[tuple[float, ...]]:
    signatures: set[tuple[float, ...]] = set()
    for row in archive_rows:
        params = row.get("parameters")
        if not isinstance(params, dict):
            continue
        try:
            signatures.add(_encoded_row_signature(_candidate_vector(params, search_space)))
        except Exception:
            continue
    return signatures


def _deduplicate_candidate_rows(
    rows: np.ndarray,
    *,
    used_signatures: set[tuple[float, ...]],
    n_candidates: int,
    rng: np.random.Generator,
    search_space: Sequence[ParameterSpec],
    fallback_centers: np.ndarray,
) -> tuple[np.ndarray, int]:
    encoded_lo = np.asarray([spec.low_encoded() for spec in search_space], dtype=float)
    encoded_hi = np.asarray([spec.high_encoded() for spec in search_space], dtype=float)
    encoded_span = encoded_hi - encoded_lo
    selected: list[np.ndarray] = []
    selected_signatures: set[tuple[float, ...]] = set()
    dropped = 0

    def add_row(row: np.ndarray, *, count_drop: bool = False) -> bool:
        nonlocal dropped
        clipped = np.clip(np.asarray(row, dtype=float), encoded_lo, encoded_hi)
        signature = _encoded_row_signature(clipped)
        if signature in used_signatures or signature in selected_signatures:
            if count_drop:
                dropped += 1
            return False
        selected.append(clipped)
        selected_signatures.add(signature)
        return True

    for row in np.asarray(rows, dtype=float):
        if len(selected) >= int(n_candidates):
            break
        if add_row(row):
            continue
        dropped += 1
        for scale_fraction in (0.012, 0.024, 0.040, 0.065):
            jittered = np.asarray(row, dtype=float) + rng.normal(
                loc=0.0,
                scale=np.maximum(float(scale_fraction) * encoded_span, 1e-6),
            )
            if add_row(jittered):
                break

    centers = np.asarray(fallback_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] == 0:
        centers = np.empty((0, len(search_space)), dtype=float)
    attempts = 0
    max_attempts = max(256, int(n_candidates) * 256)
    local_sigma = np.maximum(0.06 * encoded_span, 1e-6)
    while len(selected) < int(n_candidates) and attempts < max_attempts:
        attempts += 1
        if centers.size and float(rng.random()) < 0.75:
            center = centers[int(rng.integers(0, centers.shape[0]))]
            scale = local_sigma * (1.0 + 0.2 * (attempts // max(1, int(n_candidates))))
            row = rng.normal(loc=center, scale=scale)
        else:
            row = rng.uniform(encoded_lo, encoded_hi)
        add_row(row)

    # The parameter spaces used here are continuous, so this should almost never
    # trigger. Keep a last-resort path so campaign planning cannot fail because a
    # narrow local basin was already fully sampled.
    while len(selected) < int(n_candidates):
        row = rng.uniform(encoded_lo, encoded_hi)
        signature = _encoded_row_signature(row, digits=14)
        if signature in selected_signatures:
            continue
        selected.append(np.clip(row, encoded_lo, encoded_hi))
        selected_signatures.add(signature)

    return np.asarray(selected[: int(n_candidates)], dtype=float), int(dropped)


def _next_candidate_ids(state: dict[str, Any], n: int) -> list[str]:
    start = int(state["next_candidate_index"])
    return [f"C{index:05d}" for index in range(start, start + int(n))]


def propose_lhs_batch(
    campaign_dir: str | Path,
    *,
    search_space: Sequence[ParameterSpec],
    n_candidates: int,
    seed: int | None = None,
    stage: str = "screen",
    method: str = "latin_hypercube",
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    state = load_campaign_state(campaign_dir)
    batch_name = _batch_name(state)
    candidate_ids = _next_candidate_ids(state, n_candidates)
    samples = _sample_unit_lhs(int(n_candidates), len(search_space), seed=seed)
    params = _decode_unit_samples(samples, search_space)
    candidates = []
    for candidate_id, param_row in zip(candidate_ids, params):
        payload = dict(param_row)
        payload["optimizer_candidate_id"] = candidate_id
        payload["optimizer_method"] = method
        payload["optimizer_stage"] = stage
        payload["optimizer_batch_name"] = batch_name
        candidates.append(payload)

    batch_plan = {
        "batch_name": batch_name,
        "strategy": method,
        "stage": stage,
        "seed": seed,
        "candidate_ids": candidate_ids,
        "candidates": candidates,
    }
    _write_json(campaign_dir / "batches" / f"{batch_name}_plan.json", batch_plan)

    state["next_batch_index"] = int(state["next_batch_index"]) + 1
    state["next_candidate_index"] = int(state["next_candidate_index"]) + int(n_candidates)
    _write_campaign_state(campaign_dir, state)
    return batch_plan


def load_candidate_archive_rows(campaign_dir: str | Path) -> list[dict[str, Any]]:
    campaign_path = Path(campaign_dir)
    path = _archive_path(campaign_path, kind="candidate")
    if not path.exists():
        return []
    objective_filter = load_objective_filter(campaign_path)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = rescore_candidate_row(json.loads(line))
        if _row_matches_objective_filter(row, objective_filter):
            rows.append(row)
    return rows


def load_item_archive_rows(campaign_dir: str | Path) -> list[dict[str, Any]]:
    path = _archive_path(Path(campaign_dir), kind="item")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def propose_elite_batch(
    campaign_dir: str | Path,
    *,
    search_space: Sequence[ParameterSpec],
    n_candidates: int,
    seed: int | None = None,
    elite_frac: float = 0.25,
    explore_frac: float = 0.25,
    stage: str = "refine",
    method: str = "elite_refine",
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    archive = load_candidate_archive_rows(campaign_dir)
    valid = [row for row in archive if np.isfinite(float(row.get("pair_score", np.nan)))]
    if len(valid) < 4:
        return propose_lhs_batch(
            campaign_dir,
            search_space=search_space,
            n_candidates=n_candidates,
            seed=seed,
            stage=stage,
            method="latin_hypercube_fallback",
        )

    ranked = sorted(valid, key=lambda row: float(row["pair_score"]), reverse=True)
    elite_count = min(max(4, int(math.ceil(float(elite_frac) * len(ranked)))), 12, len(ranked))
    elite = ranked[:elite_count]
    elite_vectors = np.vstack([_candidate_vector(row["parameters"], search_space) for row in elite])
    mean = elite_vectors.mean(axis=0)
    cov = np.cov(elite_vectors.T)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=float)
    diag_jitter = np.diag(
        [
            max((spec.high_encoded() - spec.low_encoded()) * 0.04, 1e-6) ** 2
            for spec in search_space
        ]
    )
    cov = np.asarray(cov, dtype=float) + diag_jitter

    rng = np.random.default_rng(seed)
    encoded_lo = np.asarray([spec.low_encoded() for spec in search_space], dtype=float)
    encoded_hi = np.asarray([spec.high_encoded() for spec in search_space], dtype=float)
    encoded_span = encoded_hi - encoded_lo

    total_n = int(n_candidates)
    explore_n = min(max(0, int(round(float(explore_frac) * total_n))), max(total_n - 1, 0))
    if len(valid) >= 128:
        explore_n = min(explore_n, max(0, int(round(0.25 * total_n))))
    targeted_n = 0
    targeted_mode = "none"
    objective_filter = load_objective_filter(campaign_dir)
    early_frontier_after_objective_pivot = bool(objective_filter.get("target_hfo_hz")) and len(valid) >= 192
    if early_frontier_after_objective_pivot and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(12, int(round(0.75 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "frontier"
    elif len(valid) >= 448 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(12, int(round(0.75 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "frontier"
    elif len(valid) >= 416 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(12, int(round(0.75 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "basin"
    elif len(valid) >= 368 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(12, int(round(0.75 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "needle"
    elif len(valid) >= 320 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(10, int(round(0.70 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "ridge"
    elif len(valid) >= 288 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(10, int(round(0.75 * total_n))), max(total_n - explore_n - 1, 0))
        targeted_mode = "micro"
    elif len(valid) >= 256 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(8, int(round(0.60 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "combo"
    elif len(valid) >= 224 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.075 * total_n))))
        targeted_n = min(max(6, int(round(0.50 * total_n))), max(total_n - explore_n - 3, 0))
        targeted_mode = "stencil"
    elif len(valid) >= 192 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.125 * total_n))))
        targeted_n = min(max(2, int(round(0.25 * total_n))), max(total_n - explore_n - 2, 0))
        targeted_mode = "line"
    remaining_n = total_n - explore_n - targeted_n
    local_fraction = 0.70 if targeted_n > 0 else 0.55
    local_n = min(max(1, int(round(local_fraction * remaining_n))), remaining_n) if remaining_n > 0 else 0
    covariance_n = max(0, total_n - explore_n - targeted_n - local_n)

    local_source_count = min(4, len(elite))
    local_centers = elite_vectors[:local_source_count]
    raw_weights = np.asarray(
        [max(float(row.get("pair_score", 0.0)), 0.0) + 1e-6 for row in elite[:local_source_count]],
        dtype=float,
    )
    local_weights = raw_weights / raw_weights.sum()
    tight_local_n = min(local_n, max(1, int(round(0.50 * local_n)))) if local_n > 0 else 0
    broad_local_n = max(0, local_n - tight_local_n)

    tight_local_sigma = np.maximum(0.035 * encoded_span, 1e-6)
    broad_local_sigma = np.maximum(0.10 * encoded_span, 1e-6)
    local_rows = []
    if tight_local_n > 0:
        tight_source_count = min(2, local_source_count)
        tight_weights = local_weights[:tight_source_count]
        tight_weights = tight_weights / tight_weights.sum()
        for _ in range(tight_local_n):
            center = local_centers[int(rng.choice(tight_source_count, p=tight_weights))]
            row = rng.normal(loc=center, scale=tight_local_sigma)
            local_rows.append(np.clip(row, encoded_lo, encoded_hi))
    for _ in range(broad_local_n):
        center = local_centers[int(rng.choice(local_source_count, p=local_weights))]
        row = rng.normal(loc=center, scale=broad_local_sigma)
        local_rows.append(np.clip(row, encoded_lo, encoded_hi))
    local_rows = (
        np.asarray(local_rows, dtype=float)
        if local_rows
        else np.empty((0, len(search_space)), dtype=float)
    )

    covariance_rows = _sample_truncated_gaussian(
        mean,
        cov,
        search_space,
        covariance_n,
        seed=seed,
    ) if covariance_n > 0 else np.empty((0, len(search_space)), dtype=float)
    targeted_rows = _targeted_elite_probe_rows(
        elite_vectors,
        search_space,
        targeted_n,
        mode=targeted_mode,
        seed=None if seed is None else seed + 2,
        archive_rows=ranked,
    )

    if explore_n > 0:
        explore_rows = np.vstack(
            [_candidate_vector(row, search_space) for row in _decode_unit_samples(_sample_unit_lhs(explore_n, len(search_space), seed=None if seed is None else seed + 1), search_space)]
        )
        all_rows = np.vstack([targeted_rows, local_rows, covariance_rows, explore_rows])
    else:
        all_rows = np.vstack([targeted_rows, local_rows, covariance_rows])
    used_signatures = _archive_encoded_signatures(ranked, search_space)
    all_rows, duplicate_rows_dropped = _deduplicate_candidate_rows(
        all_rows,
        used_signatures=used_signatures,
        n_candidates=total_n,
        rng=rng,
        search_space=search_space,
        fallback_centers=elite_vectors[: min(6, len(elite_vectors))],
    )

    state = load_campaign_state(campaign_dir)
    batch_name = _batch_name(state)
    candidate_ids = _next_candidate_ids(state, n_candidates)
    candidates = _rows_to_candidates(
        all_rows,
        search_space,
        candidate_ids=candidate_ids,
        method=method,
        stage=stage,
        batch_name=batch_name,
    )
    batch_plan = {
        "batch_name": batch_name,
        "strategy": method,
        "stage": stage,
        "seed": seed,
        "candidate_ids": candidate_ids,
        "candidates": candidates,
        "elite_source_ids": [row["candidate_id"] for row in elite],
        "local_source_ids": [row["candidate_id"] for row in elite[:local_source_count]],
        "proposal_counts": {
            "targeted": int(targeted_n),
            "local": int(local_n),
            "covariance": int(covariance_n),
            "explore": int(explore_n),
        },
        "local_detail_counts": {
            "tight_top": int(tight_local_n),
            "broad_weighted": int(broad_local_n),
        },
        "targeted_detail": {
            "top_pair": [row["candidate_id"] for row in elite[:2]],
            "mode": targeted_mode,
            "line_probe_count": int(min(targeted_n, 3) if targeted_mode == "line" else 0),
            "coordinate_probe_count": int(max(targeted_n - 3, 0) if targeted_mode == "line" else targeted_n),
            "archive_duplicate_rows_dropped": int(duplicate_rows_dropped),
        },
    }
    _write_json(campaign_dir / "batches" / f"{batch_name}_plan.json", batch_plan)

    state["next_batch_index"] = int(state["next_batch_index"]) + 1
    state["next_candidate_index"] = int(state["next_candidate_index"]) + int(n_candidates)
    _write_campaign_state(campaign_dir, state)
    return batch_plan


def _joint_sweep_paths_for_batch(
    batch_plan: dict[str, Any],
    *,
    ketamine_block_values: dict[str, float] | None = None,
) -> dict[str, list[Any]]:
    ketamine_block_values = dict(ketamine_block_values or {"control": 1.0, "ketamine": 0.0})
    paths: dict[str, list[Any]] = {}
    for candidate in batch_plan["candidates"]:
        for condition_name, ketamine_block in ketamine_block_values.items():
            item_values = dict(candidate)
            item_values["ketamine_block"] = float(ketamine_block)
            item_values["optimizer_condition"] = str(condition_name)
            item_values["optimizer_pair_id"] = str(candidate["optimizer_candidate_id"])
            for key, value in item_values.items():
                paths.setdefault(key, []).append(value)
    return paths


def _switch_sweep_paths_for_batch(
    batch_plan: dict[str, Any],
    *,
    switch_time_ms: float,
    switch_washout_ms: float,
    ketamine_block_values: dict[str, float] | None = None,
) -> dict[str, list[Any]]:
    ketamine_block_values = dict(ketamine_block_values or {"control": 1.0, "ketamine": 0.0})
    before_block = float(ketamine_block_values.get("control", 1.0))
    after_block = float(ketamine_block_values.get("ketamine", 0.0))
    paths: dict[str, list[Any]] = {}
    for candidate in batch_plan["candidates"]:
        item_values = dict(candidate)
        item_values["ketamine_block"] = before_block
        item_values["ketamine_switch_time_ms"] = float(switch_time_ms)
        item_values["ketamine_block_after_switch"] = after_block
        item_values["ketamine_switch_washout_ms"] = float(switch_washout_ms)
        item_values["optimizer_condition"] = "switch"
        item_values["optimizer_pair_id"] = str(candidate["optimizer_candidate_id"])
        for key, value in item_values.items():
            paths.setdefault(key, []).append(value)
    return paths


def run_hfo_batch(
    campaign_dir: str | Path,
    *,
    base_config: dict[str, Any],
    batch_plan: dict[str, Any],
    ketamine_block_values: dict[str, float] | None = None,
    condition_mode: str = "separate",
    ketamine_switch_time_ms: float | None = None,
    ketamine_switch_washout_ms: float = 500.0,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    configured_condition_mode = base_config.get(
        "hfo_condition_mode",
        base_config.get("optimizer_condition_mode"),
    )
    if condition_mode == "separate" and configured_condition_mode is not None:
        condition_mode = str(configured_condition_mode)
    configured_switch_time_ms = base_config.get(
        "hfo_ketamine_switch_time_ms",
        base_config.get("ketamine_switch_time_ms"),
    )
    configured_switch_washout_ms = base_config.get(
        "hfo_ketamine_switch_washout_ms",
        base_config.get("ketamine_switch_washout_ms"),
    )
    if ketamine_switch_time_ms is None and configured_switch_time_ms is not None:
        ketamine_switch_time_ms = float(configured_switch_time_ms)
    if configured_switch_washout_ms is not None:
        ketamine_switch_washout_ms = float(configured_switch_washout_ms)
    condition_mode = str(condition_mode)
    if condition_mode == "separate":
        sweep_path = _joint_sweep_paths_for_batch(batch_plan, ketamine_block_values=ketamine_block_values)
    elif condition_mode == "switch":
        tstop_ms = float(base_config.get("tstop_ms") or 0.0)
        switch_time_ms = float(
            ketamine_switch_time_ms
            if ketamine_switch_time_ms is not None
            else max(tstop_ms * 0.5, 0.0)
        )
        sweep_path = _switch_sweep_paths_for_batch(
            batch_plan,
            switch_time_ms=switch_time_ms,
            switch_washout_ms=float(ketamine_switch_washout_ms),
            ketamine_block_values=ketamine_block_values,
        )
    else:
        raise ValueError("condition_mode must be 'separate' or 'switch'")
    config = dict(base_config)
    config["label_prefix"] = f"hfo_opt_{batch_plan['batch_name']}"
    sweep = hlp.run_parameter_sweep(config, sweep_path)
    sweep_dir = Path(sweep["sweep_dir"])
    metadata = {
        "batch_name": batch_plan["batch_name"],
        "strategy": batch_plan["strategy"],
        "stage": batch_plan["stage"],
        "condition_mode": condition_mode,
        "ketamine_switch_time_ms": (
            None if condition_mode != "switch" else float(sweep_path["ketamine_switch_time_ms"][0])
        ),
        "ketamine_switch_washout_ms": (
            None if condition_mode != "switch" else float(sweep_path["ketamine_switch_washout_ms"][0])
        ),
        "sweep_dir": str(sweep_dir),
        "item_count": len(sweep.get("items", [])),
    }
    _write_json(campaign_dir / "batches" / f"{batch_plan['batch_name']}_run.json", metadata)
    return sweep


def mean_firing_rates_by_type(result: dict[str, Any]) -> dict[str, float]:
    summary = result.get("summary") or {}
    params = summary.get("params") or {}
    tstop_ms = float(params.get("tstop") or 0.0)
    duration_s = max(tstop_ms / 1000.0, 1e-9)
    soma_spikes = result.get("soma_spikes") or {}
    labels = list(soma_spikes.get("labels") or [])
    spike_times = list(soma_spikes.get("spike_times") or [])
    grouped_counts: dict[str, list[float]] = {}
    for label, spikes in zip(labels, spike_times):
        cell_type = hlp.cell_type_of(label)
        grouped_counts.setdefault(cell_type, []).append(float(len(spikes)) / duration_s)
    return {
        cell_type: float(np.mean(rates)) if rates else 0.0
        for cell_type, rates in grouped_counts.items()
    }


def input_coverage_fraction(result: dict[str, Any]) -> float:
    """Estimate how much of the simulated interval has afferent input support."""
    summary = result.get("summary") or {}
    params = summary.get("params") or {}
    tstop_ms = float(params.get("tstop") or 0.0)
    if tstop_ms <= 0.0:
        lfp_t = result.get("lfp_t")
        if lfp_t is not None and len(lfp_t) > 0:
            tstop_ms = float(np.max(lfp_t))
    if tstop_ms <= 0.0:
        return 0.0

    max_event_ms = 0.0
    for _label, times in result.get("input_times", []) or []:
        values = np.asarray(times, dtype=float)
        if len(values):
            max_event_ms = max(max_event_ms, float(np.max(values)))
    return min(max(max_event_ms / tstop_ms, 0.0), 1.0)


def score_condition_result(
    result: dict[str, Any],
    *,
    signal: str = "lfp",
    dt_ms: float = 0.1,
    target_hz: float = 195.0,
    target_half_width_hz: float = 35.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    bands = dict(bands or DEFAULT_SCORE_BANDS)
    summary = hlp.compute_hfo_power_summary(
        result,
        signal=signal,
        bands=bands,
        dt_ms=dt_ms,
        relative_band=(15.0, 250.0),
    )
    freqs = np.asarray(summary["freqs"], dtype=float)
    psd = np.asarray(summary["psd"], dtype=float)

    target_lo, target_hi, target_hz, target_half_width_hz = _target_band_bounds(
        bands,
        target_hz=target_hz,
        target_half_width_hz=target_half_width_hz,
    )
    if len(freqs) == 0:
        return {
            "condition_score": float("-inf"),
            "peak_hz": math.nan,
            "peak_ratio": 0.0,
            "target_peak_contrast": 0.0,
            "target_density_ratio": 0.0,
            "freq_match": 0.0,
            "target_clean_fraction": 0.0,
            "supra_hfo_relative": 0.0,
            "phase_lock": 0.0,
            "rate_penalty": 0.0,
            "spike_support_rate_hz": 0.0,
            "spike_support_penalty": 0.0,
            "input_coverage_fraction": 0.0,
            "input_dropout_penalty": 0.0,
            "psd_shape_freqs_hz": list(PSD_TEMPLATE_FREQS_HZ),
            "psd_shape_power": [0.0 for _ in PSD_TEMPLATE_FREQS_HZ],
            "band_power": summary["band_power"],
            "relative_band_power": summary["relative_band_power"],
            "mean_firing_rate_by_type": {},
        }

    target_mask = (freqs >= target_lo) & (freqs <= target_hi)
    target_freqs = freqs[target_mask]
    target_psd = psd[target_mask]
    broad_mask = (freqs >= 15.0) & (freqs <= 250.0)
    if not np.any(target_mask):
        peak_hz = math.nan
        peak_power = 0.0
    else:
        local_index = int(np.argmax(target_psd))
        peak_hz = float(target_freqs[local_index])
        peak_power = float(target_psd[local_index])

    background_mask = broad_mask & ~target_mask
    shoulder_mask = ((freqs >= 100.0) & (freqs < target_lo)) | ((freqs > target_hi) & (freqs <= 240.0))
    background_floor = float(np.median(psd[background_mask])) if np.any(background_mask) else 0.0
    shoulder_floor = float(np.median(psd[shoulder_mask])) if np.any(shoulder_mask) else background_floor
    target_floor = float(np.median(target_psd)) if np.any(target_mask) else 0.0
    target_power = float(summary["band_power"].get("target_hfo", 0.0))
    target_width_hz = max(target_hi - target_lo, 1e-9)
    target_density = target_power / target_width_hz
    background_power = (
        float(np.trapezoid(psd[background_mask], freqs[background_mask]))
        if np.any(background_mask)
        else 0.0
    )
    background_width_hz = float(np.ptp(freqs[background_mask])) if np.count_nonzero(background_mask) > 1 else 0.0
    background_density = background_power / max(background_width_hz, 1e-9)
    denom = max(background_density, background_floor, shoulder_floor, 1e-18)
    target_density_ratio = target_density / denom
    peak_ratio = peak_power / max(background_floor, shoulder_floor, 1e-18)
    target_peak_contrast = peak_power / max(target_floor, shoulder_floor, background_floor, 1e-18)

    freq_match = math.exp(-0.5 * ((peak_hz - target_hz) / max(float(target_half_width_hz), 1e-9)) ** 2) if np.isfinite(peak_hz) else 0.0
    if np.any(target_mask):
        target_weights = np.maximum(target_psd - min(background_floor, shoulder_floor), 0.0)
        if float(np.sum(target_weights)) <= 1e-18:
            target_weights = np.maximum(target_psd, 0.0)
        if float(np.sum(target_weights)) > 1e-18:
            target_centroid_hz = float(np.sum(target_freqs * target_weights) / np.sum(target_weights))
        else:
            target_centroid_hz = float(peak_hz)
    else:
        target_centroid_hz = math.nan
    target_centroid_match = (
        math.exp(-0.5 * ((target_centroid_hz - target_hz) / max(float(target_half_width_hz), 1e-9)) ** 2)
        if np.isfinite(target_centroid_hz)
        else 0.0
    )
    relative_target = float(summary["relative_band_power"].get("target_hfo", 0.0))
    lower_side_power = (
        float(summary["band_power"].get("hfo_80_130", 0.0))
        + float(summary["band_power"].get("hfo_130_160", 0.0))
    )
    supra_power = float(summary["band_power"].get("supra_hfo", 0.0))
    side_power = lower_side_power + supra_power
    dominance = target_power / max(side_power, 1e-18)
    target_clean_fraction = target_power / max(target_power + side_power, 1e-18)
    supra_hfo_relative = float(summary["relative_band_power"].get("supra_hfo", 0.0))
    beta_gamma = (
        float(summary["relative_band_power"].get("beta", 0.0))
        + float(summary["relative_band_power"].get("low_gamma", 0.0))
        + float(summary["relative_band_power"].get("high_gamma", 0.0))
    )
    try:
        phase = hlp.compute_spike_phase_locking(
            result,
            signal=signal,
            band=(target_lo, target_hi),
            cell_types=("TC", "MC", "EPLI"),
            dt_ms=dt_ms,
        )
        phase_lock = float(phase.get("vector_strength", 0.0))
    except KeyError:
        phase_lock = 0.0
    mean_rates = mean_firing_rates_by_type(result)
    epli_rate = mean_rates.get("EPLI", 0.0) + mean_rates.get("PVCRH", 0.0)

    rate_penalty = 0.0
    rate_penalty += max(mean_rates.get("TC", 0.0) - 120.0, 0.0) / 60.0
    rate_penalty += max(mean_rates.get("MC", 0.0) - 80.0, 0.0) / 40.0
    rate_penalty += max(mean_rates.get("EPLI", 0.0) - 250.0, 0.0) / 100.0
    spike_support_rate = (
        mean_rates.get("MC", 0.0)
        + mean_rates.get("TC", 0.0)
        + mean_rates.get("EPLI", 0.0)
        + mean_rates.get("PVCRH", 0.0)
    )
    spike_support_penalty = 2.0 * max(1.0 - min(spike_support_rate / 5.0, 1.0), 0.0)
    input_coverage = input_coverage_fraction(result)
    input_dropout_penalty = 3.0 * max(0.85 - input_coverage, 0.0)

    condition_score = (
        2.4 * math.log10(1.0 + target_density_ratio)
        + 4.0 * relative_target
        + 1.0 * math.log10(1.0 + target_peak_contrast)
        + 2.2 * math.log10(1.0 + dominance)
        + 1.2 * target_clean_fraction
        + 0.8 * target_centroid_match
        + 0.5 * min(beta_gamma, 0.30)
        + 0.5 * phase_lock
        - 1.5 * supra_hfo_relative
        - rate_penalty
        - spike_support_penalty
        - input_dropout_penalty
    )
    return {
        "condition_score": float(condition_score),
        "peak_hz": float(peak_hz),
        "peak_ratio": float(peak_ratio),
        "target_peak_contrast": float(target_peak_contrast),
        "target_density_ratio": float(target_density_ratio),
        "freq_match": float(freq_match),
        "target_centroid_hz": float(target_centroid_hz),
        "target_centroid_match": float(target_centroid_match),
        "dominance": float(dominance),
        "target_clean_fraction": float(target_clean_fraction),
        "supra_hfo_relative": float(supra_hfo_relative),
        "beta_gamma_support": float(beta_gamma),
        "phase_lock": float(phase_lock),
        "rate_penalty": float(rate_penalty),
        "spike_support_rate_hz": float(spike_support_rate),
        "epli_rate_hz": float(epli_rate),
        "spike_support_penalty": float(spike_support_penalty),
        "input_coverage_fraction": float(input_coverage),
        "input_dropout_penalty": float(input_dropout_penalty),
        "target_band_hz": [float(target_lo), float(target_hi)],
        "target_center_hz": float(target_hz),
        "psd_shape_freqs_hz": list(PSD_TEMPLATE_FREQS_HZ),
        "psd_shape_power": [float(value) for value in _psd_shape_from_arrays(freqs, psd)],
        "band_power": {key: float(value) for key, value in summary["band_power"].items()},
        "relative_band_power": {key: float(value) for key, value in summary["relative_band_power"].items()},
        "mean_firing_rate_by_type": {key: float(value) for key, value in mean_rates.items()},
    }


def score_candidate_pair(
    *,
    control_metrics: dict[str, Any] | None,
    ketamine_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    control_metrics = dict(control_metrics or {})
    ketamine_metrics = dict(ketamine_metrics or {})
    control_score = float(control_metrics.get("condition_score", 0.0))
    ketamine_score = float(ketamine_metrics.get("condition_score", 0.0))
    if not (math.isfinite(control_score) and math.isfinite(ketamine_score)):
        return {
            "pair_score": float("-inf"),
            "target_contrast_log10": float("-inf"),
            "peak_contrast_log10": float("-inf"),
            "control_score": float(control_score),
            "ketamine_score": float(ketamine_score),
        }
    control_target = float((control_metrics.get("relative_band_power") or {}).get("target_hfo", 0.0))
    ketamine_target = float((ketamine_metrics.get("relative_band_power") or {}).get("target_hfo", 0.0))
    control_ratio = _target_density_ratio(control_metrics)
    ketamine_ratio = _target_density_ratio(ketamine_metrics)
    control_peak_contrast = float(control_metrics.get("target_peak_contrast", control_metrics.get("peak_ratio", 0.0)))
    ketamine_peak_contrast = float(ketamine_metrics.get("target_peak_contrast", ketamine_metrics.get("peak_ratio", 0.0)))
    control_supra = _supra_hfo_relative(control_metrics)
    ketamine_supra = _supra_hfo_relative(ketamine_metrics)
    control_clean = _target_clean_fraction(control_metrics)
    ketamine_clean = _target_clean_fraction(ketamine_metrics)
    control_input_dropout = float(control_metrics.get("input_dropout_penalty", 0.0))
    ketamine_input_dropout = float(ketamine_metrics.get("input_dropout_penalty", 0.0))
    control_peak_hz = float(control_metrics.get("peak_hz", math.nan))
    ketamine_peak_hz = float(ketamine_metrics.get("peak_hz", math.nan))
    control_center_match = float(control_metrics.get("target_centroid_match", control_metrics.get("freq_match", 0.0)))
    ketamine_center_match = float(ketamine_metrics.get("target_centroid_match", ketamine_metrics.get("freq_match", 0.0)))
    control_epli_rate = float(control_metrics.get("epli_rate_hz", (control_metrics.get("mean_firing_rate_by_type") or {}).get("EPLI", 0.0)))
    ketamine_epli_rate = float(ketamine_metrics.get("epli_rate_hz", (ketamine_metrics.get("mean_firing_rate_by_type") or {}).get("EPLI", 0.0)))
    target_lo_hz, target_hi_hz = _target_band_for_pair(control_metrics, ketamine_metrics)
    psd_template_metrics = _psd_template_pair_metrics(control_metrics, ketamine_metrics)

    target_contrast = math.log10((ketamine_target + 1e-12) / (control_target + 1e-12))
    density_contrast = math.log10((ketamine_ratio + 1e-12) / (control_ratio + 1e-12))
    peak_contrast = density_contrast
    compound_contrast = math.log10(
        ((ketamine_target * ketamine_ratio) + 1e-12)
        / ((control_target * control_ratio) + 1e-12)
    )
    target_delta = ketamine_target - control_target
    supra_delta = ketamine_supra - control_supra
    clean_delta = ketamine_clean - control_clean
    control_leak_penalty = (
        18.0 * control_target
        + 12.0 * control_supra
        + 0.8 * max(control_score, 0.0)
        + control_input_dropout
    )
    control_target_excess_penalty = 35.0 * max(control_target - 0.12, 0.0)
    same_peak_penalty = 0.0
    if (
        math.isfinite(control_peak_hz)
        and math.isfinite(ketamine_peak_hz)
        and target_lo_hz <= control_peak_hz <= target_hi_hz
        and target_lo_hz <= ketamine_peak_hz <= target_hi_hz
        and abs(control_peak_hz - ketamine_peak_hz) <= 5.0
    ):
        same_peak_penalty = 4.0 + 10.0 * control_target
    negative_delta_penalty = 25.0 * max(-target_delta, 0.0)
    ketamine_center_penalty = 1.5 * max(0.70 - ketamine_center_match, 0.0)
    control_center_advantage_penalty = 1.0 * max(control_center_match - ketamine_center_match, 0.0)
    ketamine_peak_contrast_penalty = 1.5 * max(1.75 - ketamine_peak_contrast, 0.0)
    control_peak_contrast_penalty = 0.75 * max(control_peak_contrast - ketamine_peak_contrast, 0.0)
    ketamine_epli_silence_penalty = 2.0 * max(2.0 - ketamine_epli_rate, 0.0) / 2.0
    epli_dropout_penalty = 0.5 * max(control_epli_rate - ketamine_epli_rate, 0.0) / 5.0
    ketamine_wrong_band_penalty = (
        8.0 * max(ketamine_supra - 0.45 * max(ketamine_target, 1e-12), 0.0)
        + 3.0 * max(0.45 - ketamine_clean, 0.0)
        + ketamine_input_dropout
    )
    control_wrong_band_penalty = 3.0 * max(control_supra - control_target, 0.0)
    ketamine_freq_match = 1.0 if ketamine_target > 0.0 else 0.0
    pair_score = (
        ketamine_score
        + 4.0 * compound_contrast
        + 18.0 * target_delta
        + 3.0 * clean_delta
        + 1.2 * math.log10(1.0 + ketamine_peak_contrast)
        - 0.8 * math.log10(1.0 + control_peak_contrast)
        + 1.5 * max(-supra_delta, 0.0)
        + 0.8 * (ketamine_center_match - control_center_match)
        + 2.0 * psd_template_metrics["psd_template_score"]
        - control_leak_penalty
        - control_target_excess_penalty
        - same_peak_penalty
        - negative_delta_penalty
        - ketamine_center_penalty
        - control_center_advantage_penalty
        - ketamine_peak_contrast_penalty
        - control_peak_contrast_penalty
        - ketamine_epli_silence_penalty
        - epli_dropout_penalty
        - 1.5 * psd_template_metrics["psd_template_loss"]
        - ketamine_wrong_band_penalty
        - control_wrong_band_penalty
    )
    return {
        "pair_score": float(pair_score),
        "pair_score_version": PAIR_SCORE_VERSION,
        "target_contrast_log10": float(target_contrast),
        "peak_contrast_log10": float(peak_contrast),
        "density_contrast_log10": float(density_contrast),
        "compound_contrast_log10": float(compound_contrast),
        "target_delta": float(target_delta),
        "supra_delta": float(supra_delta),
        "target_clean_delta": float(clean_delta),
        "control_leak_penalty": float(control_leak_penalty),
        "control_target_excess_penalty": float(control_target_excess_penalty),
        "same_peak_penalty": float(same_peak_penalty),
        "negative_delta_penalty": float(negative_delta_penalty),
        "ketamine_center_penalty": float(ketamine_center_penalty),
        "control_center_advantage_penalty": float(control_center_advantage_penalty),
        "ketamine_center_match": float(ketamine_center_match),
        "control_center_match": float(control_center_match),
        "ketamine_peak_contrast_penalty": float(ketamine_peak_contrast_penalty),
        "control_peak_contrast_penalty": float(control_peak_contrast_penalty),
        "ketamine_peak_contrast": float(ketamine_peak_contrast),
        "control_peak_contrast": float(control_peak_contrast),
        "ketamine_epli_silence_penalty": float(ketamine_epli_silence_penalty),
        "epli_dropout_penalty": float(epli_dropout_penalty),
        "ketamine_epli_rate_hz": float(ketamine_epli_rate),
        "control_epli_rate_hz": float(control_epli_rate),
        "ketamine_wrong_band_penalty": float(ketamine_wrong_band_penalty),
        "control_wrong_band_penalty": float(control_wrong_band_penalty),
        **psd_template_metrics,
        "ketamine_freq_match": float(ketamine_freq_match),
        "control_score": float(control_score),
        "ketamine_score": float(ketamine_score),
    }


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(hlp._json_ready(row), sort_keys=True) + "\n")


def _empty_condition_metrics() -> dict[str, Any]:
    return {
        "condition_score": float("-inf"),
        "peak_hz": math.nan,
        "peak_ratio": 0.0,
        "target_peak_contrast": 0.0,
        "target_density_ratio": 0.0,
        "freq_match": 0.0,
        "target_centroid_hz": math.nan,
        "target_centroid_match": 0.0,
        "dominance": 0.0,
        "target_clean_fraction": 0.0,
        "supra_hfo_relative": 0.0,
        "beta_gamma_support": 0.0,
        "phase_lock": 0.0,
        "rate_penalty": 0.0,
        "spike_support_rate_hz": 0.0,
        "epli_rate_hz": 0.0,
        "spike_support_penalty": 0.0,
        "input_coverage_fraction": 0.0,
        "input_dropout_penalty": 0.0,
        "psd_shape_freqs_hz": list(PSD_TEMPLATE_FREQS_HZ),
        "psd_shape_power": [0.0 for _ in PSD_TEMPLATE_FREQS_HZ],
        "band_power": {},
        "relative_band_power": {},
        "mean_firing_rate_by_type": {},
    }


def _result_tstop_ms(result: dict[str, Any] | None) -> float:
    if result is None:
        return 0.0
    summary = result.get("summary") or {}
    params = summary.get("params") or {}
    try:
        tstop_ms = float(params.get("tstop") or 0.0)
    except (TypeError, ValueError):
        tstop_ms = 0.0
    if tstop_ms <= 0.0:
        lfp_t = result.get("lfp_t")
        if lfp_t is not None and len(lfp_t) > 0:
            tstop_ms = float(np.max(np.asarray(lfp_t, dtype=float)))
    return max(tstop_ms, 0.0)


def _windowed_times(times: Any, start_ms: float, stop_ms: float) -> np.ndarray:
    values = np.atleast_1d(np.asarray(times, dtype=float))
    if len(values) == 0:
        return values
    mask = (values >= float(start_ms)) & (values < float(stop_ms))
    return values[mask] - float(start_ms)


def _windowed_trace(
    times: Any,
    values: Any,
    start_ms: float,
    stop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.atleast_1d(np.asarray(times, dtype=float))
    y = np.atleast_1d(np.asarray(values, dtype=float))
    if len(t) == 0 or len(y) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(len(t), len(y))
    t = t[:n]
    y = y[:n]
    mask = (t >= float(start_ms)) & (t < float(stop_ms))
    return t[mask] - float(start_ms), y[mask]


def _windowed_event_rows(rows: Any, start_ms: float, stop_ms: float) -> list[Any]:
    result = []
    for row in rows or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            row_values = list(row)
            row_values[1] = _windowed_times(row_values[1], start_ms, stop_ms)
            result.append(tuple(row_values) if isinstance(row, tuple) else row_values)
        else:
            result.append(row)
    return result


def _windowed_soma_spikes(soma_spikes: Any, start_ms: float, stop_ms: float) -> dict[str, Any]:
    if not isinstance(soma_spikes, dict):
        return {}
    payload = dict(soma_spikes)
    payload["labels"] = list(soma_spikes.get("labels") or [])
    payload["spike_times"] = [
        _windowed_times(times, start_ms, stop_ms)
        for times in (soma_spikes.get("spike_times") or [])
    ]
    metadata = dict(soma_spikes.get("metadata") or {})
    metadata["window_start_ms"] = float(start_ms)
    metadata["window_stop_ms"] = float(stop_ms)
    payload["metadata"] = metadata
    return payload


def _windowed_soma_traces(soma_vs: Any, start_ms: float, stop_ms: float) -> list[Any]:
    result = []
    for row in soma_vs or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            label, times, values, *rest = list(row)
            t_window, v_window = _windowed_trace(times, values, start_ms, stop_ms)
            new_row = [label, t_window, v_window, *rest]
            result.append(tuple(new_row) if isinstance(row, tuple) else new_row)
        else:
            result.append(row)
    return result


def window_result_for_condition(
    result: dict[str, Any],
    *,
    start_ms: float,
    stop_ms: float,
    condition: str,
) -> dict[str, Any]:
    """Return a result-like dict restricted to one postprocessed time window."""
    start_ms = max(float(start_ms), 0.0)
    stop_ms = max(float(stop_ms), start_ms)
    duration_ms = max(stop_ms - start_ms, 0.0)
    windowed = dict(result)
    summary = deepcopy(result.get("summary") or {})
    params = deepcopy(summary.get("params") or {})
    params["tstop"] = float(duration_ms)
    params["source_tstop"] = float(_result_tstop_ms(result))
    params["condition_window"] = {
        "condition": str(condition),
        "start_ms": float(start_ms),
        "stop_ms": float(stop_ms),
        "duration_ms": float(duration_ms),
    }
    summary["params"] = params
    windowed["summary"] = summary
    windowed["lfp_t"], windowed["lfp"] = _windowed_trace(
        result.get("lfp_t", []),
        result.get("lfp", []),
        start_ms,
        stop_ms,
    )
    windowed["soma_spikes"] = _windowed_soma_spikes(
        result.get("soma_spikes"),
        start_ms,
        stop_ms,
    )
    windowed["input_times"] = _windowed_event_rows(result.get("input_times"), start_ms, stop_ms)
    windowed["gc_output_events"] = _windowed_event_rows(result.get("gc_output_events"), start_ms, stop_ms)
    if "soma_vs" in result and result.get("soma_vs"):
        windowed["soma_vs"] = _windowed_soma_traces(result.get("soma_vs"), start_ms, stop_ms)
    return windowed


def _switch_condition_windows(
    result: dict[str, Any] | None,
    value: dict[str, Any],
    *,
    default_washout_ms: float,
) -> dict[str, tuple[float, float]]:
    tstop_ms = _result_tstop_ms(result)
    switch_time = value.get("ketamine_switch_time_ms", value.get("ketamine_switch_time"))
    if switch_time in (None, ""):
        summary = (result or {}).get("summary") or {}
        params = summary.get("params") or {}
        switch_payload = params.get("ketamine_switch") or {}
        switch_time = switch_payload.get("time_ms") if isinstance(switch_payload, dict) else None
    switch_time_ms = float(switch_time if switch_time not in (None, "") else tstop_ms * 0.5)
    washout_ms = float(value.get("ketamine_switch_washout_ms", default_washout_ms) or 0.0)
    control_start = min(max(washout_ms, 0.0), max(switch_time_ms, 0.0))
    ketamine_start = min(max(switch_time_ms + max(washout_ms, 0.0), 0.0), max(tstop_ms, 0.0))
    return {
        "control": (control_start, max(switch_time_ms, control_start)),
        "ketamine": (ketamine_start, max(tstop_ms, ketamine_start)),
    }


def score_hfo_batch(
    campaign_dir: str | Path,
    *,
    batch_plan: dict[str, Any],
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    target_hz: float = 195.0,
    target_half_width_hz: float = 35.0,
    switch_washout_ms: float = 500.0,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    items = sweep.get("items", [])
    item_rows = []
    grouped: dict[str, dict[str, Any]] = {}
    candidate_lookup = {
        str(candidate["optimizer_candidate_id"]): candidate
        for candidate in batch_plan["candidates"]
    }

    for item in items:
        value = dict(item.get("value") or {})
        candidate_id = str(value.get("optimizer_candidate_id") or value.get("optimizer_pair_id") or "")
        condition = str(value.get("optimizer_condition") or "")
        result = item.get("result")
        if condition == "switch":
            windows = _switch_condition_windows(
                result,
                value,
                default_washout_ms=switch_washout_ms,
            )
            for split_condition, (window_start, window_stop) in windows.items():
                if result is None or window_stop <= window_start:
                    split_metrics = _empty_condition_metrics()
                else:
                    split_metrics = score_condition_result(
                        window_result_for_condition(
                            result,
                            start_ms=window_start,
                            stop_ms=window_stop,
                            condition=split_condition,
                        ),
                        signal=signal,
                        dt_ms=dt_ms,
                        target_hz=target_hz,
                        target_half_width_hz=target_half_width_hz,
                    )
                item_row = {
                    "batch_name": batch_plan["batch_name"],
                    "candidate_id": candidate_id,
                    "condition": split_condition,
                    "source_condition": condition,
                    "label": item.get("label"),
                    "sweep_dir": str(sweep.get("sweep_dir")),
                    "result_dir": (
                        str(getattr(item.get("run"), "result_dir", ""))
                        if item.get("run") is not None
                        else ""
                    ),
                    "window_start_ms": float(window_start),
                    "window_stop_ms": float(window_stop),
                    "parameters": candidate_lookup.get(candidate_id, {}),
                    **split_metrics,
                }
                item_rows.append(item_row)
                grouped.setdefault(candidate_id, {})[split_condition] = item_row
            continue
        if result is None:
            metrics = _empty_condition_metrics()
        else:
            metrics = score_condition_result(
                result,
                signal=signal,
                dt_ms=dt_ms,
                target_hz=target_hz,
                target_half_width_hz=target_half_width_hz,
            )
        item_row = {
            "batch_name": batch_plan["batch_name"],
            "candidate_id": candidate_id,
            "condition": condition,
            "label": item.get("label"),
            "sweep_dir": str(sweep.get("sweep_dir")),
            "result_dir": str(getattr(item.get("run"), "result_dir", "")) if item.get("run") is not None else "",
            "parameters": candidate_lookup.get(candidate_id, {}),
            **metrics,
        }
        item_rows.append(item_row)
        grouped.setdefault(candidate_id, {})[condition] = item_row

    candidate_rows = []
    for candidate_id, conditions in grouped.items():
        control_row = conditions.get("control")
        ketamine_row = conditions.get("ketamine")
        parameters = candidate_lookup.get(candidate_id, {})
        pair_metrics = score_candidate_pair(
            control_metrics=control_row,
            ketamine_metrics=ketamine_row,
        )
        pair_metrics = _apply_parameter_plausibility_penalty(pair_metrics, parameters)
        candidate_rows.append(
            {
                "batch_name": batch_plan["batch_name"],
                "candidate_id": candidate_id,
                "parameters": parameters,
                "control_metrics": control_row,
                "ketamine_metrics": ketamine_row,
                **pair_metrics,
            }
        )

    _append_jsonl(_archive_path(campaign_dir, kind="item"), item_rows)
    _append_jsonl(_archive_path(campaign_dir, kind="candidate"), candidate_rows)

    scored_payload = {
        "batch_name": batch_plan["batch_name"],
        "strategy": batch_plan["strategy"],
        "stage": batch_plan["stage"],
        "sweep_dir": str(sweep.get("sweep_dir")),
        "candidate_rows": candidate_rows,
        "item_rows": item_rows,
    }
    _write_json(campaign_dir / "batches" / f"{batch_plan['batch_name']}_scored.json", scored_payload)

    state = load_campaign_state(campaign_dir)
    completed = list(state.get("completed_batches", []))
    if batch_plan["batch_name"] not in completed:
        completed.append(batch_plan["batch_name"])
        state["completed_batches"] = completed
        _write_campaign_state(campaign_dir, state)

    return scored_payload


def top_candidate_rows(
    campaign_dir: str | Path,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    archive = load_candidate_archive_rows(campaign_dir)
    archive = [row for row in archive if np.isfinite(float(row.get("pair_score", np.nan)))]
    archive.sort(key=lambda row: float(row["pair_score"]), reverse=True)
    return archive[: int(limit)]


def maybe_dataframe(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return rows
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_CAMPAIGNS_BASE",
    "DEFAULT_SCORE_BANDS",
    "ParameterSpec",
    "build_manual_allocation_remote_config",
    "default_campaign_run_config",
    "default_hfo_search_space",
    "ensure_campaign_dir",
    "infer_remote_template_from_recent_runs",
    "initialize_campaign",
    "load_campaign_state",
    "load_candidate_archive_rows",
    "load_item_archive_rows",
    "load_objective_filter",
    "lfp_source_diagnostic_configs",
    "maybe_dataframe",
    "mean_firing_rates_by_type",
    "paramiko_auth_probe",
    "parameter_plausibility_penalty",
    "propose_elite_batch",
    "propose_lhs_batch",
    "psd_template_curve",
    "rescore_candidate_row",
    "run_hfo_batch",
    "score_candidate_pair",
    "score_condition_result",
    "score_hfo_batch",
    "scaled_psd_template_curve",
    "search_space_rows",
    "sustained_odor_schedule",
    "top_candidate_rows",
    "window_result_for_condition",
    "write_objective_filter",
]
