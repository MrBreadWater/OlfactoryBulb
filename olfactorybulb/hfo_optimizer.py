"""Batch-oriented HFO regime search helpers for the provisional EPLI model.

The search strategy is intentionally batch-first:

1. Seed with a Latin-hypercube design in transformed parameter space.
2. Evaluate each candidate in paired control / ketamine-block conditions.
3. Score candidates on differential HFO expression around 180 +/- 20 Hz.
4. Refine around elites with a truncated Gaussian proposal plus exploration.

This is a better fit for Phoenix than Nelder-Mead because the objective is
noisy, non-smooth, and expensive, and Phoenix throughput is highest when we
launch many independent runs concurrently inside one long-lived allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import math

import numpy as np
from scipy.stats import qmc

import obgpu_experiment_helpers as hlp


DEFAULT_CAMPAIGNS_BASE = Path("/home/alek/OlfactoryBulb/results/notebook_runs/optimization")
DEFAULT_SCORE_BANDS = {
    "beta": (15.0, 35.0),
    "low_gamma": (35.0, 65.0),
    "high_gamma": (65.0, 100.0),
    "hfo_80_130": (80.0, 130.0),
    "target_hfo": (160.0, 200.0),
    "hfo_200_250": (200.0, 250.0),
}


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    low: float
    high: float
    scale: str = "log"
    dtype: str = "float"
    description: str = ""

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
            high=512.0,
            scale="log",
            description="KAR conductance on M/T cells",
        ),
        ParameterSpec(
            path="kar_gc_gmax",
            low=0.001,
            high=64.0,
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
    ]


def search_space_rows(search_space: Sequence[ParameterSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in search_space]


def infer_remote_template_from_recent_runs(
    *,
    results_base: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return the newest remote run config we can recover from saved run_info."""
    results_base = Path(results_base or "/home/alek/OlfactoryBulb/results/notebook_runs")
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


def default_campaign_run_config(
    remote_config: dict[str, Any],
    *,
    paramset: str = "GammaSignature_EPLI_Provisional_TCOnly",
    nranks: int = 15,
    total_tasks: int = 120,
    tstop_ms: float = 9000.0,
    cell_permute: int = 0,
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
        record_gc_output_events=True,
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
    config["enable_gc_kar"] = True
    config["sweep_parallelism"] = max(int(total_tasks) // max(int(nranks), 1), 1)
    return config


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
    return np.asarray([spec.encode(float(candidate[spec.path])) for spec in search_space], dtype=float)


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
    seed: int | None = None,
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

    # First walk the line between the current best and the strongest near miss.
    for alpha in (0.25, 0.50, 0.75):
        if len(rows) >= n:
            break
        rows.append(np.clip((1.0 - alpha) * top + alpha * second, encoded_lo, encoded_hi))

    priority_paths = [
        "gaba_gmax",
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

    # Then do small one-coordinate probes around the top two points.
    while len(rows) < n:
        center = top if (len(rows) % 2 == 0) else second
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
    path = _archive_path(Path(campaign_dir), kind="candidate")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    if len(valid) >= 192 and len(elite) >= 2 and total_n >= 8:
        explore_n = min(explore_n, max(1, int(round(0.125 * total_n))))
        targeted_n = min(max(2, int(round(0.25 * total_n))), max(total_n - explore_n - 2, 0))
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
        seed=None if seed is None else seed + 2,
    )

    if explore_n > 0:
        explore_rows = np.vstack(
            [_candidate_vector(row, search_space) for row in _decode_unit_samples(_sample_unit_lhs(explore_n, len(search_space), seed=None if seed is None else seed + 1), search_space)]
        )
        all_rows = np.vstack([targeted_rows, local_rows, covariance_rows, explore_rows])
    else:
        all_rows = np.vstack([targeted_rows, local_rows, covariance_rows])

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
            "line_probe_count": int(min(targeted_n, 3)),
            "coordinate_probe_count": int(max(targeted_n - 3, 0)),
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


def run_hfo_batch(
    campaign_dir: str | Path,
    *,
    base_config: dict[str, Any],
    batch_plan: dict[str, Any],
    ketamine_block_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    campaign_dir = Path(campaign_dir)
    sweep_path = _joint_sweep_paths_for_batch(batch_plan, ketamine_block_values=ketamine_block_values)
    config = dict(base_config)
    config["label_prefix"] = f"hfo_opt_{batch_plan['batch_name']}"
    sweep = hlp.run_parameter_sweep(config, sweep_path)
    sweep_dir = Path(sweep["sweep_dir"])
    metadata = {
        "batch_name": batch_plan["batch_name"],
        "strategy": batch_plan["strategy"],
        "stage": batch_plan["stage"],
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


def score_condition_result(
    result: dict[str, Any],
    *,
    signal: str = "lfp",
    dt_ms: float = 0.1,
    target_hz: float = 180.0,
    target_half_width_hz: float = 20.0,
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

    target_lo = float(target_hz) - float(target_half_width_hz)
    target_hi = float(target_hz) + float(target_half_width_hz)
    if len(freqs) == 0:
        return {
            "condition_score": float("-inf"),
            "peak_hz": math.nan,
            "peak_ratio": 0.0,
            "freq_match": 0.0,
            "phase_lock": 0.0,
            "rate_penalty": 0.0,
            "band_power": summary["band_power"],
            "relative_band_power": summary["relative_band_power"],
            "mean_firing_rate_by_type": {},
        }

    target_mask = (freqs >= target_lo) & (freqs <= target_hi)
    broad_mask = (freqs >= 15.0) & (freqs <= 250.0)
    if not np.any(target_mask):
        peak_hz = math.nan
        peak_power = 0.0
    else:
        local_index = int(np.argmax(psd[target_mask]))
        target_freqs = freqs[target_mask]
        target_psd = psd[target_mask]
        peak_hz = float(target_freqs[local_index])
        peak_power = float(target_psd[local_index])

    background_mask = broad_mask & ~target_mask
    shoulder_mask = ((freqs >= 100.0) & (freqs < target_lo)) | ((freqs > target_hi) & (freqs <= 240.0))
    background_floor = float(np.median(psd[background_mask])) if np.any(background_mask) else 0.0
    shoulder_floor = float(np.median(psd[shoulder_mask])) if np.any(shoulder_mask) else background_floor
    denom = max(background_floor, shoulder_floor, 1e-18)
    peak_ratio = peak_power / denom

    freq_match = math.exp(-0.5 * ((peak_hz - target_hz) / max(float(target_half_width_hz), 1e-9)) ** 2) if np.isfinite(peak_hz) else 0.0
    relative_target = float(summary["relative_band_power"].get("target_hfo", 0.0))
    target_power = float(summary["band_power"].get("target_hfo", 0.0))
    side_power = float(summary["band_power"].get("hfo_80_130", 0.0)) + float(summary["band_power"].get("hfo_200_250", 0.0))
    dominance = target_power / max(side_power, 1e-18)
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

    rate_penalty = 0.0
    rate_penalty += max(mean_rates.get("TC", 0.0) - 120.0, 0.0) / 60.0
    rate_penalty += max(mean_rates.get("MC", 0.0) - 80.0, 0.0) / 40.0
    rate_penalty += max(mean_rates.get("EPLI", 0.0) - 250.0, 0.0) / 100.0

    condition_score = (
        2.5 * freq_match
        + 2.0 * math.log10(1.0 + peak_ratio)
        + 3.0 * relative_target
        + 1.5 * math.log10(1.0 + dominance)
        + 0.5 * min(beta_gamma, 0.30)
        + 0.5 * phase_lock
        - rate_penalty
    )
    return {
        "condition_score": float(condition_score),
        "peak_hz": float(peak_hz),
        "peak_ratio": float(peak_ratio),
        "freq_match": float(freq_match),
        "dominance": float(dominance),
        "beta_gamma_support": float(beta_gamma),
        "phase_lock": float(phase_lock),
        "rate_penalty": float(rate_penalty),
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
    control_ratio = float(control_metrics.get("peak_ratio", 0.0))
    ketamine_ratio = float(ketamine_metrics.get("peak_ratio", 0.0))
    control_peak_hz = float(control_metrics.get("peak_hz", math.nan))
    ketamine_peak_hz = float(ketamine_metrics.get("peak_hz", math.nan))

    target_contrast = math.log10((ketamine_target + 1e-12) / (control_target + 1e-12))
    peak_contrast = math.log10((ketamine_ratio + 1e-12) / (control_ratio + 1e-12))
    compound_contrast = math.log10(
        ((ketamine_target * ketamine_ratio) + 1e-12)
        / ((control_target * control_ratio) + 1e-12)
    )
    target_delta = ketamine_target - control_target
    control_leak_penalty = 8.0 * control_target + 1.1 * max(control_score, 0.0)
    same_peak_penalty = 0.0
    if (
        math.isfinite(control_peak_hz)
        and math.isfinite(ketamine_peak_hz)
        and 160.0 <= control_peak_hz <= 200.0
        and abs(control_peak_hz - ketamine_peak_hz) <= 5.0
    ):
        same_peak_penalty = 4.0 + 10.0 * control_target
    negative_delta_penalty = 20.0 * max(-target_delta, 0.0)
    ketamine_freq_match = (
        math.exp(-0.5 * ((ketamine_peak_hz - 180.0) / 18.0) ** 2)
        if math.isfinite(ketamine_peak_hz)
        else 0.0
    )
    pair_score = (
        ketamine_score
        + 3.5 * compound_contrast
        + 12.0 * target_delta
        + 1.5 * ketamine_freq_match
        - control_leak_penalty
        - same_peak_penalty
        - negative_delta_penalty
    )
    return {
        "pair_score": float(pair_score),
        "target_contrast_log10": float(target_contrast),
        "peak_contrast_log10": float(peak_contrast),
        "compound_contrast_log10": float(compound_contrast),
        "target_delta": float(target_delta),
        "control_leak_penalty": float(control_leak_penalty),
        "same_peak_penalty": float(same_peak_penalty),
        "negative_delta_penalty": float(negative_delta_penalty),
        "ketamine_freq_match": float(ketamine_freq_match),
        "control_score": float(control_score),
        "ketamine_score": float(ketamine_score),
    }


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(hlp._json_ready(row), sort_keys=True) + "\n")


def score_hfo_batch(
    campaign_dir: str | Path,
    *,
    batch_plan: dict[str, Any],
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    target_hz: float = 180.0,
    target_half_width_hz: float = 20.0,
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
        if result is None:
            metrics = {
                "condition_score": float("-inf"),
                "peak_hz": math.nan,
                "peak_ratio": 0.0,
                "freq_match": 0.0,
                "dominance": 0.0,
                "beta_gamma_support": 0.0,
                "phase_lock": 0.0,
                "rate_penalty": 0.0,
                "band_power": {},
                "relative_band_power": {},
                "mean_firing_rate_by_type": {},
            }
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
        pair_metrics = score_candidate_pair(
            control_metrics=control_row,
            ketamine_metrics=ketamine_row,
        )
        candidate_rows.append(
            {
                "batch_name": batch_plan["batch_name"],
                "candidate_id": candidate_id,
                "parameters": candidate_lookup.get(candidate_id, {}),
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
    "maybe_dataframe",
    "mean_firing_rates_by_type",
    "paramiko_auth_probe",
    "propose_elite_batch",
    "propose_lhs_batch",
    "run_hfo_batch",
    "score_candidate_pair",
    "score_condition_result",
    "score_hfo_batch",
    "search_space_rows",
    "top_candidate_rows",
]
