"""Central HFO parameter registry and runtime wiring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Sequence


PARAMETER_CONTRACT_VERSION = 1

DEFAULT_TIME_CONSTANTS_MS = {
    "input_syn_tau1_ms": 6.0,
    "input_syn_tau2_ms": 12.0,
    "gaba_tau2_ms": 100.0,
    "kar_tau1_ms": 6.728726245,
    "kar_tau2_ms": 81.75126152,
    "kar_tau3_ms": 468.7337682,
}
TIME_CONSTANT_FRACTIONAL_VARIATION = 0.20


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


def _time_range(path: str) -> tuple[float, float]:
    baseline = float(DEFAULT_TIME_CONSTANTS_MS[path])
    frac = float(TIME_CONSTANT_FRACTIONAL_VARIATION)
    return baseline * (1.0 - frac), baseline * (1.0 + frac)


def default_hfo_search_space() -> list[ParameterSpec]:
    """Return the maintained HFO optimizer search space."""
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
        ParameterSpec(
            path="input_syn_tau1_ms",
            low=_time_range("input_syn_tau1_ms")[0],
            high=_time_range("input_syn_tau1_ms")[1],
            scale="linear",
            default=DEFAULT_TIME_CONSTANTS_MS["input_syn_tau1_ms"],
            description="OSN input Exp2Syn rise time",
        ),
        ParameterSpec(
            path="input_syn_tau2_ms",
            low=_time_range("input_syn_tau2_ms")[0],
            high=_time_range("input_syn_tau2_ms")[1],
            scale="linear",
            default=DEFAULT_TIME_CONSTANTS_MS["input_syn_tau2_ms"],
            description="OSN input Exp2Syn decay time",
        ),
        ParameterSpec(
            path="gaba_tau2_ms",
            low=_time_range("gaba_tau2_ms")[0],
            high=_time_range("gaba_tau2_ms")[1],
            scale="linear",
            default=DEFAULT_TIME_CONSTANTS_MS["gaba_tau2_ms"],
            description="Global GabaSyn decay time",
        ),
        ParameterSpec(
            path="kar_tau1_ms",
            low=_time_range("kar_tau1_ms")[0],
            high=_time_range("kar_tau1_ms")[1],
            scale="linear",
            default=DEFAULT_TIME_CONSTANTS_MS["kar_tau1_ms"],
            description="KAR kernel fast rise time",
        ),
        ParameterSpec(
            path="kar_tau2_ms",
            low=_time_range("kar_tau2_ms")[0],
            high=_time_range("kar_tau2_ms")[1],
            scale="linear",
            default=DEFAULT_TIME_CONSTANTS_MS["kar_tau2_ms"],
            description="KAR kernel medium decay time",
        ),
        ParameterSpec(
            path="kar_tau3_ms",
            low=_time_range("kar_tau3_ms")[0],
            high=_time_range("kar_tau3_ms")[1],
            scale="linear",
            default=DEFAULT_TIME_CONSTANTS_MS["kar_tau3_ms"],
            description="KAR kernel slow tail time",
        ),
    ]


def search_space_rows(search_space: Sequence[ParameterSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in search_space]


HFO_RUN_CONFIG_DEFAULTS: dict[str, Any] = {
    "input_syn_tau1_ms": None,
    "input_syn_tau2_ms": None,
    "mc_input_weight": None,
    "tc_input_weight": None,
    "mc_input_delay_ms": None,
    "tc_input_delay_ms": None,
    "gap_mc": None,
    "gap_tc": None,
    "ampa_nmda_gmax": None,
    "ampa_nmda_nmdafactor": None,
    "ketamine_block": None,
    "ketamine_switch_time_ms": None,
    "ketamine_block_after_switch": None,
    "ampa_block": None,
    "gaba_gmax": None,
    "gaba_tau2_ms": None,
    "gc_gaba_weight_scale": None,
    "gc_ampa_weight_scale": None,
    "epli_gaba_weight_scale": None,
    "epli_ampa_weight_scale": None,
    "kar_mt_gmax": None,
    "enable_gc_kar": None,
    "kar_gc_gmax": None,
    "kar_tau1_ms": None,
    "kar_tau2_ms": None,
    "kar_tau3_ms": None,
    "kar_amp1": None,
    "kar_amp2": None,
    "kar_amp3": None,
    "kar_kd": None,
    "kar_block": None,
    "kar_osn_weight_scale": None,
    "kar_gc_weight_scale": None,
    "gc_ka_gbar_scale": None,
}

HFO_CONTROL_HELP: dict[str, str] = {
    "input_syn_tau1_ms": "Input Exp2Syn tau1.",
    "input_syn_tau2_ms": "Input Exp2Syn tau2.",
    "mc_input_weight": "MC odor input synaptic weight.",
    "tc_input_weight": "TC odor input synaptic weight.",
    "mc_input_delay_ms": "MC odor input delay in ms.",
    "tc_input_delay_ms": "TC odor input delay in ms.",
    "gap_mc": "MC gap-junction conductance.",
    "gap_tc": "TC gap-junction conductance.",
    "ampa_nmda_gmax": "Global AmpaNmdaSyn gmax.",
    "ampa_nmda_nmdafactor": "Global AmpaNmdaSyn NMDA factor.",
    "ketamine_block": "Semantic NMDA block multiplier on AmpaNmdaSyn NMDA current.",
    "ketamine_switch_time_ms": "Optional simulation time when AmpaNmdaSyn switches to ketamine_block_after_switch.",
    "ketamine_block_after_switch": "NMDA block multiplier used after ketamine_switch_time_ms.",
    "ampa_block": "AMPA current multiplier on AmpaNmdaSyn AMPA current.",
    "gaba_gmax": "Global GabaSyn gmax.",
    "gaba_tau2_ms": "Global GabaSyn tau2.",
    "gc_gaba_weight_scale": "Multiplier applied to GC->MC/TC reciprocal GABA NetCon weights.",
    "gc_ampa_weight_scale": "Multiplier applied to MC/TC->GC reciprocal AMPA/NMDA NetCon weights.",
    "epli_gaba_weight_scale": "Multiplier applied to EPLI->MC/TC reciprocal GABA NetCon weights.",
    "epli_ampa_weight_scale": "Multiplier applied to MC/TC->EPLI reciprocal AMPA/NMDA NetCon weights.",
    "kar_mt_gmax": "Slow OSN-glutamate KAR conductance on MC/TC tuft inputs.",
    "enable_gc_kar": "Enable optional MC/TC->GC KAR conductance at reciprocal excitation sites.",
    "kar_gc_gmax": "Optional slow MC/TC-glutamate KAR conductance on GCs.",
    "kar_tau1_ms": "KAR activation rise time.",
    "kar_tau2_ms": "KAR activation decay time.",
    "kar_tau3_ms": "Slow KAR tail time constant for the fitted conductance kernel.",
    "kar_amp1": "First fitted KAR conductance-kernel amplitude.",
    "kar_amp2": "Second fitted KAR conductance-kernel amplitude.",
    "kar_amp3": "Third fitted KAR conductance-kernel amplitude.",
    "kar_kd": "KAR activation half-saturation for event-driven glutamate proxy.",
    "kar_block": "KAR current multiplier for sensitivity/blockade tests.",
    "kar_osn_weight_scale": "Multiplier applied to OSN event weights delivered to KAR synapses.",
    "kar_gc_weight_scale": "Multiplier applied to reciprocal MC/TC event weights delivered to GC KAR synapses.",
    "gc_ka_gbar_scale": "Scale GC KA/I_A conductance; 0 removes GC I_A.",
}

ROOT_FLOAT_OVERRIDE_MAP = {
    "input_syn_tau1_ms": "input_syn_tau1",
    "input_syn_tau2_ms": "input_syn_tau2",
    "mc_input_weight": "mc_input_weight",
    "tc_input_weight": "tc_input_weight",
    "mc_input_delay_ms": "mc_input_delay",
    "tc_input_delay_ms": "tc_input_delay",
}

SCALAR_FLOAT_OVERRIDE_MAP = {
    "gc_gaba_weight_scale": "gc_gaba_weight_scale",
    "gc_ampa_weight_scale": "gc_ampa_weight_scale",
    "epli_gaba_weight_scale": "epli_gaba_weight_scale",
    "epli_ampa_weight_scale": "epli_ampa_weight_scale",
    "kar_mt_gmax": "kar_mt_gmax",
    "kar_gc_gmax": "kar_gc_gmax",
    "kar_tau1_ms": "kar_tau1",
    "kar_tau2_ms": "kar_tau2",
    "kar_tau3_ms": "kar_tau3",
    "kar_amp1": "kar_amp1",
    "kar_amp2": "kar_amp2",
    "kar_amp3": "kar_amp3",
    "kar_kd": "kar_kd",
    "kar_block": "kar_block",
    "kar_osn_weight_scale": "kar_osn_weight_scale",
    "kar_gc_weight_scale": "kar_gc_weight_scale",
    "gc_ka_gbar_scale": "gc_ka_gbar_scale",
}


def hfo_run_config_defaults() -> dict[str, Any]:
    return dict(HFO_RUN_CONFIG_DEFAULTS)


def hfo_control_help() -> dict[str, str]:
    return dict(HFO_CONTROL_HELP)


def hfo_runtime_parameter_keys() -> tuple[str, ...]:
    return tuple(HFO_RUN_CONFIG_DEFAULTS)


def default_search_space_paths() -> list[str]:
    return [spec.path for spec in default_hfo_search_space()]


def parameter_display_order(
    parameters: dict[str, Any] | None = None,
    *,
    search_space_paths: Sequence[str] | None = None,
    campaign_dir: str | Path | None = None,
) -> list[str]:
    params = parameters if isinstance(parameters, dict) else {}
    preferred = list(
        search_space_paths
        or campaign_search_space_paths(campaign_dir, fallback=default_search_space_paths())
    )
    known_tail = [key for key in hfo_runtime_parameter_keys() if key not in preferred]
    extras = sorted(
        key
        for key in params
        if key not in preferred and key not in known_tail and not str(key).startswith("optimizer_")
    )
    return [*preferred, *known_tail, *extras]


def parameter_contract_snapshot(
    *,
    search_space: Sequence[ParameterSpec] | None = None,
    campaign_dir: str | Path | None = None,
) -> dict[str, Any]:
    if campaign_dir is not None:
        search_space_paths = campaign_search_space_paths(campaign_dir, fallback=default_search_space_paths())
    elif search_space is not None:
        search_space_paths = [spec.path for spec in search_space]
    else:
        search_space_paths = default_search_space_paths()
    return {
        "version": PARAMETER_CONTRACT_VERSION,
        "search_space_paths": list(search_space_paths),
        "runtime_parameter_keys": list(hfo_runtime_parameter_keys()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=128)
def _cached_campaign_search_space_paths(campaign_config_path: str) -> tuple[str, ...]:
    payload = _read_json(Path(campaign_config_path))
    rows = payload.get("search_space") or []
    paths = [
        str(row.get("path"))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("path")
    ]
    return tuple(paths)


def campaign_search_space_paths(
    campaign_dir: str | Path | None,
    *,
    fallback: Sequence[str] | None = None,
) -> list[str]:
    fallback_paths = list(fallback or default_search_space_paths())
    if campaign_dir is None:
        return fallback_paths
    campaign_config_path = Path(campaign_dir) / "campaign_config.json"
    if not campaign_config_path.exists():
        return fallback_paths
    paths = list(_cached_campaign_search_space_paths(str(campaign_config_path.resolve())))
    return paths or fallback_paths


def apply_hfo_runtime_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Mutate a benchmark override payload with HFO-related runtime settings."""
    for config_key, override_key in ROOT_FLOAT_OVERRIDE_MAP.items():
        if config.get(config_key) is not None:
            overrides[override_key] = float(config[config_key])

    if config.get("gap_mc") is not None or config.get("gap_tc") is not None:
        overrides.setdefault("gap_juction_gmax", {})
        if config.get("gap_mc") is not None:
            overrides["gap_juction_gmax"]["MC"] = float(config["gap_mc"])
        if config.get("gap_tc") is not None:
            overrides["gap_juction_gmax"]["TC"] = float(config["gap_tc"])

    if any(
        config.get(key) is not None
        for key in (
            "ampa_nmda_gmax",
            "ampa_nmda_nmdafactor",
            "ketamine_block",
            "ketamine_switch_time_ms",
            "ketamine_block_after_switch",
            "ampa_block",
            "gaba_gmax",
            "gaba_tau2_ms",
        )
    ):
        overrides.setdefault("synapse_properties", {})

    if any(
        config.get(key) is not None
        for key in (
            "ampa_nmda_gmax",
            "ampa_nmda_nmdafactor",
            "ketamine_block",
            "ketamine_switch_time_ms",
            "ketamine_block_after_switch",
            "ampa_block",
        )
    ):
        overrides["synapse_properties"].setdefault("AmpaNmdaSyn", {})
        syn = overrides["synapse_properties"]["AmpaNmdaSyn"]
        if config.get("ampa_nmda_gmax") is not None:
            syn["gmax"] = float(config["ampa_nmda_gmax"])
        if config.get("ampa_nmda_nmdafactor") is not None:
            syn["nmdafactor"] = float(config["ampa_nmda_nmdafactor"])
        if config.get("ketamine_block") is not None:
            syn["ketamine_block"] = float(config["ketamine_block"])
        if config.get("ketamine_switch_time_ms") is not None:
            syn["ketamine_switch_time"] = float(config["ketamine_switch_time_ms"])
        if config.get("ketamine_switch_time_ms") is not None and config.get("ketamine_block_after_switch") is None:
            syn["ketamine_block_after"] = 0.0
        if config.get("ketamine_block_after_switch") is not None:
            syn["ketamine_block_after"] = float(config["ketamine_block_after_switch"])
        if config.get("ampa_block") is not None:
            syn["ampa_block"] = float(config["ampa_block"])

    if config.get("gaba_gmax") is not None or config.get("gaba_tau2_ms") is not None:
        overrides["synapse_properties"].setdefault("GabaSyn", {})
        syn = overrides["synapse_properties"]["GabaSyn"]
        if config.get("gaba_gmax") is not None:
            syn["gmax"] = float(config["gaba_gmax"])
        if config.get("gaba_tau2_ms") is not None:
            syn["tau2"] = float(config["gaba_tau2_ms"])

    for config_key, override_key in SCALAR_FLOAT_OVERRIDE_MAP.items():
        if config.get(config_key) is not None:
            overrides[override_key] = float(config[config_key])

    if config.get("enable_gc_kar") is not None:
        overrides["enable_gc_kar"] = bool(config["enable_gc_kar"])


__all__ = [
    "DEFAULT_TIME_CONSTANTS_MS",
    "HFO_CONTROL_HELP",
    "HFO_RUN_CONFIG_DEFAULTS",
    "PARAMETER_CONTRACT_VERSION",
    "ParameterSpec",
    "SCALAR_FLOAT_OVERRIDE_MAP",
    "TIME_CONSTANT_FRACTIONAL_VARIATION",
    "ROOT_FLOAT_OVERRIDE_MAP",
    "apply_hfo_runtime_overrides",
    "campaign_search_space_paths",
    "default_hfo_search_space",
    "default_search_space_paths",
    "hfo_control_help",
    "hfo_run_config_defaults",
    "hfo_runtime_parameter_keys",
    "parameter_contract_snapshot",
    "parameter_display_order",
    "search_space_rows",
]
