"""Configuration helpers for slice-building workflows.

These helpers are pure Python so they can be validated without Blender. The
actual builder reads environment variables through this module, and the
``build-slice.py`` entrypoint populates those variables from CLI flags.
"""

from __future__ import annotations

import os
from typing import Any


ENV_PREFIX = "OB_SLICE_"


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(_env_key(name))
    return default if value is None or value == "" else value


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _parse_float(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def _parse_odors(value: str | None) -> list[str] | str:
    if value is None:
        return ["Apple"]
    value = str(value).strip()
    if value.lower() == "all":
        return "all"
    return [token.strip() for token in value.split(",") if token.strip()]


def slice_builder_env_kwargs(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Return ``SliceBuilderBlender`` kwargs parsed from one environment."""
    if environ is None:
        environ = os.environ

    def raw(name: str, default: str | None = None) -> str | None:
        value = environ.get(_env_key(name))
        return default if value is None or value == "" else value

    kwargs: dict[str, Any] = {
        "odors": _parse_odors(raw("ODORS")),
        "slice_object_name": raw("NAME", "DorsalColumnSlice"),
        "slice_output_name": raw("OUTPUT_NAME"),
        "max_mcs": int(raw("MAX_MCS", "10")),
        "max_tcs": _parse_int(raw("MAX_TCS")),
        "max_gcs": _parse_int(raw("MAX_GCS"), 300),
        "max_eplis": _parse_int(raw("MAX_EPLIS"), 0),
        "mc_particles_object_name": raw("MC_PARTICLES", "2 ML Particles"),
        "tc_particles_object_name": raw("TC_PARTICLES", "1 OPL Particles"),
        "gc_particles_object_name": raw("GC_PARTICLES", "4 GRL Particles"),
        "epli_particles_object_name": raw("EPLI_PARTICLES"),
        "glom_particles_object_name": raw("GLOM_PARTICLES", "0 GL Particles"),
        "glom_layer_object_name": raw("GLOM_LAYER", "0 GL"),
        "outer_opl_object_name": raw("OUTER_OPL", "1 OPL-Outer"),
        "inner_opl_object_name": raw("INNER_OPL", "1 OPL-Inner"),
        "enable_epl_interneurons": _parse_bool(raw("ENABLE_EPLI"), False),
        "epl_interneuron_model": raw("EPLI_MODEL"),
        "epl_interneuron_family": raw("EPLI_FAMILY"),
        "epli_depth_min_fraction": _parse_float(raw("EPLI_DEPTH_MIN"), 0.2),
        "epli_depth_max_fraction": _parse_float(raw("EPLI_DEPTH_MAX"), 0.8),
    }
    return kwargs


def slice_builder_env_overrides_from_cli(args: Any) -> dict[str, str]:
    """Convert parsed CLI args to environment-variable overrides."""
    overrides: dict[str, str] = {}

    def set_if(name: str, value: Any, formatter=str) -> None:
        if value is None:
            return
        overrides[_env_key(name)] = formatter(value)

    set_if("NAME", getattr(args, "slice_name", None))
    set_if("OUTPUT_NAME", getattr(args, "slice_output_name", None))
    set_if("ODORS", getattr(args, "odors", None), lambda value: ",".join(value) if isinstance(value, (list, tuple)) else str(value))
    set_if("MAX_MCS", getattr(args, "max_mcs", None))
    set_if("MAX_TCS", getattr(args, "max_tcs", None))
    set_if("MAX_GCS", getattr(args, "max_gcs", None))
    set_if("MAX_EPLIS", getattr(args, "max_eplis", None))
    set_if("MC_PARTICLES", getattr(args, "mc_particles_object_name", None))
    set_if("TC_PARTICLES", getattr(args, "tc_particles_object_name", None))
    set_if("GC_PARTICLES", getattr(args, "gc_particles_object_name", None))
    set_if("EPLI_PARTICLES", getattr(args, "epli_particles_object_name", None))
    set_if("GLOM_PARTICLES", getattr(args, "glom_particles_object_name", None))
    set_if("GLOM_LAYER", getattr(args, "glom_layer_object_name", None))
    set_if("OUTER_OPL", getattr(args, "outer_opl_object_name", None))
    set_if("INNER_OPL", getattr(args, "inner_opl_object_name", None))
    if getattr(args, "enable_epl_interneurons", False):
        overrides[_env_key("ENABLE_EPLI")] = "1"
    set_if("EPLI_MODEL", getattr(args, "epl_interneuron_model", None))
    set_if("EPLI_FAMILY", getattr(args, "epl_interneuron_family", None))
    set_if("EPLI_DEPTH_MIN", getattr(args, "epli_depth_min_fraction", None))
    set_if("EPLI_DEPTH_MAX", getattr(args, "epli_depth_max_fraction", None))
    return overrides


__all__ = [
    "ENV_PREFIX",
    "slice_builder_env_kwargs",
    "slice_builder_env_overrides_from_cli",
]
