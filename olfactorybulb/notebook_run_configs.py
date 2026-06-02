"""Concrete olfactory-bulb notebook run-config helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

from neuroinfra.artifacts.output_paths import make_timestamp as _neuroinfra_make_timestamp
from neuroinfra.notebooks.config_store import json_ready as _json_ready
from neuroinfra.remote.config import build_remote_slurm_config as _neuroinfra_build_remote_slurm_config
from olfactorybulb.hfo_features import (
    apply_hfo_runtime_overrides,
    hfo_run_config_defaults,
)
from olfactorybulb.result_artifacts import (
    DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
    DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
    DEFAULT_SOMA_SPIKE_THRESHOLD_MV,
    DEFAULT_SOMA_TRACE_DTYPE,
    DEFAULT_SOMA_TRACE_FORMAT,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "notebook_runs"


def default_local_mpi_exec() -> str:
    """Return the preferred local MPI launcher for the current shell."""
    configured = os.environ.get("OB_MPIEXEC")
    if configured:
        return configured

    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        slurm_mpi_type = os.environ.get("OB_SLURM_MPI_TYPE", "pmix").strip()
        if slurm_mpi_type:
            return f"srun --mpi={slurm_mpi_type}"
        return "srun"

    return "mpiexec"


def default_remote_mpi_exec() -> str:
    """Return the preferred MPI launcher for the Sol Slurm backend."""
    return "srun --mpi=pmix_v4 --cpu-bind=none"


def make_timestamp() -> str:
    """Return a timestamp string using the standard notebook-run convention."""
    return _neuroinfra_make_timestamp()


def build_run_config(**overrides: Any) -> dict[str, Any]:
    """Build the concrete normalized notebook control dictionary."""
    mode = overrides.pop("mode", "fast")
    base = {
        "mode": mode,
        "paramset": "GammaSignature",
        "label_prefix": "obgpu_experiment",
        "results_base": str(DEFAULT_RESULTS_BASE),
        "nranks": 1 if mode == "fast" else 2,
        "use_corenrn": None,
        "use_gpu": None,
        "cell_permute": 2,
        "tstop_ms": None,
        "sim_dt_ms": 0.1,
        "recording_period_ms": 0.1,
        "soma_trace_format": DEFAULT_SOMA_TRACE_FORMAT,
        "soma_trace_dtype": DEFAULT_SOMA_TRACE_DTYPE,
        "soma_spike_threshold_mv": DEFAULT_SOMA_SPIKE_THRESHOLD_MV,
        "soma_spike_min_prominence_mv": DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
        "soma_spike_refractory_ms": DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
        "save_soma_traces": True,
        "save_voltage_summary": True,
        "enable_lfp": True,
        "disable_status_report": True,
        "parallel_timeout": None,
        "rnd_seed": None,
        "record_from_somas": ["MC", "TC", "GC"],
        "record_gc_output_events": True,
        "keep_native_lfp_debug_files": False,
        "enable_reciprocal_synapses": True,
        "enable_epl_interneurons": None,
        "max_epl_interneurons": None,
        "epl_interneuron_cell_type": None,
        "gc_output_bin_ms": 5.0,
        "gc_output_smooth_sigma_ms": 10.0,
        "gc_output_max_connections": 120,
        "gc_output_rate_normalization": "per_target_cell",
        "input_bin_ms": 5.0,
        "input_smooth_sigma_ms": 10.0,
        "input_max_segments": 120,
        "input_rate_normalization": "per_target_cell",
        "lfp_electrode_location": [116, 1078, -61],
        "lfp_include_cell_types": None,
        "lfp_exclude_cell_types": None,
        "input_odors": None,
        "input_stimuli": None,
        "max_firing_rate_hz": None,
        "inhale_duration_ms": None,
        **hfo_run_config_defaults(),
        "analysis_dt_ms": 0.1,
        "spectrogram_signal": "lfp",
        "spectrogram_max_freq_hz": 250.0,
        "spectrogram_nperseg": 256,
        "spectrogram_noverlap": 192,
        "wavelet_signal": "lfp",
        "max_voltage_traces_per_type": 4,
        "max_spike_raster_cells_per_type": 24,
        "extra_overrides": {},
        "runner_backend": "local",
        "mpi_exec": default_local_mpi_exec(),
        "remote_mpi_exec": default_remote_mpi_exec(),
        "remote_host": None,
        "remote_repo_root": None,
        "remote_results_root": None,
        "remote_conda_activate_cmd": "source tools/setup/activate_obgpu.sh",
        "remote_runtime_profiles": [],
        "remote_fallback_conda_activate_cmd": None,
        "remote_fast_node_feature": None,
        "remote_mechanism_profile": "default",
        "remote_fallback_mechanism_profile": "portable",
        "remote_repo_mode": "shared",
        "remote_git_ref": None,
        "remote_git_fetch": False,
        "remote_git_remote": "origin",
        "slurm_allocation_job_id": None,
        "slurm_reuse_allocation": False,
        "slurm_allocation_time": None,
        "slurm_allocation_name": None,
        "sweep_engine": "auto",
        "sweep_parallelism": None,
        "sweep_sync_live": True,
        "sweep_live_sync_max_items_per_poll": 8,
        "sweep_sync_soma_vs": False,
        "sweep_sync_voltage_summary": False,
        "remote_poll_interval_s": 1.0,
        "remote_log_poll_interval_s": 5.0,
        "remote_live_status": True,
        "remote_live_logs": True,
        "remote_heartbeat_timeout_s": 120,
        "remote_ssh_command_timeout_s": 300,
        "remote_ssh_exec_timeout_s": 30,
        "remote_ssh_upload_timeout_s": 120,
        "remote_poll_command_timeout_s": 60,
        "remote_cleanup_stale_allocations": True,
        "remote_defer_soma_vs_sync": False,
        "remote_preserve_paramiko_session": True,
        "slurm_partition": None,
        "slurm_account": None,
        "slurm_time": None,
        "slurm_gpus": None,
        "slurm_cpus_per_task": None,
        "slurm_step_ntasks": None,
        "slurm_mem": None,
        "slurm_extra_args": [],
        "ssh_options": [],
        "ssh_transport": "paramiko",
        "ssh_keepalive_s": 30,
        "ssh_connect_retries": 4,
        "ssh_connect_retry_backoff_s": 1.0,
        "add_connections": [],
        "modify_connections": [],
        "swap_cell_types": [],
    }
    base.update(overrides)
    return base


def warn_remote_execution_mode_reset() -> None:
    """Warn that remote configs clear local acceleration toggles and infer mode from Slurm."""
    warnings.warn(
        "Remote Slurm configs reset use_corenrn/use_gpu to auto. "
        "If you apply them via RUN_CONFIG.update(...), any previous local values for those keys "
        "will be cleared. Remote execution mode will then be inferred from slurm_gpus unless you "
        "explicitly set use_corenrn/use_gpu again after applying the remote config.",
        stacklevel=2,
    )


def build_slurm_remote_config(
    *,
    remote_host: str,
    remote_repo_root: str | Path,
    remote_results_root: str | Path | None = None,
    remote_conda_activate_cmd: str = "source tools/setup/activate_obgpu.sh",
    remote_runtime_profiles: list[dict[str, Any]] | None = None,
    remote_fallback_conda_activate_cmd: str | None = None,
    remote_fast_node_feature: str | None = None,
    remote_mechanism_profile: str = "default",
    remote_fallback_mechanism_profile: str = "portable",
    remote_mpi_exec: str | None = None,
    slurm_partition: str | None = None,
    slurm_account: str | None = None,
    slurm_time: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_step_ntasks: int | None = None,
    slurm_mem: str | None = None,
    sweep_sync_live: bool = False,
    remote_poll_interval_s: float = 1.0,
    remote_log_poll_interval_s: float = 5.0,
    remote_live_status: bool = True,
    remote_live_logs: bool = True,
    remote_heartbeat_timeout_s: int = 120,
    remote_ssh_command_timeout_s: float | None = 300,
    remote_ssh_exec_timeout_s: float | None = 30,
    remote_ssh_upload_timeout_s: float | None = 120,
    remote_poll_command_timeout_s: float | None = 60,
    remote_cleanup_stale_allocations: bool = True,
    remote_defer_soma_vs_sync: bool = False,
    sweep_live_sync_max_items_per_poll: int = 8,
    sweep_sync_soma_vs: bool = False,
    sweep_sync_voltage_summary: bool = False,
    remote_preserve_paramiko_session: bool = True,
    remote_allow_paramiko_reauth: bool = False,
    remote_repo_mode: str = "shared",
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = False,
    remote_git_remote: str = "origin",
    slurm_allocation_job_id: str | None = None,
    slurm_reuse_allocation: bool = False,
    slurm_allocation_time: str | None = None,
    slurm_allocation_name: str | None = None,
    ssh_options: list[str] | None = None,
    slurm_extra_args: list[str] | None = None,
    ssh_connect_retries: int = 4,
    ssh_connect_retry_backoff_s: float = 1.0,
) -> dict[str, Any]:
    """Return a generic remote Slurm config with notebook-facing defaults."""
    warn_remote_execution_mode_reset()
    return _neuroinfra_build_remote_slurm_config(
        remote_host=remote_host,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        remote_conda_activate_cmd=remote_conda_activate_cmd,
        remote_runtime_profiles=remote_runtime_profiles,
        remote_fallback_conda_activate_cmd=remote_fallback_conda_activate_cmd,
        remote_fast_node_feature=remote_fast_node_feature,
        remote_mechanism_profile=remote_mechanism_profile,
        remote_fallback_mechanism_profile=remote_fallback_mechanism_profile,
        remote_mpi_exec=remote_mpi_exec,
        default_remote_mpi_exec=default_remote_mpi_exec(),
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        slurm_time=slurm_time,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_step_ntasks=slurm_step_ntasks,
        slurm_mem=slurm_mem,
        sweep_sync_live=sweep_sync_live,
        remote_poll_interval_s=remote_poll_interval_s,
        remote_log_poll_interval_s=remote_log_poll_interval_s,
        remote_live_status=remote_live_status,
        remote_live_logs=remote_live_logs,
        remote_heartbeat_timeout_s=remote_heartbeat_timeout_s,
        remote_ssh_command_timeout_s=remote_ssh_command_timeout_s,
        remote_ssh_exec_timeout_s=remote_ssh_exec_timeout_s,
        remote_ssh_upload_timeout_s=remote_ssh_upload_timeout_s,
        remote_poll_command_timeout_s=remote_poll_command_timeout_s,
        remote_cleanup_stale_allocations=remote_cleanup_stale_allocations,
        remote_defer_soma_vs_sync=remote_defer_soma_vs_sync,
        sweep_live_sync_max_items_per_poll=sweep_live_sync_max_items_per_poll,
        sweep_sync_soma_vs=sweep_sync_soma_vs,
        sweep_sync_voltage_summary=sweep_sync_voltage_summary,
        remote_preserve_paramiko_session=remote_preserve_paramiko_session,
        remote_allow_paramiko_reauth=remote_allow_paramiko_reauth,
        remote_repo_mode=remote_repo_mode,
        remote_git_ref=remote_git_ref,
        remote_git_fetch=remote_git_fetch,
        remote_git_remote=remote_git_remote,
        slurm_allocation_job_id=slurm_allocation_job_id,
        slurm_reuse_allocation=slurm_reuse_allocation,
        slurm_allocation_time=slurm_allocation_time,
        slurm_allocation_name=slurm_allocation_name,
        ssh_options=ssh_options,
        slurm_extra_args=slurm_extra_args,
        ssh_connect_retries=ssh_connect_retries,
        ssh_connect_retry_backoff_s=ssh_connect_retry_backoff_s,
        runner_backend="slurm_remote",
    )


def build_sol_remote_config(
    *,
    remote_host: str,
    remote_repo_root: str | Path,
    remote_results_root: str | Path | None = None,
    remote_conda_activate_cmd: str = "source tools/setup/activate_sol_obgpu.sh",
    remote_runtime_profiles: list[dict[str, Any]] | None = None,
    remote_fallback_conda_activate_cmd: str | None = None,
    remote_fast_node_feature: str | None = None,
    remote_mechanism_profile: str = "default",
    remote_fallback_mechanism_profile: str = "portable",
    remote_mpi_exec: str | None = None,
    slurm_partition: str | None = None,
    slurm_account: str | None = None,
    slurm_time: str | None = None,
    slurm_gpus: int | None = None,
    slurm_cpus_per_task: int | None = None,
    slurm_step_ntasks: int | None = None,
    slurm_mem: str | None = None,
    sweep_sync_live: bool = False,
    remote_poll_interval_s: float = 1.0,
    remote_log_poll_interval_s: float = 5.0,
    remote_live_status: bool = True,
    remote_live_logs: bool = True,
    remote_heartbeat_timeout_s: int = 120,
    remote_ssh_command_timeout_s: float | None = 300,
    remote_ssh_exec_timeout_s: float | None = 30,
    remote_ssh_upload_timeout_s: float | None = 120,
    remote_poll_command_timeout_s: float | None = 60,
    remote_cleanup_stale_allocations: bool = True,
    remote_defer_soma_vs_sync: bool = False,
    sweep_live_sync_max_items_per_poll: int = 8,
    sweep_sync_soma_vs: bool = False,
    sweep_sync_voltage_summary: bool = False,
    remote_preserve_paramiko_session: bool = True,
    remote_allow_paramiko_reauth: bool = False,
    remote_repo_mode: str = "shared",
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = False,
    remote_git_remote: str = "origin",
    slurm_allocation_job_id: str | None = None,
    slurm_reuse_allocation: bool = False,
    slurm_allocation_time: str | None = None,
    slurm_allocation_name: str | None = None,
    ssh_options: list[str] | None = None,
    slurm_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Return a Sol-specific remote runner config with Sol activation defaults."""
    config = build_slurm_remote_config(
        remote_host=remote_host,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        remote_conda_activate_cmd=remote_conda_activate_cmd,
        remote_runtime_profiles=remote_runtime_profiles,
        remote_fallback_conda_activate_cmd=remote_fallback_conda_activate_cmd,
        remote_fast_node_feature=remote_fast_node_feature,
        remote_mechanism_profile=remote_mechanism_profile,
        remote_fallback_mechanism_profile=remote_fallback_mechanism_profile,
        remote_mpi_exec=remote_mpi_exec,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        slurm_time=slurm_time,
        slurm_gpus=slurm_gpus,
        slurm_cpus_per_task=slurm_cpus_per_task,
        slurm_step_ntasks=slurm_step_ntasks,
        slurm_mem=slurm_mem,
        sweep_sync_live=sweep_sync_live,
        remote_poll_interval_s=remote_poll_interval_s,
        remote_log_poll_interval_s=remote_log_poll_interval_s,
        remote_live_status=remote_live_status,
        remote_live_logs=remote_live_logs,
        remote_heartbeat_timeout_s=remote_heartbeat_timeout_s,
        remote_ssh_command_timeout_s=remote_ssh_command_timeout_s,
        remote_ssh_exec_timeout_s=remote_ssh_exec_timeout_s,
        remote_ssh_upload_timeout_s=remote_ssh_upload_timeout_s,
        remote_poll_command_timeout_s=remote_poll_command_timeout_s,
        remote_cleanup_stale_allocations=remote_cleanup_stale_allocations,
        remote_defer_soma_vs_sync=remote_defer_soma_vs_sync,
        sweep_live_sync_max_items_per_poll=sweep_live_sync_max_items_per_poll,
        sweep_sync_soma_vs=sweep_sync_soma_vs,
        sweep_sync_voltage_summary=sweep_sync_voltage_summary,
        remote_preserve_paramiko_session=remote_preserve_paramiko_session,
        remote_allow_paramiko_reauth=remote_allow_paramiko_reauth,
        remote_repo_mode=remote_repo_mode,
        remote_git_ref=remote_git_ref,
        remote_git_fetch=remote_git_fetch,
        remote_git_remote=remote_git_remote,
        slurm_allocation_job_id=slurm_allocation_job_id,
        slurm_reuse_allocation=slurm_reuse_allocation,
        slurm_allocation_time=slurm_allocation_time,
        slurm_allocation_name=slurm_allocation_name,
        ssh_options=ssh_options,
        slurm_extra_args=slurm_extra_args,
    )
    config["runner_backend"] = "sol_slurm"
    return config


def default_sol_runtime_profiles(
    *,
    grace_hopper_env: str = "OBGPU",
    arm_env: str = "OBGPU",
    x86_env: str = "OBGPU",
    grace_hopper_mechanism_profile: str = "sol-gh",
    arm_mechanism_profile: str = "sol-arm",
    x86_mechanism_profile: str = "sol-x86_64",
) -> list[dict[str, Any]]:
    """Return ordered runtime profiles for Sol's Grace Hopper, ARM, and x86 nodes."""
    return [
        {
            "name": "sol-grace-hopper",
            "conda_activate_cmd": f"source tools/setup/activate_sol_obgpu.sh {grace_hopper_env}",
            "mechanism_profile": grace_hopper_mechanism_profile,
            "match_arch": ["aarch64", "arm64"],
            "match_any": ["grace", "hopper", "gh200"],
        },
        {
            "name": "sol-arm",
            "conda_activate_cmd": f"source tools/setup/activate_sol_obgpu.sh {arm_env}",
            "mechanism_profile": arm_mechanism_profile,
            "match_arch": ["aarch64", "arm64"],
            "reject_any": ["grace", "hopper", "gh200"],
        },
        {
            "name": "sol-x86_64",
            "conda_activate_cmd": f"source tools/setup/activate_sol_obgpu.sh {x86_env}",
            "mechanism_profile": x86_mechanism_profile,
            "match_arch": ["x86_64", "amd64"],
        },
    ]


def make_label(config: dict[str, Any], timestamp: str | None = None) -> str:
    """Build the timestamped notebook label for a run configuration."""
    timestamp = timestamp or make_timestamp()
    mode = str(config.get("mode", "run"))
    paramset = str(config.get("paramset", "Paramset"))
    prefix = str(config.get("label_prefix", "obgpu_experiment"))
    return f"{prefix}_{paramset}_{mode}_{timestamp}"


def resolve_execution_mode(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the effective CoreNEURON/GPU execution mode for one run config."""
    runner_backend = str(config.get("runner_backend", "local"))
    explicit_corenrn = config.get("use_corenrn")
    explicit_gpu = config.get("use_gpu")

    if explicit_corenrn is not None or explicit_gpu is not None:
        resolved_corenrn = bool(explicit_corenrn)
        resolved_gpu = bool(explicit_gpu)
        if resolved_gpu and not resolved_corenrn:
            resolved_corenrn = True
        source = "explicit"
    elif runner_backend in {"sol_slurm", "slurm_remote"}:
        slurm_gpus = config.get("slurm_gpus")
        resolved_gpu = False if slurm_gpus in (None, "") else int(slurm_gpus) > 0
        resolved_corenrn = resolved_gpu
        source = "remote_slurm"
    else:
        resolved_corenrn = True
        resolved_gpu = True
        source = "local_default"

    return {
        "use_corenrn": resolved_corenrn,
        "use_gpu": resolved_gpu,
        "source": source,
    }


def deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``source`` into ``target`` in place."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def normalize_input_odors(value: Any) -> Any:
    """Convert JSON-decoded odor schedules back to numeric onset keys when possible."""
    if not isinstance(value, dict):
        return value

    normalized = {}
    for key, entry in value.items():
        try:
            time_key = float(key)
        except (TypeError, ValueError):
            time_key = key
        else:
            if isinstance(time_key, float) and time_key.is_integer():
                time_key = int(time_key)

        normalized[time_key] = deepcopy(entry)

    return normalized


def build_param_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Translate notebook controls into model param overrides."""
    record_from_somas = list(config.get("record_from_somas", ["MC", "TC", "GC"]))
    if config.get("enable_epl_interneurons"):
        epli_cell_type = str(config.get("epl_interneuron_cell_type") or "EPLI")
        if epli_cell_type not in record_from_somas:
            record_from_somas.append(epli_cell_type)

    overrides = {
        "sim_dt": float(config["sim_dt_ms"]),
        "recording_period": float(config.get("recording_period_ms", config["sim_dt_ms"])),
        "soma_trace_format": str(config.get("soma_trace_format", DEFAULT_SOMA_TRACE_FORMAT)),
        "soma_trace_dtype": str(config.get("soma_trace_dtype", DEFAULT_SOMA_TRACE_DTYPE)),
        "soma_spike_threshold": (
            None if config.get("soma_spike_threshold_mv") is None else float(config["soma_spike_threshold_mv"])
        ),
        "soma_spike_min_prominence_mv": float(
            config.get("soma_spike_min_prominence_mv", DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV)
        ),
        "soma_spike_refractory_ms": float(
            config.get("soma_spike_refractory_ms", DEFAULT_SOMA_SPIKE_REFRACTORY_MS)
        ),
        "enable_reciprocal_synapses": bool(config.get("enable_reciprocal_synapses", True)),
        "record_from_somas": record_from_somas,
        "record_gc_output_events": bool(config.get("record_gc_output_events", True)),
        "save_soma_traces": bool(config.get("save_soma_traces", True)),
        "save_voltage_summary": bool(config.get("save_voltage_summary", True)),
        "keep_native_lfp_debug_files": bool(config.get("keep_native_lfp_debug_files", False)),
        "lfp_electrode_location": list(config.get("lfp_electrode_location", [116, 1078, -61])),
    }
    if config.get("lfp_include_cell_types") is not None:
        value = config["lfp_include_cell_types"]
        overrides["lfp_include_cell_types"] = [value] if isinstance(value, str) else list(value)
    if config.get("lfp_exclude_cell_types") is not None:
        value = config["lfp_exclude_cell_types"]
        overrides["lfp_exclude_cell_types"] = [value] if isinstance(value, str) else list(value)
    if "enable_lfp" in config:
        overrides["enable_lfp"] = bool(config["enable_lfp"])
    if config.get("rnd_seed") is not None:
        overrides["rnd_seed"] = int(config["rnd_seed"])
    if config.get("input_odors") is not None:
        overrides["input_odors"] = normalize_input_odors(config["input_odors"])
    if config.get("input_stimuli") is not None:
        from olfactorybulb.inputs import serialize_input_stimuli

        raw = config["input_stimuli"]
        normalized = {}
        for key, value in raw.items():
            try:
                normalized_key = int(float(key)) if float(key).is_integer() else float(key)
            except (TypeError, ValueError):
                normalized_key = key
            normalized[normalized_key] = value
        json_safe, dill_blob = serialize_input_stimuli(normalized)
        if dill_blob is not None:
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".inputspec.dill",
                prefix="ob_",
            )
            tmp.write(dill_blob)
            tmp.close()
            overrides["_input_spec_file"] = tmp.name
        else:
            overrides["input_stimuli"] = json_safe
    if config.get("max_firing_rate_hz") is not None:
        overrides["max_firing_rate"] = float(config["max_firing_rate_hz"])
    if config.get("inhale_duration_ms") is not None:
        overrides["inhale_duration"] = float(config["inhale_duration_ms"])
    apply_hfo_runtime_overrides(config, overrides)
    if config.get("enable_epl_interneurons") is not None:
        overrides["enable_epl_interneurons"] = bool(config["enable_epl_interneurons"])
    if config.get("max_epl_interneurons") is not None:
        overrides["max_epl_interneurons"] = int(config["max_epl_interneurons"])
    if config.get("epl_interneuron_cell_type") is not None:
        overrides["epl_interneuron_cell_type"] = str(config["epl_interneuron_cell_type"])
    extra = dict(config.get("extra_overrides", {}))
    deep_update(overrides, extra)
    return overrides


def apply_param_override_object(params: Any, overrides: dict[str, Any]) -> None:
    """Apply notebook-style override dictionaries onto a paramset object."""
    for key, value in overrides.items():
        if key == "input_odors":
            value = normalize_input_odors(value)
        current = getattr(params, key, None)
        if isinstance(value, dict) and isinstance(current, dict):
            merged = deepcopy(current)
            deep_update(merged, deepcopy(value))
            setattr(params, key, merged)
        else:
            setattr(params, key, deepcopy(value))


def _is_snapshot_value(name: str, value: Any) -> bool:
    """Return ``True`` when a param attribute should be included in a JSON snapshot."""
    if name.startswith("_"):
        return False
    if isinstance(value, (staticmethod, classmethod, property)):
        return False
    if callable(value):
        return False
    return True


def snapshot_param_object(params: Any) -> dict[str, Any]:
    """Capture a JSON-ready snapshot of a paramset instance and its class defaults."""
    snapshot = {}

    for cls in reversed(type(params).__mro__):
        if cls is object:
            continue
        for name, value in vars(cls).items():
            if _is_snapshot_value(name, value):
                snapshot[name] = deepcopy(value)

    for name, value in vars(params).items():
        if _is_snapshot_value(name, value):
            snapshot[name] = deepcopy(value)

    snapshot["name"] = getattr(params, "name", type(params).__name__)
    return _json_ready(snapshot)


def resolve_paramset_defaults(paramset_name: str) -> dict[str, Any]:
    """Instantiate a paramset and snapshot its clean inherited defaults."""
    import olfactorybulb.model as obmodel

    params = getattr(obmodel, str(paramset_name))()
    return snapshot_param_object(params)


def resolve_effective_params(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the effective params used by a notebook run configuration."""
    import olfactorybulb.model as obmodel

    config = build_run_config(**(config or {}))
    params = getattr(obmodel, config["paramset"])()
    apply_param_override_object(params, build_param_overrides(config))
    if config.get("extra_overrides"):
        apply_param_override_object(params, config["extra_overrides"])

    input_odors_source = "override" if config.get("input_odors") is not None else "paramset"
    input_odors = deepcopy(getattr(params, "input_odors", {}))
    odor_names = sorted(
        {
            entry.get("name")
            for entry in input_odors.values()
            if isinstance(entry, dict) and entry.get("name")
        }
    )

    return {
        "paramset": config["paramset"],
        "input_odors_source": input_odors_source,
        "input_odors": input_odors,
        "n_odor_presentations": len(input_odors),
        "odor_names": odor_names,
        "max_firing_rate_hz": getattr(params, "max_firing_rate", None),
        "inhale_duration_ms": getattr(params, "inhale_duration", None),
        "mc_input_weight": getattr(params, "mc_input_weight", None),
        "tc_input_weight": getattr(params, "tc_input_weight", None),
        "mc_input_delay_ms": getattr(params, "mc_input_delay", None),
        "tc_input_delay_ms": getattr(params, "tc_input_delay", None),
        "lfp_electrode_location": deepcopy(getattr(params, "lfp_electrode_location", None)),
        "sim_dt_ms": getattr(params, "sim_dt", None),
        "recording_period_ms": getattr(params, "recording_period", None),
        "full_param_snapshot": snapshot_param_object(params),
    }


def extract_runtime_control_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Extract notebook-only runtime and analysis controls from a run config."""
    runtime_keys = [
        "mode",
        "runner_backend",
        "nranks",
        "mpi_exec",
        "use_corenrn",
        "use_gpu",
        "cell_permute",
        "label_prefix",
        "results_base",
        "disable_status_report",
        "parallel_timeout",
        "lfp_include_cell_types",
        "lfp_exclude_cell_types",
        "analysis_dt_ms",
        "spectrogram_signal",
        "spectrogram_max_freq_hz",
        "spectrogram_nperseg",
        "spectrogram_noverlap",
        "wavelet_signal",
        "max_voltage_traces_per_type",
        "max_spike_raster_cells_per_type",
        "gc_output_bin_ms",
        "gc_output_smooth_sigma_ms",
        "gc_output_max_connections",
        "gc_output_rate_normalization",
        "input_bin_ms",
        "input_smooth_sigma_ms",
        "input_max_segments",
        "input_rate_normalization",
        "sniff_count",
        "remote_host",
        "remote_repo_root",
        "remote_results_root",
        "remote_conda_activate_cmd",
        "remote_runtime_profiles",
        "remote_fallback_conda_activate_cmd",
        "remote_fast_node_feature",
        "remote_mechanism_profile",
        "remote_fallback_mechanism_profile",
        "remote_repo_mode",
        "remote_git_ref",
        "remote_git_fetch",
        "remote_poll_interval_s",
        "remote_live_status",
        "remote_live_logs",
        "remote_mpi_exec",
        "slurm_partition",
        "slurm_account",
        "slurm_time",
        "slurm_gpus",
        "slurm_cpus_per_task",
        "slurm_mem",
        "slurm_extra_args",
        "ssh_options",
        "ssh_keepalive_s",
        "ssh_transport",
        "remote_preserve_paramiko_session",
    ]
    snapshot = {key: _json_ready(config.get(key)) for key in runtime_keys if key in config}
    snapshot["resolved_execution_mode"] = _json_ready(resolve_execution_mode(config))
    return snapshot
