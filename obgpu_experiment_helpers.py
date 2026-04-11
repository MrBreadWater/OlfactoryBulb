"""Notebook-facing helpers for running, loading, and analyzing OBGPU simulations.

This module is the maintained convenience layer for the interactive notebooks in
``notebooks/``. It keeps heavy NEURON work in subprocesses when possible so
notebook reruns do not corrupt the live HOC state.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from base64 import b64encode
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from hashlib import sha1
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
try:
    import pywt
except ImportError:  # pragma: no cover - optional runtime dependency
    pywt = None
try:
    import pexpect
except ImportError:  # pragma: no cover - optional runtime dependency
    pexpect = None
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, lfilter, spectrogram, welch
from modify_model import (
    add_synaptic_connection,
    modify_synaptic_connection,
    perform_cell_type_swaps,
    build_synapse_map
)

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "benchmarks" / "benchmark_ob.py"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "notebook_runs"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
CONTROL_HELP = {
    "mode": "Use 'fast' for 1-rank exploration or 'parity' for exact match to a previous version.",
    "nranks": "MPI rank count for the run. 1 is faster on this machine",
    "tstop_ms": "Simulation duration in ms. Use None to keep the paramset default.",
    "sim_dt_ms": "Requested simulation dt in ms.",
    "recording_period_ms": "Saved sample period for LFP and soma traces.",
    "legacy_parallel_dt": "When True, preserve the older parallel dt behavior. When False, let sim_dt_ms control dt more directly.",
    "lfp_electrode_location": "Probe location as [x, y, z] in microns.",
    "rnd_seed": "Random seed for odor input generation.",
    "record_from_somas": "Which cell types to record from, e.g. ['MC', 'TC', 'GC'].",
    "record_gc_output_events": "Record reciprocal GC->MC/TC GABA event times for direct inhibitory-output plots.",
    "keep_native_lfp_debug_files": "Keep raw CoreNEURON native-LFP TSV/config artifacts instead of deleting them after lfp.pkl is written.",
    "gc_output_bin_ms": "Bin width in ms for the GC inhibitory-output population-rate plot.",
    "gc_output_smooth_sigma_ms": "Gaussian smoothing sigma in ms for the GC inhibitory-output rate plot.",
    "gc_output_max_connections": "Maximum reciprocal GABA connections to include in the GC-output raster.",
    "gc_output_rate_normalization": "How to normalize GC inhibitory-output rates: 'per_target_cell', 'per_connection', 'per_source_cell', or 'total'.",
    "input_bin_ms": "Bin width in ms for the odor-input event-rate plot.",
    "input_smooth_sigma_ms": "Gaussian smoothing sigma in ms for the odor-input event-rate plot.",
    "input_max_segments": "Maximum odor-input target segments to include in the input raster.",
    "input_rate_normalization": "How to normalize odor-input rates: 'per_target_cell', 'per_segment', or 'total'.",
    "input_odors": "Full odor schedule dict keyed by onset ms.",
    "max_firing_rate_hz": "Maximum ORN firing rate.",
    "inhale_duration_ms": "Inhalation duration in ms.",
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
    "gaba_gmax": "Global GabaSyn gmax.",
    "gaba_tau2_ms": "Global GabaSyn tau2.",
    "enable_reciprocal_synapses": "Toggle GC<->MC/TC reciprocal synapses.",
    "extra_overrides": "Any raw paramset overrides not exposed above.",
    "spectrogram_signal": "Signal for spectrogram plots, e.g. 'lfp', 'mean_MC_voltage', or 'MC5[0].soma'.",
    "wavelet_signal": "Signal for wavelet plots, e.g. 'lfp', 'mean_TC_voltage', or a soma label.",
    "runner_backend": "Execution backend: 'local' or 'sol_slurm'.",
    "mpi_exec": "MPI launcher for local notebook runs, e.g. 'mpiexec' or 'srun --mpi=pmix'.",
    "remote_mpi_exec": "MPI launcher on the remote host, e.g. 'srun --mpi=pmix' or 'mpiexec'.",
    "remote_host": "SSH target used by the Sol backend, e.g. 'user@sol.asu.edu'.",
    "remote_repo_root": "Absolute repo path on Sol.",
    "remote_results_root": "Remote root directory where timestamped notebook runs are written.",
    "remote_conda_activate_cmd": "Shell snippet used on Sol before launching the benchmark command. Defaults to 'source tools/setup/activate_sol_obgpu.sh'.",
    "remote_git_ref": "Git commit, tag, or branch to check out on Sol before running. Defaults to the current local HEAD commit.",
    "remote_git_fetch": "When True, fetch the configured remote on Sol before checking out remote_git_ref.",
    "remote_git_remote": "Git remote name on Sol used when remote_git_fetch=True. Defaults to 'origin'.",
    "remote_poll_interval_s": "Polling interval in seconds for remote Slurm jobs.",
    "slurm_partition": "Optional Slurm partition for remote submission. Use 'arm' on Sol for Grace Hopper.",
    "slurm_account": "Optional Slurm account for remote submission.",
    "slurm_time": "Optional Slurm walltime, e.g. '02:00:00'.",
    "slurm_gpus": "Optional GPU count requested from Slurm.",
    "slurm_cpus_per_task": "Optional CPU count requested per Slurm task.",
    "slurm_mem": "Optional Slurm memory request, e.g. '32G'.",
    "slurm_extra_args": "Optional extra sbatch arguments passed as raw strings.",
    "ssh_binary": "SSH client executable used by the remote backend.",
    "ssh_options": "Extra SSH options, e.g. ['-J', 'jumphost'].",
    "ssh_multiplex": "When True, reuse a persistent SSH control socket so Sol auth happens once instead of on every submit/poll/sync step.",
    "ssh_allow_interactive_auth": "When True, open the initial SSH control master through an interactive PTY so notebook-launched Sol runs can prompt for password and 2FA once.",
    "ssh_control_path": "Optional path for the shared SSH control socket. Defaults to a hashed path under XDG_RUNTIME_DIR, TMPDIR, or the system temp directory.",
    "ssh_control_persist_s": "How long to keep the SSH control master alive after the last use.",
    "rsync_binary": "rsync executable used to sync remote results back locally.",
    "rsync_options": "Extra rsync options used by the remote backend.",
    "add_connections": "Add new connections between existing neurons.",
    "modify_connections": "Modify the synaptic weight between two specific neurons.",
    "swap_cell_types": "A list of cells to swap to another cell type."
}


@dataclass
class RunRecord:
    """Metadata and captured stdout/stderr for a timestamped notebook run."""

    label: str
    timestamp: str
    result_dir: Path
    summary: dict
    config: dict
    overrides: dict
    command: list[str]
    stdout: str
    stderr: str


_LIVE_INSPECTION_MODEL = None
_LIVE_INSPECTION_SIGNATURE = None
_LIVE_SSH_MASTERS: dict[str, Any] = {}


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
    return "srun --mpi=pmix"


def make_timestamp() -> str:
    """Return a timestamp string using the notebook-run naming convention."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def build_run_config(**overrides: Any) -> dict[str, Any]:
    """Build a normalized notebook control dictionary."""
    mode = overrides.pop("mode", "fast")
    base = {
        "mode": mode,
        "paramset": "GammaSignature",
        "label_prefix": "obgpu_experiment",
        "results_base": str(DEFAULT_RESULTS_BASE),
        "nranks": 1 if mode == "fast" else 2,
        "use_corenrn": True,
        "use_gpu": True,
        "cell_permute": 2,
        "tstop_ms": None,
        "sim_dt_ms": 0.1,
        "recording_period_ms": 0.1,
        "legacy_parallel_dt": False if mode == "fast" else True,
        "enable_lfp": True,
        "disable_status_report": True,
        "parallel_timeout": None,
        "rnd_seed": None,
        "record_from_somas": ["MC", "TC", "GC"],
        "record_gc_output_events": True,
        "keep_native_lfp_debug_files": False,
        "enable_reciprocal_synapses": True,
        "gc_output_bin_ms": 5.0,
        "gc_output_smooth_sigma_ms": 10.0,
        "gc_output_max_connections": 120,
        "gc_output_rate_normalization": "per_target_cell",
        "input_bin_ms": 5.0,
        "input_smooth_sigma_ms": 10.0,
        "input_max_segments": 120,
        "input_rate_normalization": "per_target_cell",
        "lfp_electrode_location": [116, 1078, -61],
        "input_odors": None,
        "max_firing_rate_hz": None,
        "inhale_duration_ms": None,
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
        "gaba_gmax": None,
        "gaba_tau2_ms": None,
        "analysis_dt_ms": 0.1,
        "spectrogram_signal": "lfp",
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
        "remote_conda_activate_cmd": "source tools/setup/activate_sol_obgpu.sh",
        "remote_git_ref": None,
        "remote_git_fetch": True,
        "remote_git_remote": "origin",
        "remote_poll_interval_s": 10.0,
        "slurm_partition": "arm",
        "slurm_account": None,
        "slurm_time": None,
        "slurm_gpus": 1,
        "slurm_cpus_per_task": None,
        "slurm_mem": None,
        "slurm_extra_args": [],
        "ssh_binary": "ssh",
        "ssh_options": [],
        "ssh_multiplex": True,
        "ssh_allow_interactive_auth": True,
        "ssh_control_path": None,
        "ssh_control_persist_s": 28800,
        "rsync_binary": "rsync",
        "rsync_options": ["-az"],
        "add_connections": [],
        "modify_connections": [],
        "swap_cell_types": []
    }
    base.update(overrides)
    return base


def build_sol_remote_config(
    *,
    remote_host: str,
    remote_repo_root: str | Path,
    remote_results_root: str | Path | None = None,
    slurm_partition: str = "arm",
    slurm_account: str | None = None,
    slurm_time: str = "02:00:00",
    slurm_gpus: int = 1,
    slurm_cpus_per_task: int | None = None,
    slurm_mem: str | None = None,
    remote_poll_interval_s: float = 15.0,
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = True,
    remote_git_remote: str = "origin",
    ssh_options: list[str] | None = None,
    rsync_options: list[str] | None = None,
    slurm_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Return a Sol-specific remote runner config with Grace Hopper defaults.

    This helper is the supported notebook-facing way to send runs to Sol. By
    default it chooses the `arm` partition for Grace Hopper, uses the Sol
    activation helper, and launches the benchmark through `srun --mpi=pmix`.
    """
    remote_repo_root = str(remote_repo_root)
    if remote_results_root is None:
        remote_results_root = str(PurePosixPath(remote_repo_root) / "results" / "notebook_runs")

    return {
        "runner_backend": "sol_slurm",
        "remote_host": str(remote_host),
        "remote_repo_root": remote_repo_root,
        "remote_results_root": str(remote_results_root),
        "remote_conda_activate_cmd": "source tools/setup/activate_sol_obgpu.sh",
        "remote_mpi_exec": default_remote_mpi_exec(),
        "remote_poll_interval_s": float(remote_poll_interval_s),
        "remote_git_ref": remote_git_ref,
        "remote_git_fetch": bool(remote_git_fetch),
        "remote_git_remote": str(remote_git_remote),
        "slurm_partition": str(slurm_partition),
        "slurm_account": slurm_account,
        "slurm_time": str(slurm_time),
        "slurm_gpus": int(slurm_gpus),
        "slurm_cpus_per_task": slurm_cpus_per_task,
        "slurm_mem": slurm_mem,
        "slurm_extra_args": list(slurm_extra_args or []),
        "ssh_options": list(ssh_options or []),
        "rsync_options": list(rsync_options or ["-az"]),
    }


def make_label(config: dict[str, Any], timestamp: str | None = None) -> str:
    """Build the timestamped notebook label for a run configuration."""
    timestamp = timestamp or make_timestamp()
    mode = str(config.get("mode", "run"))
    paramset = str(config.get("paramset", "Paramset"))
    prefix = str(config.get("label_prefix", "obgpu_experiment"))
    return f"{prefix}_{paramset}_{mode}_{timestamp}"


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
    overrides = {
        "sim_dt": float(config["sim_dt_ms"]),
        "recording_period": float(config.get("recording_period_ms", config["sim_dt_ms"])),
        "legacy_parallel_dt": bool(config.get("legacy_parallel_dt", True)),
        "enable_reciprocal_synapses": bool(config.get("enable_reciprocal_synapses", True)),
        "record_from_somas": list(config.get("record_from_somas", ["MC", "TC", "GC"])),
        "record_gc_output_events": bool(config.get("record_gc_output_events", True)),
        "keep_native_lfp_debug_files": bool(config.get("keep_native_lfp_debug_files", False)),
        "lfp_electrode_location": list(config.get("lfp_electrode_location", [116, 1078, -61])),
    }
    if "enable_lfp" in config:
        overrides["enable_lfp"] = bool(config["enable_lfp"])
    if config.get("rnd_seed") is not None:
        overrides["rnd_seed"] = int(config["rnd_seed"])
    if config.get("input_odors") is not None:
        overrides["input_odors"] = normalize_input_odors(config["input_odors"])
    if config.get("max_firing_rate_hz") is not None:
        overrides["max_firing_rate"] = float(config["max_firing_rate_hz"])
    if config.get("inhale_duration_ms") is not None:
        overrides["inhale_duration"] = float(config["inhale_duration_ms"])
    if config.get("input_syn_tau1_ms") is not None:
        overrides["input_syn_tau1"] = float(config["input_syn_tau1_ms"])
    if config.get("input_syn_tau2_ms") is not None:
        overrides["input_syn_tau2"] = float(config["input_syn_tau2_ms"])
    if config.get("mc_input_weight") is not None:
        overrides["mc_input_weight"] = float(config["mc_input_weight"])
    if config.get("tc_input_weight") is not None:
        overrides["tc_input_weight"] = float(config["tc_input_weight"])
    if config.get("mc_input_delay_ms") is not None:
        overrides["mc_input_delay"] = float(config["mc_input_delay_ms"])
    if config.get("tc_input_delay_ms") is not None:
        overrides["tc_input_delay"] = float(config["tc_input_delay_ms"])
    if config.get("gap_mc") is not None or config.get("gap_tc") is not None:
        overrides.setdefault("gap_juction_gmax", {})
        if config.get("gap_mc") is not None:
            overrides["gap_juction_gmax"]["MC"] = float(config["gap_mc"])
        if config.get("gap_tc") is not None:
            overrides["gap_juction_gmax"]["TC"] = float(config["gap_tc"])
    if any(
        config.get(key) is not None
        for key in ("ampa_nmda_gmax", "ampa_nmda_nmdafactor", "gaba_gmax", "gaba_tau2_ms")
    ):
        overrides.setdefault("synapse_properties", {})
    if config.get("ampa_nmda_gmax") is not None or config.get("ampa_nmda_nmdafactor") is not None:
        overrides["synapse_properties"].setdefault("AmpaNmdaSyn", {})
        if config.get("ampa_nmda_gmax") is not None:
            overrides["synapse_properties"]["AmpaNmdaSyn"]["gmax"] = float(config["ampa_nmda_gmax"])
        if config.get("ampa_nmda_nmdafactor") is not None:
            overrides["synapse_properties"]["AmpaNmdaSyn"]["nmdafactor"] = float(
                config["ampa_nmda_nmdafactor"]
            )
    if config.get("gaba_gmax") is not None or config.get("gaba_tau2_ms") is not None:
        overrides["synapse_properties"].setdefault("GabaSyn", {})
        if config.get("gaba_gmax") is not None:
            overrides["synapse_properties"]["GabaSyn"]["gmax"] = float(config["gaba_gmax"])
        if config.get("gaba_tau2_ms") is not None:
            overrides["synapse_properties"]["GabaSyn"]["tau2"] = float(config["gaba_tau2_ms"])
    extra = dict(config.get("extra_overrides", {}))
    deep_update(overrides, extra)
    return overrides


def available_controls() -> dict[str, str]:
    """Return the notebook control catalog."""
    return dict(CONTROL_HELP)


def print_available_controls() -> None:
    """Pretty-print the notebook control catalog."""
    print(json.dumps(available_controls(), indent=2, sort_keys=True))

def add_new_connections(ob, new_connections_config):
    """Create new synaptic connections described by notebook config entries."""
    for config in new_connections_config:
        add_synaptic_connection(ob, config)

def modify_existing_connections(ob, modifications_config):
    """Apply in-place edits to existing synapses described by notebook config entries."""
    synapse_map = build_synapse_map(ob)
    for config in modifications_config:
        modify_synaptic_connection(ob, synapse_map, config)

def build_run_command(
    config: dict[str, Any],
    label: str,
    *,
    repo_root: str | Path | None = None,
    results_base: str | Path | None = None,
    mpi_exec: str | None = None,
) -> list[str]:
    """Build the benchmark subprocess command for a notebook run."""
    repo_root = repo_root or REPO_ROOT
    results_base = results_base or config.get("results_base", DEFAULT_RESULTS_BASE)
    mpi_exec = mpi_exec or str(config.get("mpi_exec", default_local_mpi_exec()))
    benchmark_script = Path(repo_root) / "tools" / "benchmarks" / "benchmark_ob.py"
    command = [
        *shlex.split(mpi_exec),
        "-n",
        str(int(config["nranks"])),
        "nrniv",
        "-mpi",
        "-python",
        str(benchmark_script),
        "--repo-root",
        str(repo_root),
        "--paramset",
        str(config["paramset"]),
        "--label",
        label,
        "--results-base",
        str(results_base),
        "--overrides-json",
        json.dumps(build_param_overrides(config), sort_keys=True),
    ]

    if config.get("tstop_ms") is not None:
        command.extend(["--tstop-override", str(float(config["tstop_ms"]))])

    if config.get("use_corenrn", True):
        command.append("--coreneuron")
    if config.get("use_gpu", True):
        command.append("--coreneuron-gpu")
    if config.get("disable_status_report", True):
        command.append("--disable-status-report")
    if not config.get("enable_lfp", True):
        command.append("--disable-lfp-electrode")
    if config.get("parallel_timeout") is not None:
        command.extend(["--parallel-timeout", str(float(config["parallel_timeout"]))])

    return command


def _shell_join(command: list[str] | tuple[str, ...]) -> str:
    """Return a POSIX-safe shell rendering of a command list."""
    return shlex.join([str(part) for part in command])


def _remote_repo_root(config: dict[str, Any]) -> PurePosixPath:
    """Return the configured repo root on the remote Sol host."""
    remote_repo_root = config.get("remote_repo_root")
    if not remote_repo_root:
        raise ValueError("runner_backend='sol_slurm' requires remote_repo_root")
    return PurePosixPath(str(remote_repo_root))


def _remote_results_root(config: dict[str, Any]) -> PurePosixPath:
    """Return the configured results root on the remote Sol host."""
    configured = config.get("remote_results_root")
    if configured:
        return PurePosixPath(str(configured))
    return _remote_repo_root(config) / "results" / "notebook_runs"


def _resolve_local_git_head() -> str | None:
    """Return the current local git HEAD commit or ``None`` when unavailable."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    head = (completed.stdout or "").strip()
    return head or None


def _resolve_remote_git_ref(config: dict[str, Any]) -> str | None:
    """Return the requested Sol git ref, defaulting to the current local HEAD commit."""
    configured = config.get("remote_git_ref")
    if configured not in (None, ""):
        return str(configured)
    return _resolve_local_git_head()


def _require_remote_host(config: dict[str, Any]) -> str:
    """Return the configured remote SSH target."""
    remote_host = str(config.get("remote_host") or "").strip()
    if not remote_host:
        raise ValueError("runner_backend='sol_slurm' requires remote_host")
    return remote_host


def _ssh_control_path(config: dict[str, Any]) -> Path | None:
    """Return the shared SSH control-socket path for the remote backend."""
    if not bool(config.get("ssh_multiplex", True)):
        return None

    configured = config.get("ssh_control_path")
    if configured not in (None, ""):
        return Path(str(configured)).expanduser()

    host = _require_remote_host(config)
    digest = sha1(host.encode("utf-8")).hexdigest()[:12]
    runtime_base = (
        os.environ.get("XDG_RUNTIME_DIR")
        or os.environ.get("TMPDIR")
        or tempfile.gettempdir()
    )
    return Path(runtime_base) / f"obgpu-ssh-{digest}.sock"


def _ssh_common_options(config: dict[str, Any]) -> list[str]:
    """Return SSH options shared by submit, poll, and rsync operations."""
    options = [str(option) for option in config.get("ssh_options", [])]
    control_path = _ssh_control_path(config)
    if control_path is not None:
        options.extend(
            [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPath={control_path}",
                "-o",
                f"ControlPersist={int(config.get('ssh_control_persist_s', 28800))}s",
            ]
        )
    return options


def _ssh_command_env() -> dict[str, str]:
    """Return a stable environment for non-interactive SSH subprocesses."""
    env = os.environ.copy()
    # The notebook backend handles first-use auth through pexpect when needed.
    # Do not fall back to GUI askpass helpers from plain subprocess calls.
    env["SSH_ASKPASS_REQUIRE"] = "never"
    return env


def _ssh_master_is_ready(config: dict[str, Any], control_path: Path) -> bool:
    """Return True when the configured SSH control master accepts mux requests."""
    checked = subprocess.run(
        [
            str(config.get("ssh_binary", "ssh")),
            *[str(option) for option in config.get("ssh_options", [])],
            "-o",
            f"ControlPath={control_path}",
            "-O",
            "check",
            _require_remote_host(config),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_ssh_command_env(),
    )
    return checked.returncode == 0


def _ssh_master_socket_is_live(config: dict[str, Any], control_path: Path) -> bool:
    """Return True when the control socket exists and appears to back a live master."""
    if not control_path.exists():
        return False
    return _ssh_master_is_ready(config, control_path)


def _kill_stale_ssh_master_processes(config: dict[str, Any], control_path: Path) -> None:
    """Kill stale SSH master processes still tied to a removed or broken control path."""
    listed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return

    control_path_text = str(control_path)
    current_pid = os.getpid()
    target_pids: list[int] = []
    for raw_line in (listed.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if control_path_text not in args:
            continue
        if "ssh" not in args:
            continue
        target_pids.append(pid)

    for sig in (signal.SIGTERM, signal.SIGKILL):
        survivors: list[int] = []
        for pid in target_pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            except OSError:
                survivors.append(pid)
                continue
            survivors.append(pid)
        if not survivors:
            return
        time.sleep(0.2)
        target_pids = []
        for pid in survivors:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except OSError:
                continue
            target_pids.append(pid)
        if not target_pids:
            return


def _reset_ssh_master(config: dict[str, Any]) -> None:
    """Tear down and remove any cached SSH control socket for the remote host."""
    control_path = _ssh_control_path(config)
    if control_path is None:
        return

    stored_child = _LIVE_SSH_MASTERS.pop(str(control_path), None)
    if stored_child is not None:
        try:
            if hasattr(stored_child, "terminate"):
                stored_child.terminate()
            else:
                stored_child.close(force=True)
        except Exception:
            pass

    host = _require_remote_host(config)
    ssh_binary = str(config.get("ssh_binary", "ssh"))
    user_options = [str(option) for option in config.get("ssh_options", [])]
    exit_command = [
        ssh_binary,
        *user_options,
        "-o",
        f"ControlPath={control_path}",
        "-O",
        "exit",
        host,
    ]
    subprocess.run(
        exit_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_ssh_command_env(),
    )
    if control_path.exists():
        try:
            control_path.unlink()
        except OSError:
            pass
    _kill_stale_ssh_master_processes(config, control_path)


def _ssh_failure_needs_reset(stderr: str) -> bool:
    """Return True when SSH stderr indicates a stale or unusable control master."""
    message = (stderr or "").lower()
    return any(
        token in message
        for token in (
            "master refused session request",
            "control socket connect",
            "permission denied",
            "connection reset",
            "broken pipe",
        )
    )


def _ensure_ssh_master(config: dict[str, Any]) -> None:
    """Ensure a reusable SSH control master exists for the configured remote host."""
    control_path = _ssh_control_path(config)
    if control_path is None:
        return

    control_path.parent.mkdir(parents=True, exist_ok=True)
    host = _require_remote_host(config)
    ssh_binary = str(config.get("ssh_binary", "ssh"))
    user_options = [str(option) for option in config.get("ssh_options", [])]
    persist_seconds = int(config.get("ssh_control_persist_s", 28800))

    if _ssh_master_is_ready(config, control_path):
        return

    _reset_ssh_master(config)

    start_command = [
        ssh_binary,
        *user_options,
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=yes",
        "-o",
        f"ControlPath={control_path}",
        "-o",
        f"ControlPersist={persist_seconds}s",
        "-MN",
        host,
    ]
    started = subprocess.Popen(
        start_command,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_ssh_command_env(),
    )
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _ssh_master_socket_is_live(config, control_path):
            _LIVE_SSH_MASTERS[str(control_path)] = started
            return
        if started.poll() is not None:
            break
        time.sleep(0.1)

    stdout, stderr = started.communicate(timeout=1) if started.poll() is not None else ("", "")
    if started.poll() is None:
        try:
            started.terminate()
            started.wait(timeout=1)
        except Exception:
            try:
                started.kill()
            except Exception:
                pass
        stdout = stdout or ""
        stderr = stderr or ""

    if not bool(config.get("ssh_allow_interactive_auth", True)):
        raise RuntimeError(
            "Could not establish a persistent SSH control master for the Sol backend.\n"
            f"Host: {host}\n"
            f"Stdout:\n{stdout}\n\nStderr:\n{stderr}"
        )
    _reset_ssh_master(config)
    _start_ssh_master_interactive(config, start_command)


def _start_ssh_master_interactive(config: dict[str, Any], start_command: list[str]) -> None:
    """Start the SSH control master through a PTY so notebook users can answer auth prompts."""
    if pexpect is None:
        raise RuntimeError(
            "Interactive SSH auth requires the optional 'pexpect' dependency in the notebook environment."
        )

    interactive_command: list[str] = []
    index = 0
    while index < len(start_command):
        part = start_command[index]
        if (
            part == "-o"
            and index + 1 < len(start_command)
            and start_command[index + 1] == "BatchMode=yes"
        ):
            index += 2
            continue
        interactive_command.append(part)
        index += 1
    control_path = _ssh_control_path(config)
    if control_path is None:
        raise RuntimeError("Interactive SSH auth requires ssh_multiplex=True")

    child = pexpect.spawn(
        interactive_command[0],
        interactive_command[1:],
        cwd=str(REPO_ROOT),
        encoding="utf-8",
        timeout=15,
    )

    password_prompt = re.compile(r"(?i)(?:^|\n).*(?:password|passphrase).{0,20}:\s*$")
    otp_prompt = re.compile(r"(?i)(?:^|\n).*(?:passcode|verification|otp|token|duo|2fa|two-factor).{0,40}:\s*$")
    generic_prompt = re.compile(r"(?i)(?:^|\n).{0,120}:\s*$")
    hostkey_prompt = re.compile(r"(?i)are you sure you want to continue connecting")

    deadline = time.time() + 180.0
    auth_answered = False

    while True:
        idx = child.expect(
            [
                hostkey_prompt,
                password_prompt,
                otp_prompt,
                generic_prompt,
                pexpect.EOF,
                pexpect.TIMEOUT,
            ]
        )
        if idx == 0:
            child.sendline("yes")
            auth_answered = True
            continue
        if idx == 1:
            prompt = child.after.strip() or f"Password for {_require_remote_host(config)}: "
            child.sendline(getpass(prompt + " "))
            auth_answered = True
            continue
        if idx == 2:
            prompt = child.after.strip() or f"2FA for {_require_remote_host(config)}: "
            child.sendline(input(prompt + " "))
            auth_answered = True
            continue
        if idx == 3:
            prompt = child.after.strip() or f"SSH prompt for {_require_remote_host(config)}: "
            if "password" in prompt.lower() or "passphrase" in prompt.lower():
                child.sendline(getpass(prompt + " "))
            else:
                child.sendline(input(prompt + " "))
            auth_answered = True
            continue
        if idx == 4:
            child.close()
            if child.exitstatus == 0:
                if _ssh_master_socket_is_live(config, control_path):
                    _LIVE_SSH_MASTERS[str(control_path)] = child
                    return
            raise RuntimeError(
                "Interactive SSH authentication failed while opening the Sol control master.\n"
                f"Exit status: {child.exitstatus}\n"
                f"Output:\n{child.before}"
            )
        if control_path.exists() and child.isalive() and auth_answered:
            _LIVE_SSH_MASTERS[str(control_path)] = child
            return
        if time.time() > deadline:
            child.close(force=True)
            raise RuntimeError(
                "Timed out while waiting for the Sol SSH control master to authenticate.\n"
                f"Partial output:\n{child.before}"
            )


def _ssh_prefix(config: dict[str, Any]) -> list[str]:
    """Build the SSH command prefix used for Sol orchestration."""
    _ensure_ssh_master(config)
    prefix = [str(config.get("ssh_binary", "ssh"))]
    prefix.extend(_ssh_common_options(config))
    prefix.append(_require_remote_host(config))
    return prefix


def _run_ssh_shell(
    config: dict[str, Any],
    remote_shell_command: str,
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one shell command on the remote Sol host over SSH."""
    command = _ssh_prefix(config) + ["bash", "-lc", remote_shell_command]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
        env=_ssh_command_env(),
    )
    if (
        completed.returncode != 0
        and bool(config.get("ssh_multiplex", True))
        and _ssh_failure_needs_reset(completed.stderr or "")
    ):
        _reset_ssh_master(config)
        _ensure_ssh_master(config)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=check,
            env=_ssh_command_env(),
        )
    return completed


def _sync_remote_result_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Sync one remote result directory back into the local notebook results tree."""
    local_result_dir.mkdir(parents=True, exist_ok=True)
    _ensure_ssh_master(config)
    command = [str(config.get("rsync_binary", "rsync"))]
    command.extend(str(option) for option in config.get("rsync_options", ["-az"]))
    ssh_command = [str(config.get("ssh_binary", "ssh"))]
    ssh_command.extend(_ssh_common_options(config))
    if len(ssh_command) > 1:
        command.extend(["-e", _shell_join(ssh_command)])
    command.extend(
        [
            f"{_require_remote_host(config)}:{remote_result_dir.as_posix().rstrip('/')}/",
            str(local_result_dir) + "/",
        ]
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_ssh_command_env(),
    )
    if (
        completed.returncode != 0
        and bool(config.get("ssh_multiplex", True))
        and _ssh_failure_needs_reset(completed.stderr or "")
    ):
        _reset_ssh_master(config)
        _ensure_ssh_master(config)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=_ssh_command_env(),
        )
    return completed


def _build_remote_submit_command(
    config: dict[str, Any],
    *,
    label: str,
    remote_repo_root: PurePosixPath,
    remote_results_root: PurePosixPath,
    benchmark_command: list[str],
    remote_git_ref: str | None,
) -> str:
    """Build the remote `submit_sol_run.py` invocation shell line."""
    remote_helper = remote_repo_root / "tools" / "remote" / "submit_sol_run.py"
    benchmark_b64 = b64encode(json.dumps(benchmark_command).encode("utf-8")).decode("ascii")
    command = [
        "python",
        str(remote_helper),
        "--repo-root",
        remote_repo_root.as_posix(),
        "--results-base",
        remote_results_root.as_posix(),
        "--label",
        label,
        "--benchmark-command-b64",
        benchmark_b64,
        "--conda-activate-cmd",
        str(config.get("remote_conda_activate_cmd")),
    ]

    if remote_git_ref:
        command.extend(["--git-ref", remote_git_ref])
    if bool(config.get("remote_git_fetch", True)):
        command.append("--git-fetch")
        command.extend(["--git-remote", str(config.get("remote_git_remote", "origin"))])

    for key, flag in (
        ("slurm_partition", "--partition"),
        ("slurm_account", "--account"),
        ("slurm_time", "--time"),
        ("slurm_mem", "--mem"),
    ):
        value = config.get(key)
        if value not in (None, ""):
            command.extend([flag, str(value)])

    for key, flag in (
        ("slurm_gpus", "--gpus"),
        ("slurm_cpus_per_task", "--cpus-per-task"),
    ):
        value = config.get(key)
        if value not in (None, ""):
            command.extend([flag, str(int(value))])

    for extra in config.get("slurm_extra_args", []):
        command.extend(["--sbatch-arg", str(extra)])

    return _shell_join(command)


def _build_remote_poll_command(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    remote_result_dir: PurePosixPath,
    job_id: str,
) -> str:
    """Build the remote `poll_sol_run.py` invocation shell line."""
    remote_helper = remote_repo_root / "tools" / "remote" / "poll_sol_run.py"
    command = [
        "python",
        str(remote_helper),
        "--job-id",
        str(job_id),
        "--result-dir",
        remote_result_dir.as_posix(),
    ]
    return _shell_join(command)


def _remote_submission_payload(
    config: dict[str, Any],
    *,
    label: str,
) -> tuple[PurePosixPath, PurePosixPath, list[str], dict[str, Any], str]:
    """Prepare the remote paths and benchmark command for a Sol run."""
    remote_repo_root = _remote_repo_root(config)
    remote_results_root = _remote_results_root(config)
    remote_git_ref = _resolve_remote_git_ref(config)
    remote_mpi_exec = config.get("remote_mpi_exec") or default_remote_mpi_exec()
    remote_command = build_run_command(
        config,
        label,
        repo_root=remote_repo_root,
        results_base=remote_results_root,
        mpi_exec=str(remote_mpi_exec),
    )
    submit_command = _build_remote_submit_command(
        config,
        label=label,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        benchmark_command=remote_command,
        remote_git_ref=remote_git_ref,
    )
    return (
        remote_repo_root,
        remote_results_root,
        remote_command,
        {
            "runner_backend": "sol_slurm",
            "remote_host": _require_remote_host(config),
            "remote_repo_root": remote_repo_root.as_posix(),
            "remote_results_root": remote_results_root.as_posix(),
            "remote_mpi_exec": str(remote_mpi_exec),
            "remote_git_ref": remote_git_ref,
            "remote_git_fetch": bool(config.get("remote_git_fetch", True)),
            "remote_git_remote": str(config.get("remote_git_remote", "origin")),
        },
        submit_command,
    )


def _run_remote_simulation(
    config: dict[str, Any],
    *,
    label: str,
    timestamp: str,
    local_result_dir: Path,
) -> RunRecord:
    """Submit one Sol Slurm job, wait for completion, sync results, and return a run record."""
    (
        remote_repo_root,
        _remote_results_root_value,
        remote_benchmark_command,
        remote_metadata,
        submit_shell,
    ) = _remote_submission_payload(config, label=label)

    submit_completed = _run_ssh_shell(config, submit_shell)
    local_result_dir.mkdir(parents=True, exist_ok=True)
    (local_result_dir / "submit_stdout.txt").write_text(submit_completed.stdout or "")
    (local_result_dir / "submit_stderr.txt").write_text(submit_completed.stderr or "")

    if submit_completed.returncode != 0:
        completed = SimpleNamespace(
            returncode=submit_completed.returncode,
            stdout=submit_completed.stdout or "",
            stderr=submit_completed.stderr or "",
        )
        _write_notebook_run_info(
            local_result_dir,
            config=config,
            label=label,
            timestamp=timestamp,
            command=remote_benchmark_command,
            env={},
            completed=completed,
            extra_payload={"remote": remote_metadata},
        )
        raise RuntimeError(
            "Remote Sol submission failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Submit stderr:\n{submit_completed.stderr}"
        )

    try:
        submission = json.loads((submit_completed.stdout or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Remote Sol submission did not return valid JSON.\n"
            f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
        ) from exc

    remote_result_dir = PurePosixPath(submission["result_dir"])
    poll_interval_s = max(float(config.get("remote_poll_interval_s", 10.0)), 1.0)
    poll_transcript: list[dict[str, Any]] = []
    final_status: dict[str, Any] | None = None

    while True:
        poll_shell = _build_remote_poll_command(
            config,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_result_dir,
            job_id=str(submission["job_id"]),
        )
        poll_completed = _run_ssh_shell(config, poll_shell)
        if poll_completed.returncode != 0:
            raise RuntimeError(
                "Remote Sol status poll failed.\n"
                f"Stdout:\n{poll_completed.stdout}\n\nStderr:\n{poll_completed.stderr}"
            )

        try:
            status = json.loads((poll_completed.stdout or "").strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Remote Sol poll did not return valid JSON.\n"
                f"Stdout:\n{poll_completed.stdout}\n\nStderr:\n{poll_completed.stderr}"
            ) from exc

        poll_transcript.append(status)
        if status.get("done"):
            final_status = status
            break
        time.sleep(poll_interval_s)

    sync_completed = _sync_remote_result_dir(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
    )
    (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
    (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
    if sync_completed.returncode != 0:
        raise RuntimeError(
            "Remote Sol result sync failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"rsync stderr:\n{sync_completed.stderr}"
        )

    stdout_text = (local_result_dir / "stdout.txt").read_text() if (local_result_dir / "stdout.txt").exists() else ""
    stderr_text = (local_result_dir / "stderr.txt").read_text() if (local_result_dir / "stderr.txt").exists() else ""
    remote_git_commit = (
        (local_result_dir / "git_commit.txt").read_text().strip()
        if (local_result_dir / "git_commit.txt").exists()
        else None
    )
    remote_git_ref = (
        (local_result_dir / "git_ref.txt").read_text().strip()
        if (local_result_dir / "git_ref.txt").exists()
        else remote_metadata.get("remote_git_ref")
    )
    returncode = 0 if final_status and final_status.get("ok") else 1
    completed = SimpleNamespace(returncode=returncode, stdout=stdout_text, stderr=stderr_text)

    summary_path = local_result_dir / "summary.json"
    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    _write_notebook_run_info(
        local_result_dir,
        config=config,
        label=label,
        timestamp=timestamp,
        command=remote_benchmark_command,
        env={},
        completed=completed,
        summary=summary,
        extra_payload={
            "remote": {
                **remote_metadata,
                "job_id": submission.get("job_id"),
                "remote_result_dir": str(remote_result_dir),
                "submit_response": submission,
                "final_status": final_status,
                "poll_transcript": poll_transcript,
                "resolved_git_ref": remote_git_ref,
                "resolved_git_commit": remote_git_commit,
            }
        },
    )

    if returncode != 0:
        stderr_tail = stderr_text.strip()[-4000:]
        stdout_tail = stdout_text.strip()[-2000:]
        raise RuntimeError(
            "Remote Sol simulation failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Command: {_shell_join(remote_benchmark_command)}\n"
            f"Stdout tail:\n{stdout_tail}\n\n"
            f"Stderr tail:\n{stderr_tail}"
        )

    if summary is None:
        raise FileNotFoundError(f"Expected synced benchmark summary at {summary_path}")

    return RunRecord(
        label=label,
        timestamp=timestamp,
        result_dir=local_result_dir,
        summary=summary,
        config=config,
        overrides=build_param_overrides(config),
        command=remote_benchmark_command,
        stdout=stdout_text,
        stderr=stderr_text,
    )


def _json_ready(value: Any) -> Any:
    """Convert arrays, scalars, and paths into JSON-serializable equivalents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_notebook_run_info(
    result_dir,
    *,
    config,
    label,
    timestamp,
    command,
    env,
    completed,
    summary=None,
    extra_payload: dict[str, Any] | None = None,
):
    """Persist normalized config, effective params, and subprocess metadata for a run."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    run_info_path = result_dir / "run_info.json"
    existing = {}
    if run_info_path.exists() and run_info_path.stat().st_size > 0:
        with open(run_info_path) as f:
            existing = json.load(f)

    payload = dict(existing)
    payload.update(
        {
            "label": label,
            "requested_label": label,
            "timestamp": timestamp,
            "runner": "obgpu_experiment_helpers.run_simulation",
            "config": _json_ready(config),
            "overrides": _json_ready(build_param_overrides(config)),
            "command": list(command),
            "returncode": int(completed.returncode),
            "env": {
                "OB_RUN_TIMESTAMP": env.get("OB_RUN_TIMESTAMP"),
                "OB_RESULT_LABEL": env.get("OB_RESULT_LABEL"),
                "OB_CORENRN_CELL_PERMUTE": env.get("OB_CORENRN_CELL_PERMUTE"),
                "OB_RESULTS_BASE": env.get("OB_RESULTS_BASE"),
            },
        }
    )

    try:
        payload["effective_params"] = _json_ready(resolve_effective_params(config))
    except Exception as exc:
        payload["effective_params_error"] = f"{type(exc).__name__}: {exc}"

    if summary is not None:
        payload["summary"] = _json_ready(summary)

    if extra_payload:
        payload.update(_json_ready(extra_payload))

    run_info_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return run_info_path


def run_simulation(config: dict[str, Any] | None = None) -> RunRecord:
    """Run one timestamped notebook simulation and return its recorded metadata."""
    config = build_run_config(**(config or {}))
    timestamp = make_timestamp()
    label = make_label(config, timestamp=timestamp)
    result_dir = Path(config.get("results_base", DEFAULT_RESULTS_BASE)) / label
    runner_backend = str(config.get("runner_backend", "local"))

    if runner_backend == "sol_slurm":
        return _run_remote_simulation(
            config,
            label=label,
            timestamp=timestamp,
            local_result_dir=result_dir,
        )

    if runner_backend != "local":
        raise ValueError(f"Unsupported runner_backend={runner_backend!r}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["OB_RUN_TIMESTAMP"] = timestamp
    env["OB_RESULT_LABEL"] = label
    env["OB_RESULTS_BASE"] = str(config.get("results_base", DEFAULT_RESULTS_BASE))
    env["OB_CORENRN_CELL_PERMUTE"] = str(int(config.get("cell_permute", 2)))

    command = build_run_command(config, label)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "command.txt").write_text(" ".join(command) + "\n")
    (result_dir / "stdout.txt").write_text(completed.stdout or "")
    (result_dir / "stderr.txt").write_text(completed.stderr or "")

    if completed.returncode != 0:
        _write_notebook_run_info(
            result_dir,
            config=config,
            label=label,
            timestamp=timestamp,
            command=command,
            env=env,
            completed=completed,
        )
        stderr_tail = (completed.stderr or "").strip()[-4000:]
        stdout_tail = (completed.stdout or "").strip()[-2000:]
        raise RuntimeError(
            "Simulation failed.\n"
            f"Result dir: {result_dir}\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout tail:\n{stdout_tail}\n\n"
            f"Stderr tail:\n{stderr_tail}"
        )

    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected benchmark summary at {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    _write_notebook_run_info(
        result_dir,
        config=config,
        label=label,
        timestamp=timestamp,
        command=command,
        env=env,
        completed=completed,
        summary=summary,
        extra_payload={"remote": None},
    )

    return RunRecord(
        label=label,
        timestamp=timestamp,
        result_dir=result_dir,
        summary=summary,
        config=config,
        overrides=build_param_overrides(config),
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def list_notebook_runs(
    prefix: str | None = None,
    results_base: str | Path = DEFAULT_RESULTS_BASE,
) -> list[Path]:
    """List saved notebook-run directories, optionally filtered by label prefix."""
    results_base = Path(results_base)
    if not results_base.exists():
        return []
    runs = [path for path in results_base.iterdir() if path.is_dir()]
    if prefix:
        runs = [path for path in runs if path.name.startswith(prefix)]
    return sorted(runs)


def _read_json_if_present(path: str | Path) -> dict[str, Any] | None:
    """Return parsed JSON when a file exists and is non-empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path) as f:
        return json.load(f)


def resolve_notebook_run(
    run_or_dir: str | os.PathLike[str] | RunRecord | None = None,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path = DEFAULT_RESULTS_BASE,
) -> Path:
    """Resolve a run identifier, path, or prefix/index pair into a run directory."""
    if run_or_dir is not None:
        return Path(run_or_dir.result_dir if isinstance(run_or_dir, RunRecord) else run_or_dir)

    runs = list_notebook_runs(prefix=prefix, results_base=results_base)
    if not runs:
        raise FileNotFoundError(f"No notebook runs found in {results_base} with prefix={prefix!r}")
    return runs[index]


def load_run_record(
    run_or_dir: str | os.PathLike[str] | RunRecord | None = None,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path = DEFAULT_RESULTS_BASE,
) -> RunRecord:
    """Load notebook-run metadata from a timestamped results directory."""
    result_dir = resolve_notebook_run(
        run_or_dir=run_or_dir,
        prefix=prefix,
        index=index,
        results_base=results_base,
    )
    summary = _read_json_if_present(result_dir / "summary.json") or {}
    run_info = _read_json_if_present(result_dir / "run_info.json") or {}

    stdout = ""
    stdout_path = result_dir / "stdout.txt"
    if stdout_path.exists():
        stdout = stdout_path.read_text()

    stderr = ""
    stderr_path = result_dir / "stderr.txt"
    if stderr_path.exists():
        stderr = stderr_path.read_text()

    label = (
        run_info.get("label")
        or summary.get("label")
        or run_info.get("requested_label")
        or summary.get("requested_label")
        or result_dir.name
    )
    timestamp = run_info.get("timestamp") or summary.get("timestamp") or ""

    return RunRecord(
        label=label,
        timestamp=timestamp,
        result_dir=result_dir,
        summary=summary,
        config=run_info.get("config", {}),
        overrides=run_info.get("overrides", {}),
        command=run_info.get("command", []),
        stdout=stdout,
        stderr=stderr,
    )


def _path_parts(path: Any) -> list[str]:
    """Split a dotted or indexed override path into addressable components."""
    if isinstance(path, (list, tuple)):
        return list(path)
    text = str(path).replace("[", ".").replace("]", "")
    return [part for part in text.split(".") if part]


def set_path_value(obj: Any, path: Any, value: Any) -> None:
    """Assign ``value`` inside a nested dict/list structure addressed by ``path``."""
    parts = _path_parts(path)
    current = obj
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(current, list):
            part = int(part)
            while len(current) <= part:
                current.append({} if not str(next_part).isdigit() else [])
            current = current[part]
            continue
        if part not in current or current[part] is None:
            current[part] = [] if str(next_part).isdigit() else {}
        current = current[part]
    final = parts[-1]
    if isinstance(current, list):
        final = int(final)
        while len(current) <= final:
            current.append(None)
        current[final] = value
    else:
        current[final] = value


def run_parameter_sweep(
    base_config: dict[str, Any],
    sweep_path: str | list[str],
    values: list[Any] | tuple[Any, ...],
) -> dict[str, Any]:
    """Run a one-parameter sweep by repeatedly calling :func:`run_and_load`."""
    base_config = build_run_config(**deepcopy(base_config))
    items = []
    for value in values:
        sweep_config = deepcopy(base_config)
        set_path_value(sweep_config, sweep_path, value)
        run, result = run_and_load(sweep_config)
        items.append(
            {
                "value": value,
                "config": sweep_config,
                "run": run,
                "result": result,
            }
        )
    return {"path": sweep_path, "values": list(values), "items": items}


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_result(run_or_dir: RunRecord | str | Path) -> dict[str, Any]:
    """Load the standard saved outputs for a notebook run directory."""
    result_dir = Path(run_or_dir.result_dir if isinstance(run_or_dir, RunRecord) else run_or_dir)
    summary = _read_json_if_present(result_dir / "summary.json")
    run_info = _read_json_if_present(result_dir / "run_info.json")

    result = {
        "result_dir": result_dir,
        "summary": summary,
        "run_info": run_info,
        "input_times": [],
        "soma_vs": [],
        "gc_output_events": [],
        "lfp_t": np.array([]),
        "lfp": np.array([]),
    }

    input_path = result_dir / "input_times.pkl"
    if input_path.exists():
        result["input_times"] = load_pickle(input_path)

    soma_path = result_dir / "soma_vs.pkl"
    if soma_path.exists():
        result["soma_vs"] = load_pickle(soma_path)

    gc_output_path = result_dir / "gc_output_events.pkl"
    if gc_output_path.exists():
        result["gc_output_events"] = load_pickle(gc_output_path)

    lfp_path = result_dir / "lfp.pkl"
    if lfp_path.exists():
        lfp_t, lfp = load_pickle(lfp_path)
        result["lfp_t"] = np.asarray(lfp_t, dtype=float)
        result["lfp"] = np.asarray(lfp, dtype=float)

    return result


def load_run_pair(
    run_or_dir: RunRecord | str | Path | None = None,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path = DEFAULT_RESULTS_BASE,
) -> tuple[RunRecord, dict[str, Any]]:
    """Resolve a saved run and load its standard result payload."""
    run = load_run_record(
        run_or_dir=run_or_dir,
        prefix=prefix,
        index=index,
        results_base=results_base,
    )
    return run, load_result(run)


def run_and_load(config: dict[str, Any] | None = None) -> tuple[RunRecord, dict[str, Any]]:
    """Run a simulation and immediately load its outputs from disk."""
    run = run_simulation(config)
    return run, load_result(run)


def normalize_cell_name(name: Any) -> str:
    """Strip HOC prefixes and section suffixes down to a canonical cell label."""
    return str(name).removeprefix("h.").split(".", 1)[0]


def cell_type_of(name: Any) -> str:
    """Infer the cell family prefix such as ``MC`` or ``GC`` from a label."""
    match = re.match(r"([A-Z]+)", normalize_cell_name(name))
    if not match:
        raise ValueError(f"Could not infer cell type from {name!r}")
    return match.group(1)


def get_slice_dir(slice_name: str = "DorsalColumnSlice") -> Path:
    """Return the on-disk directory for a named slice export."""
    return REPO_ROOT / "olfactorybulb" / "slices" / str(slice_name)


def load_slice_connectivity(slice_name: str = "DorsalColumnSlice") -> dict[str, Any]:
    """Load the static glomerular and reciprocal connectivity JSON for a slice."""
    slice_dir = get_slice_dir(slice_name)
    with open(slice_dir / "glom_cells.json") as f:
        glom_cells = json.load(f)

    synapse_sets = {}
    for synapse_set_name in ("GCs__MCs", "GCs__TCs"):
        path = slice_dir / f"{synapse_set_name}.json"
        if path.exists():
            with open(path) as f:
                synapse_sets[synapse_set_name] = json.load(f)["entries"]

    return {
        "slice_name": slice_name,
        "slice_dir": slice_dir,
        "glom_cells": glom_cells,
        "synapse_sets": synapse_sets,
    }


def find_cell_drivers(cell_name: str, slice_name: str = "DorsalColumnSlice") -> dict[str, Any]:
    """Summarize glomerular peers and reciprocal GC inputs for one cell."""
    target = normalize_cell_name(cell_name)
    target_type = cell_type_of(target)
    connectivity = load_slice_connectivity(slice_name=slice_name)
    glom_cells = connectivity["glom_cells"]

    glomeruli = sorted(glom for glom, cells in glom_cells.items() if target in cells)
    glomerulus_members = {glom: list(glom_cells[glom]) for glom in glomeruli}
    glomerulus_peers = {
        glom: [cell for cell in glom_cells[glom] if cell != target]
        for glom in glomeruli
    }
    gap_junction_peers = {
        glom: [
            cell
            for cell in glom_cells[glom]
            if cell != target and cell_type_of(cell) == target_type
        ]
        for glom in glomeruli
        if target_type in {"MC", "TC"}
    }

    reciprocal_set = None
    if target_type == "MC":
        reciprocal_set = "GCs__MCs"
    elif target_type == "TC":
        reciprocal_set = "GCs__TCs"

    reciprocal_inputs = []
    source_counts = Counter()
    dest_section_counts = Counter()
    if reciprocal_set is not None:
        entries = connectivity["synapse_sets"].get(reciprocal_set, [])
        reciprocal_inputs = [
            row for row in entries if normalize_cell_name(row["dest_section"]) == target
        ]
        source_counts = Counter(normalize_cell_name(row["source_section"]) for row in reciprocal_inputs)
        dest_section_counts = Counter(row["dest_section"].split(".", 1)[1] for row in reciprocal_inputs)

    return {
        "target_cell": target,
        "target_type": target_type,
        "slice_name": slice_name,
        "glomeruli": glomeruli,
        "glomerulus_members": glomerulus_members,
        "glomerulus_peers": glomerulus_peers,
        "gap_junction_peers": gap_junction_peers,
        "reciprocal_synapse_set": reciprocal_set,
        "reciprocal_inputs": reciprocal_inputs,
        "reciprocal_source_counts": dict(source_counts),
        "reciprocal_dest_section_counts": dict(dest_section_counts),
    }


def print_cell_drivers(
    cell_name: str,
    slice_name: str = "DorsalColumnSlice",
    max_sources: int = 10,
) -> None:
    """Print a compact textual summary of the drivers returned by ``find_cell_drivers``."""
    info = find_cell_drivers(cell_name, slice_name=slice_name)
    print(f"Target: {info['target_cell']} ({info['target_type']})")
    print(f"Slice: {info['slice_name']}")
    print(f"Glomeruli: {info['glomeruli']}")

    for glom in info["glomeruli"]:
        print(f"\nGlomerulus {glom} members:")
        print(info["glomerulus_members"][glom])
        if glom in info["gap_junction_peers"]:
            print(f"Gap-junction peers in glomerulus {glom}: {info['gap_junction_peers'][glom]}")

    if info["reciprocal_synapse_set"] is not None:
        print(f"\nIncoming reciprocal contacts via {info['reciprocal_synapse_set']}: {len(info['reciprocal_inputs'])}")
        top_sources = sorted(
            info["reciprocal_source_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_sources]
        print("Top reciprocal source cells:")
        print(top_sources)

        top_sections = sorted(
            info["reciprocal_dest_section_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_sources]
        print("Most targeted destination sections:")
        print(top_sections)


def _apply_param_override_object(params: Any, overrides: dict[str, Any]) -> None:
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
    _apply_param_override_object(params, build_param_overrides(config))
    if config.get("extra_overrides"):
        _apply_param_override_object(params, config["extra_overrides"])

    input_odors_source = "override" if config.get("input_odors") is not None else "paramset"
    input_odors = deepcopy(getattr(params, "input_odors", {}))
    odor_names = sorted({entry.get("name") for entry in input_odors.values() if isinstance(entry, dict) and entry.get("name")})

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


def flatten_for_diff(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into ``path -> value`` pairs for diff reporting."""
    items = {}
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: str(item)):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(flatten_for_diff(value[key], next_prefix))
        return items
    items[prefix or "$"] = value
    return items


def diff_values(before: Any, after: Any) -> list[dict[str, Any]]:
    """Return value changes between two nested JSON-like structures."""
    before_flat = flatten_for_diff(before)
    after_flat = flatten_for_diff(after)
    keys = sorted(set(before_flat) | set(after_flat))
    changes = []
    for key in keys:
        before_value = before_flat.get(key)
        after_value = after_flat.get(key)
        if before_value != after_value:
            changes.append(
                {
                    "path": key,
                    "before": before_value,
                    "after": after_value,
                }
            )
    return changes


def _format_diff_value(value: Any, max_len: int = 160) -> str:
    """Render a compact JSON string for a diff value."""
    text = json.dumps(_json_ready(value), sort_keys=True)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def print_diff_section(title: str, changes: list[dict[str, Any]], max_items: int | None = None) -> None:
    """Print a human-readable diff section for notebook summaries."""
    print(f"\n{title}:")
    if not changes:
        print("  (no differences)")
        return

    if max_items is None:
        max_items = len(changes)

    for change in changes[:max_items]:
        print(
            f"- {change['path']}: "
            f"{_format_diff_value(change['before'])} -> {_format_diff_value(change['after'])}"
        )

    remaining = len(changes) - max_items
    if remaining > 0:
        print(f"- ... {remaining} more differences")


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
        "analysis_dt_ms",
        "spectrogram_signal",
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
        "remote_git_ref",
        "remote_git_fetch",
        "remote_poll_interval_s",
        "remote_mpi_exec",
        "slurm_partition",
        "slurm_account",
        "slurm_time",
        "slurm_gpus",
        "slurm_cpus_per_task",
        "slurm_mem",
        "slurm_extra_args",
        "ssh_binary",
        "ssh_options",
        "ssh_multiplex",
        "ssh_allow_interactive_auth",
        "ssh_control_path",
        "ssh_control_persist_s",
        "rsync_binary",
        "rsync_options",
    ]
    return {key: _json_ready(config.get(key)) for key in runtime_keys if key in config}


def build_live_inspection_model(
    paramset: str = "GammaSignature",
    *,
    extra_overrides: dict[str, Any] | None = None,
    enable_lfp: bool = False,
    record_from_somas: tuple[str, ...] | list[str] = (),
    use_corenrn: bool = False,
    use_gpu: bool = False,
    runtime_mode: str = "scientific",
    reuse_existing: bool = True,
) -> Any:
    """Build one live model inside the kernel for morphology/connectivity inspection.

    The notebook runner normally keeps NEURON in a subprocess to avoid kernel
    corruption. This function intentionally breaks that rule for read-only
    inspection workflows and therefore only permits one model build per kernel.
    """
    global _LIVE_INSPECTION_MODEL, _LIVE_INSPECTION_SIGNATURE

    extra_overrides = deepcopy(extra_overrides or {})
    requested_signature = json.dumps(
        {
            "paramset": paramset,
            "extra_overrides": extra_overrides,
            "enable_lfp": bool(enable_lfp),
            "record_from_somas": list(record_from_somas),
            "use_corenrn": bool(use_corenrn),
            "use_gpu": bool(use_gpu),
            "runtime_mode": runtime_mode,
        },
        sort_keys=True,
    )

    if _LIVE_INSPECTION_MODEL is not None:
        if reuse_existing and requested_signature == _LIVE_INSPECTION_SIGNATURE:
            return _LIVE_INSPECTION_MODEL
        raise RuntimeError(
            "A live inspection model is already loaded in this kernel. "
            "Restart the kernel before building a different one."
        )

    import olfactorybulb.model as obmodel
    from olfactorybulb.model import OlfactoryBulb

    params = getattr(obmodel, paramset)()
    params.runtime_mode = runtime_mode
    params.enable_status_report = False
    params.enable_lfp = bool(enable_lfp)
    params.record_from_somas = list(record_from_somas)
    params.coreneuron = SimpleNamespace(
        enable=bool(use_corenrn),
        gpu=bool(use_gpu),
        file_mode=False,
        verbose=0,
        cell_permute=2 if use_gpu else 0,
        warp_balance=128 if use_gpu else 0,
    )
    if extra_overrides:
        _apply_param_override_object(params, extra_overrides)

    model = OlfactoryBulb(params, autorun=False)
    _LIVE_INSPECTION_MODEL = model
    _LIVE_INSPECTION_SIGNATURE = requested_signature
    return model


def get_live_cell(model: Any, cell_name: str) -> Any:
    """Return a live cell object from a live inspection model."""
    target = normalize_cell_name(cell_name)
    target_type = cell_type_of(target)
    for cell in model.cells.get(target_type, []):
        if normalize_cell_name(str(cell.soma)) == target:
            return cell
    raise KeyError(f"Cell {target!r} not found in live model")


def get_live_section(model: Any, section_name: str) -> Any:
    """Resolve a section string like ``TC5[12].dend[3]`` in a live model."""
    section_name = str(section_name).removeprefix("h.")
    if "(" not in section_name:
        seg_expr = f"h.{section_name}(0.5)"
    else:
        seg_expr = section_name if section_name.startswith("h.") else f"h.{section_name}"
    return model.resolve_segment(seg_expr).sec


def get_section_parent_chain(model: Any, section_name: str) -> list[str]:
    """Return the parent-section chain from a section back to the root."""
    sec = get_live_section(model, section_name)
    chain = []
    while sec is not None:
        chain.append(str(sec))
        parent_seg = sec.parentseg()
        sec = None if parent_seg is None else parent_seg.sec
    return chain


def get_cell_section_parent_map(model: Any, cell_name: str) -> dict[str, str | None]:
    """Map every section of one cell to its parent section."""
    cell = get_live_cell(model, cell_name)
    parent_map = {}
    for sec in cell.soma.wholetree():
        parent_seg = sec.parentseg()
        parent_map[str(sec)] = None if parent_seg is None else str(parent_seg.sec)
    return parent_map


def result_overview(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize the key dimensions and timing fields of a loaded result."""
    summary = result.get("summary") or {}
    params = summary.get("params", {})
    timings = summary.get("timing_seconds", {})
    return {
        "result_dir": str(result["result_dir"]),
        "label": summary.get("label"),
        "paramset": summary.get("paramset"),
        "nranks": summary.get("nranks"),
        "tstop_ms": params.get("tstop"),
        "sim_dt_ms": params.get("sim_dt"),
        "actual_dt_ms": params.get("actual_dt"),
        "recording_period_ms": params.get("recording_period"),
        "run_seconds": timings.get("run_max_rank"),
        "total_seconds": timings.get("total_max_rank"),
        "n_inputs": len(result.get("input_times", [])),
        "n_soma_traces": len(result.get("soma_vs", [])),
        "n_gc_output_connections": len(result.get("gc_output_events", [])),
        "n_lfp_samples": int(len(result.get("lfp", []))),
    }


def uniform_trace(
    t: np.ndarray | list[float],
    y: np.ndarray | list[float],
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a trace onto a uniform time grid suitable for spectral analysis."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return t, y
    if dt_ms is None:
        dt_ms = float(np.median(np.diff(t)))
    grid = np.arange(float(t[0]), float(t[-1]) + 0.5 * dt_ms, dt_ms)
    interp = interp1d(t, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    return grid, interp(grid)


def butter_bandpass_filter(
    signal: np.ndarray | list[float],
    lowcut_hz: float,
    highcut_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter, falling back to causal filtering if needed."""
    signal = np.asarray(signal, dtype=float)
    nyquist = 0.5 * fs_hz
    b, a = butter(order, [lowcut_hz / nyquist, highcut_hz / nyquist], btype="band")
    min_len = 3 * max(len(a), len(b))
    if len(signal) <= min_len:
        return lfilter(b, a, signal)
    return filtfilt(b, a, signal)


def compute_lfp_bandpassed(
    result: dict[str, Any],
    dt_ms: float | None = None,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 120.0,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the saved LFP resampled and band-pass filtered."""
    t, lfp = uniform_trace(result["lfp_t"], result["lfp"], dt_ms=dt_ms)
    fs_hz = 1000.0 / float(np.median(np.diff(t)))
    return t, butter_bandpass_filter(lfp, lowcut_hz, highcut_hz, fs_hz, order=order)


def compute_spectrogram(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    dt_ms: float | None = None,
    max_freq_hz: float = 150.0,
    nperseg: int = 512,
    noverlap: int = 448,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a standard spectrogram on a uniform time base."""
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    if len(t) < 4:
        raise ValueError("Trace is too short for spectral analysis")
    fs_hz = 1000.0 / float(np.median(np.diff(t)))
    nperseg = min(nperseg, len(y))
    noverlap = min(noverlap, max(0, nperseg - 1))
    freqs, times_s, power = spectrogram(
        y,
        fs=fs_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    mask = freqs <= max_freq_hz
    return times_s * 1000.0, freqs[mask], power[mask]


def compute_wavelet_map(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    dt_ms: float = 0.1,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 120.0,
    wavelet: str = "cgau5",
    scale_low: float = 3.0,
    scale_high: float = 32.0,
    n_scales: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the legacy-style continuous wavelet map used in the notebooks."""
    if pywt is None:
        raise ModuleNotFoundError(
            "PyWavelets is required for wavelet analysis. Install the 'pywavelets' package."
        )
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    fs_hz = 1000.0 / dt_ms
    y_bp = butter_bandpass_filter(y, lowcut_hz, highcut_hz, fs_hz, order=4)
    scales = np.linspace(scale_low / dt_ms, scale_high / dt_ms, n_scales)
    cfs, freqs = pywt.cwt(y_bp, scales, wavelet, dt_ms / 1000.0)
    power = np.log1p(np.abs(cfs))
    return t, y_bp, freqs, power


def compute_wavelet_band_power(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Collapse wavelet power into named frequency-band time series."""
    if bands is None:
        bands = {
            "beta": (15.0, 35.0),
            "low_gamma": (35.0, 65.0),
            "high_gamma": (65.0, 100.0),
        }
    t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
    traces = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if np.any(mask):
            traces[name] = power[mask].mean(axis=0)
        else:
            traces[name] = np.zeros(power.shape[1])
    return t, freqs, power, traces


def load_legacy_wavelet_analysis(
    result: dict[str, Any],
    dt: float = 0.1,
    sniff_count: int = 8,
) -> dict[str, Any]:
    """Reproduce the legacy LFP wavelet-analysis pipeline for one result."""
    if pywt is None:
        raise ModuleNotFoundError(
            "PyWavelets is required for wavelet analysis. Install the 'pywavelets' package."
        )
    input_times = sorted(result["input_times"], key=lambda row: row[0])
    events = {}
    for seg_name, seg_times in input_times:
        events[seg_name] = events.get(seg_name, []) + list(seg_times)

    vs = list(result["soma_vs"])
    vs.sort(key=lambda row: row[0][0:2])

    t, lfp = uniform_trace(result["lfp_t"], result["lfp"], dt_ms=dt)
    lfp_bp = butter_bandpass_filter(lfp, 30, 120, 1 / dt * 1000, order=4)

    scales = np.linspace(3 / dt, 32 / dt, 50)
    cfs, frequencies = pywt.cwt(lfp_bp, scales, "cgau5", dt / 1000.0)
    lfp_wavelet_power = np.log(1 + np.abs(cfs))

    sniff_duration = 200
    skip_first_n_sniffs = 1
    step = int(round(sniff_duration / dt))
    chunks = []
    for i in range(sniff_count + skip_first_n_sniffs)[skip_first_n_sniffs:]:
        start = i * step
        stop = (i + 1) * step - 2
        if stop <= lfp_wavelet_power.shape[1]:
            chunks.append(lfp_wavelet_power[:, start:stop])
    if chunks:
        lfp_wavelet_power_average = sum(chunks)
        t_average = t[0:chunks[0].shape[1]]
    else:
        lfp_wavelet_power_average = lfp_wavelet_power[:, : max(1, step - 2)]
        t_average = t[0:lfp_wavelet_power_average.shape[1]]

    return {
        "events": events,
        "vs": vs,
        "t": t,
        "lfp": lfp,
        "lfp_bp": lfp_bp,
        "lfp_wavelet_power": lfp_wavelet_power,
        "frequencies": frequencies,
        "t_average": t_average,
        "lfp_wavelet_power_average": lfp_wavelet_power_average,
    }


def plot_legacy_sniff_average(
    t_average: np.ndarray,
    frequencies: np.ndarray,
    lfp_wavelet_power_average: np.ndarray,
    show: bool = True,
    yaxis: bool = True,
    xlabel: bool = True,
) -> None:
    """Plot the sniff-averaged legacy wavelet view used in older notebooks."""
    if show:
        plt.subplots(figsize=(4, 5))

    plt.contourf(t_average, frequencies, lfp_wavelet_power_average, 256, cmap="jet")
    plt.ylim((20, 140))

    if yaxis:
        plt.ylabel("Frequency [Hz]")
    else:
        plt.gca().axes.get_yaxis().set_visible(False)

    if xlabel:
        plt.xlabel("Time Since Sniff Onset [ms]")

    plt.xticks(np.arange(round(min(t_average)), max(t_average) + 1, 50.0)[:-1])

    if show:
        plt.show()


def show_legacy_plots(
    result: dict[str, Any],
    sniff_count: int = 8,
    dt: float = 0.1,
    fig_width: float = 27,
) -> dict[str, Any]:
    """Render the legacy voltage, LFP, and wavelet figure set for one run."""
    legacy = load_legacy_wavelet_analysis(result, dt=dt, sniff_count=sniff_count)

    i = 0
    plt.subplots(figsize=(fig_width, len(legacy["vs"]) * 0.1))
    for cell, t, v in legacy["vs"]:
        if "MC" in cell:
            col = "blue"
        if "TC" in cell:
            col = "red"
        if "GC" in cell:
            col = "orange"

        plt.plot(t, np.array(v) + i, col, label=cell)
        i += 100

    events = [(seg, times) for seg, times in legacy["events"].items()]
    events.sort(key=lambda row: row[0])

    for seg, times in events:
        if "MC" in seg:
            col = "b"
        if "TC" in seg:
            col = "r"
        plt.plot(times, [i] * len(times), col + "|", ms=5, label=seg)
        i += 10

    plt.xticks(np.arange(min(legacy["t"]), max(legacy["t"]) + 1, 50.0))
    plt.margins(0)
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.xlabel("Simulation Time [ms]")
    plt.show()

    plt.subplots(figsize=(fig_width, 5))
    plt.margins(0)
    plt.plot(legacy["t"], legacy["lfp"] * 1000)
    plt.plot(legacy["t"], legacy["lfp_bp"] * 10000 - 200)
    plt.xticks(np.arange(min(legacy["t"]), max(legacy["t"]) + 1, 50.0))
    plt.yticks([])
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.xlabel("Simulation Time [ms]")
    plt.show()

    plt.subplots(figsize=(fig_width, 5))
    plt.contourf(legacy["t"], legacy["frequencies"], legacy["lfp_wavelet_power"], 256, cmap="jet")
    plt.ylim((20, 140))
    plt.xticks(np.arange(round(min(legacy["t"])), max(legacy["t"]) + 1, 50.0))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Simulation Time [ms]")
    plt.show()

    plot_legacy_sniff_average(
        legacy["t_average"],
        legacy["frequencies"],
        legacy["lfp_wavelet_power_average"],
    )
    return legacy


def detect_spikes(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float = 0.0,
) -> np.ndarray:
    """Detect upward threshold crossings from a soma voltage trace."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 2:
        return np.array([])
    crossings = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
    return t[crossings]


def calculate_instantaneous_frequency(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spike times from one trace into instantaneous frequency samples."""
    spikes = detect_spikes(t, v, threshold=threshold)
    if len(spikes) < 2:
        return np.array([]), np.array([])
    t_freq = (spikes[:-1] + spikes[1:]) / 2.0
    spiking_hz = 1000.0 / np.diff(spikes)
    return t_freq, spiking_hz


def calculate_event_frequency(times: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert event times into midpoint/frequency samples."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.array([]), np.array([])
    t_freq = (times[:-1] + times[1:]) / 2.0
    event_hz = 1000.0 / np.diff(times)
    return t_freq, event_hz


def plot_spiking_frequencies(
    result: dict[str, Any],
    indices: list[int] | range | None = None,
    ax: Any = None,
    threshold: float = 0.0,
) -> Any:
    """Plot instantaneous firing-rate traces for selected saved soma voltages."""
    ax = ax or plt.subplots(figsize=(10, 6))[1]
    soma_vs = result["soma_vs"]
    if indices is None:
        indices = range(len(soma_vs))

    for i in indices:
        label, t, v = soma_vs[i]
        t_freq, spiking_hz = calculate_instantaneous_frequency(t, v, threshold=threshold)
        if len(t_freq) > 0:
            ax.plot(t_freq, spiking_hz, label=label)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Instantaneous Spiking Frequencies")
    if ax.lines:
        ax.legend(loc="upper right", fontsize=8)
    return ax


def split_traces_by_type(result: dict[str, Any]) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Group saved soma traces by cell family prefix."""
    grouped = {"MC": [], "TC": [], "GC": [], "other": []}
    for label, t, v in result["soma_vs"]:
        bucket = "other"
        for candidate in ("MC", "TC", "GC"):
            if label.startswith(candidate):
                bucket = candidate
                break
        grouped[bucket].append((label, np.asarray(t, dtype=float), np.asarray(v, dtype=float)))
    return grouped


def filter_gc_output_events(
    result: dict[str, Any],
    target_types: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Filter saved GC inhibitory-output events by destination cell family."""
    events = list(result.get("gc_output_events", []))
    if not target_types:
        return events

    target_types = {str(name) for name in target_types}
    filtered = []
    for entry in events:
        dest_cell = normalize_cell_name(entry.get("dest_section", ""))
        if any(dest_cell.startswith(cell_type) for cell_type in target_types):
            filtered.append(entry)
    return filtered


def collect_gc_output_frequency_samples(
    result: dict[str, Any],
    indices: list[int] | range | None = None,
    target_types: list[str] | tuple[str, ...] | None = None,
    modulus: float | None = None,
) -> dict[str, Any]:
    """Collect instantaneous GC inhibitory-output frequency samples for KDE plots."""
    events = filter_gc_output_events(result, target_types=target_types)
    if indices is None:
        indices = range(len(events))

    selected_events = []
    all_freq_t = []
    all_freq = []

    for i in indices:
        if i >= len(events):
            break
        entry = events[i]
        t_freq, event_hz = calculate_event_frequency(entry.get("times", []))
        if len(t_freq) == 0:
            continue
        if modulus is not None:
            t_freq = np.mod(t_freq, float(modulus))
        all_freq_t.append(np.asarray(t_freq, dtype=float))
        all_freq.append(np.asarray(event_hz, dtype=float))
        selected_events.append(entry)

    if all_freq_t:
        times = np.concatenate(all_freq_t)
        freqs = np.concatenate(all_freq)
    else:
        times = np.array([], dtype=float)
        freqs = np.array([], dtype=float)

    return {
        "times": times,
        "freqs": freqs,
        "events": selected_events,
        "n_events": len(selected_events),
    }


def _resolve_event_tstop(result: dict[str, Any], event_series: list[np.ndarray]) -> float:
    """Infer the latest relevant time from LFP, soma traces, or event series."""
    if len(result.get("lfp_t", [])) > 0:
        return float(result["lfp_t"][-1])

    t_stop = 0.0
    for _label, t, _v in result.get("soma_vs", []):
        if len(t) > 0:
            t_stop = max(t_stop, float(t[-1]))
    for times in event_series:
        if len(times) > 0:
            t_stop = max(t_stop, float(times[-1]))
    return t_stop


def _smooth_rate(rate_hz: np.ndarray, *, bin_ms: float, smooth_sigma_ms: float) -> np.ndarray:
    """Gaussian-smooth a binned rate trace."""
    if smooth_sigma_ms and smooth_sigma_ms > 0:
        sigma_bins = float(smooth_sigma_ms) / float(bin_ms)
        radius = max(1, int(round(4.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        rate_hz = np.convolve(rate_hz, kernel, mode="same")
    return rate_hz


def _event_rate_from_series(
    event_series: list[np.ndarray],
    *,
    t_stop: float,
    bin_ms: float,
    smooth_sigma_ms: float,
    denominator: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin one or more event series into a smoothed population-rate trace."""
    if t_stop <= 0.0:
        return np.array([]), np.array([])

    edges = np.arange(0.0, t_stop + float(bin_ms), float(bin_ms))
    if edges.size < 2:
        edges = np.array([0.0, float(bin_ms)], dtype=float)

    flat_times = []
    for times in event_series:
        times = np.asarray(times, dtype=float)
        if times.size:
            flat_times.append(times)

    if flat_times:
        counts, _edges = np.histogram(np.concatenate(flat_times), bins=edges)
    else:
        counts = np.zeros(len(edges) - 1, dtype=float)

    rate_hz = counts.astype(float) / (float(bin_ms) / 1000.0)
    denom = max(float(denominator), 1.0)
    rate_hz /= denom
    rate_hz = _smooth_rate(rate_hz, bin_ms=bin_ms, smooth_sigma_ms=smooth_sigma_ms)
    centers = edges[:-1] + float(bin_ms) / 2.0
    return centers, rate_hz


def _gc_rate_normalizer(events: list[dict[str, Any]], normalization: str) -> tuple[float, str]:
    """Return the denominator and ylabel for GC-output rate normalization."""
    normalization = str(normalization or "per_target_cell")
    if normalization == "total":
        return 1.0, "events/s"
    if normalization == "per_connection":
        return float(len(events)), "events/s per connection"
    if normalization == "per_source_cell":
        source_cells = {normalize_cell_name(entry.get("source_section", "")) for entry in events}
        return float(len(source_cells)), "events/s per source GC"
    if normalization == "per_target_cell":
        target_cells = {normalize_cell_name(entry.get("dest_section", "")) for entry in events}
        return float(len(target_cells)), "events/s per target cell"
    raise ValueError(f"Unsupported GC normalization mode {normalization!r}")


def compute_gc_output_rate(
    result: dict[str, Any],
    bin_ms: float = 5.0,
    smooth_sigma_ms: float = 10.0,
    target_types: list[str] | tuple[str, ...] | None = None,
    normalization: str = "per_target_cell",
    return_metadata: bool = False,
) -> Any:
    """Compute a GC inhibitory-output rate trace with configurable normalization."""
    events = filter_gc_output_events(result, target_types=target_types)
    event_series = [np.asarray(entry.get("times", []), dtype=float) for entry in events]
    t_stop = _resolve_event_tstop(result, event_series)
    denominator, unit = _gc_rate_normalizer(events, normalization)
    centers, rate_hz = _event_rate_from_series(
        event_series,
        t_stop=t_stop,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        denominator=denominator,
    )
    if return_metadata:
        return centers, rate_hz, {
            "normalization": normalization,
            "unit": unit,
            "denominator": max(float(denominator), 1.0),
            "n_connections": len(events),
            "n_source_cells": len({normalize_cell_name(entry.get("source_section", "")) for entry in events}),
            "n_target_cells": len({normalize_cell_name(entry.get("dest_section", "")) for entry in events}),
        }
    return centers, rate_hz


def filter_input_events(
    result: dict[str, Any],
    target_types: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, Any]]:
    """Filter odor-input event rows by destination cell family."""
    rows = list(result.get("input_times", []))
    if not target_types:
        return rows

    target_types = {str(name) for name in target_types}
    filtered = []
    for section_name, times in rows:
        cell_name = normalize_cell_name(section_name)
        if any(cell_name.startswith(cell_type) for cell_type in target_types):
            filtered.append((section_name, times))
    return filtered


def _input_rate_normalizer(rows: list[tuple[str, Any]], normalization: str) -> tuple[float, str]:
    """Return the denominator and ylabel for odor-input rate normalization."""
    normalization = str(normalization or "per_target_cell")
    if normalization == "total":
        return 1.0, "events/s"
    if normalization in {"per_segment", "per_input_segment"}:
        return float(len(rows)), "events/s per input segment"
    if normalization in {"per_cell", "per_target_cell"}:
        target_cells = {normalize_cell_name(section_name) for section_name, _times in rows}
        return float(len(target_cells)), "events/s per target cell"
    raise ValueError(f"Unsupported input normalization mode {normalization!r}")


def compute_input_rate(
    result: dict[str, Any],
    bin_ms: float = 5.0,
    smooth_sigma_ms: float = 10.0,
    target_types: list[str] | tuple[str, ...] | None = None,
    normalization: str = "per_target_cell",
    return_metadata: bool = False,
) -> Any:
    """Compute an odor-input event-rate trace with configurable normalization."""
    rows = filter_input_events(result, target_types=target_types)
    event_series = [np.asarray(times, dtype=float) for _section_name, times in rows]
    t_stop = _resolve_event_tstop(result, event_series)
    denominator, unit = _input_rate_normalizer(rows, normalization)
    centers, rate_hz = _event_rate_from_series(
        event_series,
        t_stop=t_stop,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        denominator=denominator,
    )
    if return_metadata:
        return centers, rate_hz, {
            "normalization": normalization,
            "unit": unit,
            "denominator": max(float(denominator), 1.0),
            "n_segments": len(rows),
            "n_target_cells": len({normalize_cell_name(section_name) for section_name, _times in rows}),
        }
    return centers, rate_hz


def _rate_series_label(base_label: str, metadata: dict[str, Any]) -> str:
    """Append denominator information to a plotted rate-series label."""
    normalization = str(metadata.get("normalization", ""))
    if normalization == "per_target_cell":
        return f"{base_label} (n={metadata.get('n_target_cells', 0)} cells)"
    if normalization == "per_source_cell":
        return f"{base_label} (n={metadata.get('n_source_cells', 0)} GCs)"
    if normalization == "per_connection":
        return f"{base_label} (n={metadata.get('n_connections', 0)} connections)"
    if normalization == "per_cell":
        return f"{base_label} (n={metadata.get('n_target_cells', 0)} cells)"
    if normalization in {"per_segment", "per_input_segment"}:
        return f"{base_label} (n={metadata.get('n_segments', 0)} segments)"
    return base_label


def get_named_signal(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one named analysis signal into a uniform time/value trace."""
    if signal == "lfp":
        return uniform_trace(result["lfp_t"], result["lfp"], dt_ms=dt_ms)

    if signal in {"gc_output_rate", "gc_output_rate_MC", "gc_output_rate_TC"}:
        target_types = None
        if signal.endswith("_MC"):
            target_types = ["MC"]
        elif signal.endswith("_TC"):
            target_types = ["TC"]
        bin_ms = 5.0 if dt_ms is None else float(dt_ms)
        return compute_gc_output_rate(
            result,
            bin_ms=bin_ms,
            smooth_sigma_ms=max(2.0 * bin_ms, 5.0),
            target_types=target_types,
            normalization="per_target_cell",
        )

    if signal in {"input_rate", "input_rate_MC", "input_rate_TC"}:
        target_types = None
        if signal.endswith("_MC"):
            target_types = ["MC"]
        elif signal.endswith("_TC"):
            target_types = ["TC"]
        bin_ms = 5.0 if dt_ms is None else float(dt_ms)
        return compute_input_rate(
            result,
            bin_ms=bin_ms,
            smooth_sigma_ms=max(2.0 * bin_ms, 5.0),
            target_types=target_types,
            normalization="per_target_cell",
        )

    grouped = split_traces_by_type(result)
    if signal in {"mean_MC_voltage", "mean_TC_voltage", "mean_GC_voltage"}:
        cell_type = signal.split("_", 1)[1].split("_", 1)[0]
        traces = grouped.get(cell_type, [])
        if not traces:
            raise KeyError(f"No soma traces found for {cell_type}")
        first_t, _first_v = uniform_trace(traces[0][1], traces[0][2], dt_ms=dt_ms)
        aligned = []
        for _label, t, v in traces:
            interp_t, interp_v = uniform_trace(t, v, dt_ms=float(np.median(np.diff(first_t))) if len(first_t) > 1 else dt_ms)
            n = min(len(first_t), len(interp_t))
            aligned.append(interp_v[:n])
        n = min(len(values) for values in aligned)
        return first_t[:n], np.mean(np.vstack([values[:n] for values in aligned]), axis=0)

    for label, t, v in result["soma_vs"]:
        if label == signal:
            return uniform_trace(t, v, dt_ms=dt_ms)

    raise KeyError(f"Unsupported signal {signal!r}")


def _recommended_raster_fontsize(n_rows: int, *, default: float = 7.0) -> float:
    """Choose a compact but readable y-label font size for dense rasters."""
    if n_rows >= 140:
        return 5.0
    if n_rows >= 80:
        return 6.0
    return float(default)


def _recommended_raster_height(n_rows: int, *, min_height: float = 4.0) -> float:
    """Estimate a reasonable figure height for a raster plot."""
    if n_rows <= 0:
        return float(min_height)
    return max(float(min_height), 0.06 * float(n_rows) + 1.5)


def _ensure_raster_axis(
    ax: Any,
    n_rows: int,
    *,
    width: float = 14.0,
    min_height: float = 4.0,
    per_row_height: float = 0.22,
) -> Any:
    """Create a raster axis sized to the current row count when needed."""
    if ax is None:
        height = max(min_height, per_row_height * max(int(n_rows), 1) + 1.0)
        _fig, ax = plt.subplots(figsize=(width, height))
    return ax


def _style_raster_axis(
    ax: Any,
    labels: list[str],
    *,
    ylabel: str,
    title: str,
    fontsize: float = 7,
    line_spacing: float = 1.4,
) -> np.ndarray:
    """Apply shared styling and row offsets to a raster axis."""
    n_rows = len(labels)
    offsets = np.arange(n_rows, dtype=float) * float(line_spacing)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels, fontsize=fontsize)
    if n_rows:
        pad = max(0.7, line_spacing)
        ax.set_ylim(offsets[0] - pad, offsets[-1] + pad)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return offsets


def _fit_raster_labels(
    ax: Any,
    offsets: np.ndarray,
    *,
    min_fontsize: float = 4.5,
    target_ratio: float = 0.9,
    min_height: float = 4.0,
    max_iter: int = 8,
) -> Any:
    """Shrink labels or grow the figure until label height fits the row spacing."""
    if len(offsets) < 2:
        return ax

    fig = ax.figure
    labels = [label for label in ax.get_yticklabels() if label.get_text()]
    if not labels:
        return ax

    for _ in range(max_iter):
        fig.canvas.draw()
        labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if not labels:
            return ax

        renderer = fig.canvas.get_renderer()
        max_label_height_px = max(label.get_window_extent(renderer=renderer).height for label in labels)
        p0 = ax.transData.transform((0.0, float(offsets[0])))[1]
        p1 = ax.transData.transform((0.0, float(offsets[1])))[1]
        spacing_px = abs(float(p1 - p0))
        if spacing_px <= 0:
            return ax

        ratio = max_label_height_px / spacing_px
        if ratio > target_ratio:
            current_font = labels[0].get_fontsize()
            if current_font > min_fontsize + 0.05:
                scale = max(target_ratio / ratio * 0.98, min_fontsize / current_font)
                new_font = max(min_fontsize, current_font * scale)
                for label in labels:
                    label.set_fontsize(new_font)
                continue

            width, height = fig.get_size_inches()
            new_height = max(float(min_height), height * (ratio / target_ratio) * 1.02)
            if abs(new_height - height) < 0.05:
                break
            fig.set_size_inches(width, new_height, forward=True)
            continue

        if ratio < target_ratio * 0.65:
            width, height = fig.get_size_inches()
            shrink = max(ratio / target_ratio, 0.75)
            new_height = max(float(min_height), height * shrink)
            if abs(new_height - height) < 0.05:
                break
            fig.set_size_inches(width, new_height, forward=True)
            continue

        break

    return ax


def plot_input_raster(
    result: dict[str, Any],
    ax: Any = None,
    max_segments: int = 80,
    target_types: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Plot the saved odor-input event raster."""
    rows = sorted(filter_input_events(result, target_types=target_types), key=lambda row: row[0])[:max_segments]
    ax = _ensure_raster_axis(ax, len(rows), width=14.0, min_height=4.0, per_row_height=0.10)
    if not rows:
        ax.set_title("No input events saved")
        return ax
    times = [row[1] for row in rows]
    labels = [row[0].replace("h.", "") for row in rows]
    offsets = _style_raster_axis(
        ax,
        labels,
        ylabel="Input Segment",
        title="Odor Input Raster",
        fontsize=_recommended_raster_fontsize(len(rows)),
        line_spacing=1.4,
    )
    ax.eventplot(times, colors="black", lineoffsets=offsets, linelengths=1.0)
    _fit_raster_labels(ax, offsets, min_height=4.0)
    return ax


def plot_input_rate(
    result: dict[str, Any],
    bin_ms: float = 5.0,
    smooth_sigma_ms: float = 10.0,
    normalization: str = "per_target_cell",
    ax: Any = None,
) -> Any:
    """Plot normalized odor-input event-rate traces over time."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    traces = [
        ("All inputs", None, "black"),
        ("To MCs", ["MC"], "tab:blue"),
        ("To TCs", ["TC"], "tab:red"),
    ]
    plotted = False
    ylabel = None
    for base_label, target_types, color in traces:
        t, rate_hz, meta = compute_input_rate(
            result,
            bin_ms=bin_ms,
            smooth_sigma_ms=smooth_sigma_ms,
            target_types=target_types,
            normalization=normalization,
            return_metadata=True,
        )
        if len(t) == 0:
            continue
        ylabel = meta["unit"]
        ax.plot(t, rate_hz, color=color, linewidth=1.2, label=_rate_series_label(base_label, meta))
        plotted = True

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel or "events/s")
    ax.set_title("Odor Input Event Rate")
    if plotted:
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No input events saved", ha="center", va="center", transform=ax.transAxes)
    return ax


def plot_voltage_traces(result: dict[str, Any], max_per_type: int = 4, ax: Any = None) -> Any:
    """Plot a small representative subset of saved soma voltages."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    grouped = split_traces_by_type(result)
    offset = 0.0
    colors = {"MC": "tab:blue", "TC": "tab:red", "GC": "tab:orange", "other": "tab:gray"}
    for cell_type in ("MC", "TC", "GC"):
        for label, t, v in grouped[cell_type][:max_per_type]:
            ax.plot(t, v + offset, color=colors[cell_type], linewidth=1.0, label=label)
            offset += 120.0 if cell_type != "GC" else 40.0
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Offset Voltage")
    ax.set_title("Sample Soma Voltages")
    if ax.lines:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    return ax


def plot_spike_raster(
    result: dict[str, Any],
    threshold: float = 0.0,
    max_cells_per_type: int = 24,
    ax: Any = None,
) -> Any:
    """Plot a soma-spike raster derived from the saved voltage traces."""
    grouped = split_traces_by_type(result)
    rows = []
    for cell_type in ("MC", "TC", "GC"):
        rows.extend(grouped[cell_type][:max_cells_per_type])
    ax = _ensure_raster_axis(ax, len(rows), width=14.0, min_height=4.5, per_row_height=0.10)
    if not rows:
        ax.set_title("No soma traces saved")
        return ax
    spike_times = [detect_spikes(t, v, threshold=threshold) for _label, t, v in rows]
    colors = [
        "tab:blue" if label.startswith("MC") else "tab:red" if label.startswith("TC") else "tab:orange"
        for label, _t, _v in rows
    ]
    offsets = _style_raster_axis(
        ax,
        [label for label, _t, _v in rows],
        ylabel="Cell",
        title="Detected Soma Spike Raster",
        fontsize=_recommended_raster_fontsize(len(rows)),
        line_spacing=1.3,
    )
    ax.eventplot(spike_times, colors=colors, lineoffsets=offsets, linelengths=1.0)
    _fit_raster_labels(ax, offsets, min_height=4.5)
    return ax


def plot_gc_output_event_raster(
    result: dict[str, Any],
    max_connections: int = 120,
    target_types: list[str] | tuple[str, ...] | None = None,
    ax: Any = None,
    *,
    fontsize: float = 7,
    line_spacing: float = 1.4,
) -> Any:
    """Plot the saved reciprocal GC inhibitory-output event raster."""
    rows = filter_gc_output_events(result, target_types=target_types)[:max_connections]
    ax = _ensure_raster_axis(ax, len(rows), width=16.0, min_height=4.5, per_row_height=0.10)
    if not rows:
        ax.set_title("No GC inhibitory-output events saved")
        return ax

    times = [np.asarray(row.get("times", []), dtype=float) for row in rows]
    labels = [
        f"{normalize_cell_name(row.get('source_section', 'GC'))}->{normalize_cell_name(row.get('dest_section', 'cell'))}"
        for row in rows
    ]
    offsets = _style_raster_axis(
        ax,
        labels,
        ylabel="Reciprocal GABA Connection",
        title="GC Inhibitory Output Events",
        fontsize=min(float(fontsize), _recommended_raster_fontsize(len(rows), default=float(fontsize))),
        line_spacing=line_spacing,
    )
    ax.eventplot(times, lineoffsets=offsets, linelengths=1.0, colors="black")
    _fit_raster_labels(ax, offsets, min_height=4.5)
    return ax


def plot_gc_output_rate(
    result: dict[str, Any],
    bin_ms: float = 5.0,
    smooth_sigma_ms: float = 10.0,
    normalization: str = "per_target_cell",
    ax: Any = None,
) -> Any:
    """Plot normalized GC inhibitory-output rate traces over time."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    traces = [
        ("All targets", None, "black"),
        ("To MCs", ["MC"], "tab:blue"),
        ("To TCs", ["TC"], "tab:red"),
    ]
    plotted = False
    ylabel = None
    for base_label, target_types, color in traces:
        t, rate_hz, meta = compute_gc_output_rate(
            result,
            bin_ms=bin_ms,
            smooth_sigma_ms=smooth_sigma_ms,
            target_types=target_types,
            normalization=normalization,
            return_metadata=True,
        )
        if len(t) == 0:
            continue
        ylabel = meta["unit"]
        ax.plot(t, rate_hz, color=color, linewidth=1.2, label=_rate_series_label(base_label, meta))
        plotted = True

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel or "events/s")
    ax.set_title("GC Inhibitory Output Rate")
    if plotted:
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GC inhibitory-output events saved", ha="center", va="center", transform=ax.transAxes)
    return ax


def plot_input_overview(
    result: dict[str, Any],
    bin_ms: float = 5.0,
    smooth_sigma_ms: float = 10.0,
    max_segments: int = 120,
    normalization: str = "per_target_cell",
) -> tuple[Any, Any]:
    """Render the standard input raster + input-rate overview figure."""
    rows = sorted(result.get("input_times", []), key=lambda row: row[0])[:max_segments]
    n_rows = len(rows)
    label_fontsize = _recommended_raster_fontsize(n_rows)
    line_spacing = 1.6 if n_rows > 80 else 1.4
    raster_height = _recommended_raster_height(n_rows, min_height=4.5)
    rate_height = 4.0
    total_height = raster_height + rate_height

    max_label_len = max((len(row[0].replace("h.", "")) for row in rows), default=0)
    left_margin = min(0.5, max(0.22, 0.15 + 0.006 * max_label_len))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, total_height),
        sharex=False,
        gridspec_kw={"height_ratios": [raster_height, rate_height]},
    )
    plot_input_raster(
        result,
        ax=axes[0],
        max_segments=max_segments,
    )
    plot_input_rate(
        result,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        normalization=normalization,
        ax=axes[1],
    )
    fig.subplots_adjust(left=left_margin, hspace=0.25)
    return fig, axes


def plot_gc_output_overview(
    result: dict[str, Any],
    bin_ms: float = 5.0,
    smooth_sigma_ms: float = 10.0,
    max_connections: int = 120,
    normalization: str = "per_target_cell",
) -> tuple[Any, Any]:
    """Render the standard GC output raster + rate overview figure."""
    rows = filter_gc_output_events(result)[:max_connections]
    n_rows = len(rows)
    label_fontsize = _recommended_raster_fontsize(n_rows)
    line_spacing = 1.6 if n_rows > 80 else 1.4
    raster_height = _recommended_raster_height(n_rows, min_height=4.5)
    rate_height = 4.0
    total_height = raster_height + rate_height

    max_label_len = 0
    for row in rows:
        label = (
            f"{normalize_cell_name(row.get('source_section', 'GC'))}->"
            f"{normalize_cell_name(row.get('dest_section', 'cell'))}"
        )
        max_label_len = max(max_label_len, len(label))

    left_margin = min(0.5, max(0.22, 0.15 + 0.007 * max_label_len))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, total_height),
        sharex=False,
        gridspec_kw={"height_ratios": [raster_height, rate_height]},
    )
    plot_gc_output_event_raster(
        result,
        max_connections=max_connections,
        ax=axes[0],
        fontsize=label_fontsize,
        line_spacing=line_spacing,
    )
    plot_gc_output_rate(
        result,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        normalization=normalization,
        ax=axes[1],
    )
    fig.subplots_adjust(left=left_margin, hspace=0.25)
    return fig, axes


def plot_lfp_overview(
    result: dict[str, Any],
    dt_ms: float = 0.1,
    lowcut_hz: float = 30.0,
    highcut_hz: float = 120.0,
) -> tuple[Any, Any]:
    """Plot raw LFP, band-passed LFP, and a Welch PSD summary."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    t = result["lfp_t"]
    lfp = result["lfp"]
    axes[0].plot(t, lfp, color="black", linewidth=1.0)
    axes[0].set_title("Raw LFP")
    axes[0].set_ylabel("LFP")

    bp_t, bp_lfp = compute_lfp_bandpassed(result, dt_ms=dt_ms, lowcut_hz=lowcut_hz, highcut_hz=highcut_hz)
    axes[1].plot(bp_t, bp_lfp, color="tab:purple", linewidth=1.0)
    axes[1].set_title(f"Band-passed LFP ({lowcut_hz:.0f}-{highcut_hz:.0f} Hz)")
    axes[1].set_ylabel("Filtered LFP")

    fs_hz = 1000.0 / float(np.median(np.diff(bp_t)))
    freqs, power = welch(bp_lfp, fs=fs_hz, nperseg=min(2048, len(bp_lfp)))
    axes[2].plot(freqs, power, color="tab:green", linewidth=1.0)
    axes[2].set_xlim(0, 150)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("PSD")
    axes[2].set_title("Welch Power Spectrum")
    fig.tight_layout()
    return fig, axes


def plot_named_signal(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    ax: Any = None,
) -> Any:
    """Plot one named analysis signal as a time trace."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    t, y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    ax.plot(t, y, linewidth=1.0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(signal)
    ax.set_title(f"{signal} Trace")
    return ax


def plot_spectrogram(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    max_freq_hz: float = 150.0,
    nperseg: int = 512,
    noverlap: int = 448,
    ax: Any = None,
) -> Any:
    """Plot a spectrogram for a named analysis signal."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    ax = ax or plt.subplots(figsize=(14, 5))[1]
    times_ms, freqs, power = compute_spectrogram(
        signal_t,
        signal_y,
        dt_ms=dt_ms,
        max_freq_hz=max_freq_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    mesh = ax.pcolormesh(times_ms, freqs, 10.0 * np.log10(power + 1e-12), shading="auto")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{signal.upper()} Spectrogram")
    plt.colorbar(mesh, ax=ax, label="Power (dB)")
    return ax


def plot_wavelet(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    ax: Any = None,
) -> Any:
    """Plot the continuous wavelet power map for a named signal."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    ax = ax or plt.subplots(figsize=(14, 5))[1]
    t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
    mesh = ax.pcolormesh(t, freqs, power, shading="auto")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{signal.upper()} Wavelet Power")
    plt.colorbar(mesh, ax=ax, label="log(1 + |cwt|)")
    return ax


def plot_wavelet_band_power(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    bands: dict[str, tuple[float, float]] | None = None,
    ax: Any = None,
) -> Any:
    """Plot band-collapsed wavelet power traces over time."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    t, _freqs, _power, traces = compute_wavelet_band_power(signal_t, signal_y, bands=bands, dt_ms=dt_ms)
    for name, values in traces.items():
        ax.plot(t, values, linewidth=1.2, label=name)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean Wavelet Power")
    ax.set_title("Band Power Over Time")
    ax.legend(loc="upper right")
    return ax


def _format_sweep_value(value: Any) -> str:
    """Format a sweep value compactly for figure titles."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _safe_name(name: Any) -> str:
    """Make a filesystem-safe artifact basename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "animation"


def animate_lfp_sweep(
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    interval: int = 1000,
) -> animation.FuncAnimation:
    """Animate trace-style outputs across a one-parameter sweep."""
    if signal != "lfp":
        traces = [get_named_signal(item["result"], signal=signal, dt_ms=dt_ms) for item in sweep["items"]]
        y_min = min(float(np.min(y)) for _t, y in traces)
        y_max = max(float(np.max(y)) for _t, y in traces)
        fig, ax = plt.subplots(figsize=(12, 4))
        line, = ax.plot([], [], linewidth=1.2)
        ax.set_ylim(y_min, y_max if y_max > y_min else y_min + 1e-9)

        def update(frame_index):
            t, y = traces[frame_index]
            line.set_data(t, y)
            ax.set_xlim(float(t[0]), float(t[-1]) if len(t) else 1.0)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(signal)
            ax.set_title(f"{signal} | {sweep['path']} = {_format_sweep_value(sweep['items'][frame_index]['value'])}")
            return [line]

        anim = animation.FuncAnimation(fig, update, frames=len(sweep["items"]), interval=interval, repeat=True)
        plt.close(fig)
        return anim

    legacy_items = [load_legacy_wavelet_analysis(item["result"], dt=dt_ms, sniff_count=8) for item in sweep["items"]]
    raw_min = min(float(np.min(item["lfp"] * 1000)) for item in legacy_items)
    raw_max = max(float(np.max(item["lfp"] * 1000)) for item in legacy_items)
    bp_min = min(float(np.min(item["lfp_bp"] * 10000 - 200)) for item in legacy_items)
    bp_max = max(float(np.max(item["lfp_bp"] * 10000 - 200)) for item in legacy_items)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    raw_line, = axes[0].plot([], [], linewidth=1.0)
    bp_line, = axes[1].plot([], [], linewidth=1.0, color="tab:purple")
    axes[0].set_ylim(raw_min, raw_max if raw_max > raw_min else raw_min + 1e-9)
    axes[1].set_ylim(bp_min, bp_max if bp_max > bp_min else bp_min + 1e-9)

    def update(frame_index):
        item = legacy_items[frame_index]
        raw_line.set_data(item["t"], item["lfp"] * 1000)
        bp_line.set_data(item["t"], item["lfp_bp"] * 10000 - 200)
        axes[0].set_xlim(float(item["t"][0]), float(item["t"][-1]) if len(item["t"]) else 1.0)
        axes[0].set_ylabel("Raw LFP x1000")
        axes[1].set_ylabel("BP LFP x10000 - 200")
        axes[1].set_xlabel("Simulation Time [ms]")
        axes[0].set_title(f"LFP view | {sweep['path']} = {_format_sweep_value(sweep['items'][frame_index]['value'])}")
        return [raw_line, bp_line]

    anim = animation.FuncAnimation(fig, update, frames=len(sweep["items"]), interval=interval, repeat=True)
    plt.close(fig)
    return anim


def animate_spectrogram_sweep(
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    max_freq_hz: float = 150.0,
    nperseg: int = 512,
    noverlap: int = 448,
    interval: int = 1000,
) -> animation.FuncAnimation:
    """Animate spectrograms across a one-parameter sweep."""
    specs = []
    vmin = None
    vmax = None
    for item in sweep["items"]:
        signal_t, signal_y = get_named_signal(item["result"], signal=signal, dt_ms=dt_ms)
        times_ms, freqs, power = compute_spectrogram(
            signal_t,
            signal_y,
            dt_ms=dt_ms,
            max_freq_hz=max_freq_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        db = 10.0 * np.log10(power + 1e-12)
        specs.append((times_ms, freqs, db))
        vmin = float(np.min(db)) if vmin is None else min(vmin, float(np.min(db)))
        vmax = float(np.max(db)) if vmax is None else max(vmax, float(np.max(db)))

    fig, ax = plt.subplots(figsize=(12, 4))

    def update(frame_index):
        ax.clear()
        times_ms, freqs, db = specs[frame_index]
        mesh = ax.pcolormesh(times_ms, freqs, db, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{signal} spectrogram | {sweep['path']} = {_format_sweep_value(sweep['items'][frame_index]['value'])}")
        return [mesh]

    anim = animation.FuncAnimation(fig, update, frames=len(sweep["items"]), interval=interval, repeat=True)
    plt.close(fig)
    return anim


def animate_wavelet_sweep(
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    interval: int = 1000,
) -> animation.FuncAnimation:
    """Animate wavelet maps across a one-parameter sweep."""
    maps = []
    for item in sweep["items"]:
        if signal == "lfp":
            legacy = load_legacy_wavelet_analysis(item["result"], dt=dt_ms, sniff_count=8)
            maps.append((legacy["t"], legacy["frequencies"], legacy["lfp_wavelet_power"]))
        else:
            signal_t, signal_y = get_named_signal(item["result"], signal=signal, dt_ms=dt_ms)
            t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
            maps.append((t, freqs, power))

    fig, ax = plt.subplots(figsize=(12, 4))

    def update(frame_index):
        ax.clear()
        t, freqs, power = maps[frame_index]
        mesh = ax.contourf(t, freqs, power, 256, cmap="jet")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim((20, 140))
        ax.set_title(f"{signal} wavelet | {sweep['path']} = {_format_sweep_value(sweep['items'][frame_index]['value'])}")
        return [mesh]

    anim = animation.FuncAnimation(fig, update, frames=len(sweep["items"]), interval=interval, repeat=True)
    plt.close(fig)
    return anim


def animate_sniff_average_sweep(
    sweep: dict[str, Any],
    dt_ms: float = 0.1,
    sniff_count: int = 8,
    interval: int = 1000,
) -> animation.FuncAnimation:
    """Animate sniff-averaged wavelet views across a sweep."""
    maps = [load_legacy_wavelet_analysis(item["result"], dt=dt_ms, sniff_count=sniff_count) for item in sweep["items"]]
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame_index):
        ax.clear()
        item = maps[frame_index]
        mesh = ax.contourf(
            item["t_average"],
            item["frequencies"],
            item["lfp_wavelet_power_average"],
            256,
            cmap="jet",
        )
        ax.set_ylim((20, 140))
        ax.set_xlabel("Time Since Sniff Onset [ms]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"Sniff average | {sweep['path']} = {_format_sweep_value(sweep['items'][frame_index]['value'])}")
        return [mesh]

    anim = animation.FuncAnimation(fig, update, frames=len(sweep["items"]), interval=interval, repeat=True)
    plt.close(fig)
    return anim


def save_animation(
    anim: animation.FuncAnimation,
    name: str,
    output_dir: str | Path | None = None,
    fps: int = 2,
) -> Path:
    """Save an animation as a GIF and return the written path."""
    output_dir = Path(output_dir or (DEFAULT_RESULTS_BASE / "animations" / make_timestamp()))
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{_safe_name(name)}.gif"
    writer = animation.PillowWriter(fps=max(1, int(fps)))
    anim.save(str(gif_path), writer=writer)
    return gif_path


def save_figure(
    name: str,
    fig: Any = None,
    run_or_result: RunRecord | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    dpi: int = 200,
    close: bool = False,
) -> Path:
    """Save a Matplotlib figure near a run directory or in a timestamped folder."""
    fig = fig or plt.gcf()

    if output_dir is None and run_or_result is not None:
        if isinstance(run_or_result, RunRecord):
            output_dir = Path(run_or_result.result_dir)
        elif isinstance(run_or_result, dict) and "result_dir" in run_or_result:
            output_dir = Path(run_or_result["result_dir"])

    output_dir = Path(output_dir or (DEFAULT_RESULTS_BASE / "figures" / make_timestamp()))
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{_safe_name(name)}.png"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")

    if close:
        plt.close(fig)

    return png_path


def show_all_outputs(result: dict[str, Any], config: dict[str, Any] | None = None) -> None:
    """Render the standard notebook figure set for one loaded result."""
    config = config or {}
    dt_ms = float(config.get("analysis_dt_ms", 0.1))
    input_bin_ms = float(config.get("input_bin_ms", 5.0))
    input_smooth_ms = float(config.get("input_smooth_sigma_ms", 10.0))
    input_max_segments = int(config.get("input_max_segments", 120))
    input_norm = str(config.get("input_rate_normalization", "per_target_cell"))
    max_voltage = int(config.get("max_voltage_traces_per_type", 4))
    max_raster = int(config.get("max_spike_raster_cells_per_type", 24))
    gc_bin_ms = float(config.get("gc_output_bin_ms", 5.0))
    gc_smooth_ms = float(config.get("gc_output_smooth_sigma_ms", 10.0))
    gc_max_connections = int(config.get("gc_output_max_connections", 120))
    gc_norm = str(config.get("gc_output_rate_normalization", "per_target_cell"))
    sniff_count = int(config.get("sniff_count", 8))

    show_legacy_plots(result, sniff_count=sniff_count, dt=dt_ms)

    plot_input_overview(
        result,
        bin_ms=input_bin_ms,
        smooth_sigma_ms=input_smooth_ms,
        max_segments=input_max_segments,
        normalization=input_norm,
    )
    plt.show()

    plot_voltage_traces(result, max_per_type=max_voltage)
    plt.show()

    plot_spike_raster(result, max_cells_per_type=max_raster)
    plt.show()

    plot_gc_output_overview(
        result,
        bin_ms=gc_bin_ms,
        smooth_sigma_ms=gc_smooth_ms,
        max_connections=gc_max_connections,
        normalization=gc_norm,
    )
    plt.show()

    plot_lfp_overview(result, dt_ms=dt_ms)
    plt.show()

    plot_spectrogram(result, signal=config.get("spectrogram_signal", "lfp"), dt_ms=dt_ms)
    plt.show()

    plot_wavelet(result, signal=config.get("wavelet_signal", "lfp"), dt_ms=dt_ms)
    plt.show()

    plot_wavelet_band_power(result, signal=config.get("wavelet_signal", "lfp"), dt_ms=dt_ms)
    plt.show()


def print_run_summary(
    run: RunRecord,
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Print a concise run summary plus param/runtime diffs for notebook use."""
    info = result_overview(result)
    print(json.dumps(info, indent=2, sort_keys=True))
    config = config or run.config or (result.get("run_info") or {}).get("config") or {}
    remote_info = (result.get("run_info") or {}).get("remote")
    if config:
        normalized_config = build_run_config(**config)
        effective = (result.get("run_info") or {}).get("effective_params") or {}
        if "full_param_snapshot" not in effective:
            effective = resolve_effective_params(normalized_config)
        print("\nEffective inputs:")
        print(json.dumps({
            "input_odors_source": effective["input_odors_source"],
            "n_odor_presentations": effective["n_odor_presentations"],
            "odor_names": effective["odor_names"],
            "input_odors": effective["input_odors"],
            "max_firing_rate_hz": effective["max_firing_rate_hz"],
            "inhale_duration_ms": effective["inhale_duration_ms"],
            "mc_input_weight": effective["mc_input_weight"],
            "tc_input_weight": effective["tc_input_weight"],
        }, indent=2, sort_keys=True))

        base_snapshot = resolve_paramset_defaults(normalized_config["paramset"])
        full_snapshot = effective.get("full_param_snapshot", {})
        param_changes = diff_values(base_snapshot, full_snapshot)
        print_diff_section("Requested/effective param changes vs clean paramset", param_changes)

        print("\nRuntime and analysis controls:")
        print(json.dumps(extract_runtime_control_snapshot(normalized_config), indent=2, sort_keys=True))
        if remote_info:
            print("\nRemote execution metadata:")
            print(json.dumps(remote_info, indent=2, sort_keys=True))
    print(f"\nResult directory: {run.result_dir}")
    print(f"Command: {' '.join(run.command)}")


# ---------------------------------------------------------------------------
# Config persistence helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Save a notebook run config dict to a JSON file for future reproduction.

    The saved file can be reloaded with :func:`load_config` and passed directly
    to :func:`run_simulation` or :func:`run_and_load`.

    Parameters
    ----------
    config:
        A config dict as returned by :func:`build_run_config`.
    path:
        Destination file path (JSON). Parent directories are created as needed.

    Returns
    -------
    Path
        The resolved path that was written.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(config)), indent=2, sort_keys=True))
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a previously saved run config from a JSON file.

    The returned dict can be passed directly to :func:`run_simulation` or
    :func:`run_and_load`.  Odor-schedule keys are normalized back to numeric
    types after JSON round-trip.

    Parameters
    ----------
    path:
        Path to a JSON config file previously written by :func:`save_config`.
    """
    path = Path(path).expanduser().resolve()
    with open(path) as f:
        data = json.load(f)
    if data.get("input_odors") is not None:
        data["input_odors"] = normalize_input_odors(data["input_odors"])
    return data


def config_from_run(
    run_or_dir: RunRecord | str | Path | None = None,
    *,
    prefix: str | None = None,
    index: int = -1,
    results_base: str | Path = DEFAULT_RESULTS_BASE,
) -> dict[str, Any]:
    """Extract the original notebook config from a past run.

    The returned config is a deep copy of the dict originally passed to
    :func:`run_simulation`, ready to be fed back unchanged (for exact
    reproduction) or modified before re-running.

    Parameters
    ----------
    run_or_dir:
        A :class:`RunRecord`, a path to a result directory, or ``None`` to
        select by *prefix* / *index*.
    prefix:
        Optional label prefix filter when *run_or_dir* is ``None``.
    index:
        Index into the sorted run list when *run_or_dir* is ``None``.
        Defaults to ``-1`` (most recent).
    results_base:
        Base directory for notebook runs.

    Example
    -------
    ::

        cfg = config_from_run()          # most recent run
        cfg["gaba_tau2_ms"] = 50         # tweak one parameter
        run, result = run_and_load(cfg)  # re-run with the change
    """
    record = load_run_record(
        run_or_dir, prefix=prefix, index=index, results_base=results_base
    )
    return deepcopy(record.config)


def list_saved_configs(directory: str | Path | None = None) -> list[Path]:
    """Return a sorted list of JSON config files in *directory*.

    Defaults to the ``configs/`` directory at the repository root.  Returns an
    empty list when the directory does not exist.

    Parameters
    ----------
    directory:
        Directory to search.  Defaults to ``<repo_root>/configs``.
    """
    directory = Path(directory).expanduser().resolve() if directory else DEFAULT_CONFIGS_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def list_paramsets(
    include_saved: bool = False,
    configs_dir: str | Path | None = None,
) -> list[str] | dict[str, list]:
    """Return available paramset sources.

    By default returns a sorted list of built-in paramset class names that can
    be used as the ``paramset`` key in :func:`build_run_config`.

    When *include_saved* is ``True``, returns a dict with two keys:

    * ``"builtin"`` — sorted list of Python paramset class names.
    * ``"saved"``   — sorted list of :class:`~pathlib.Path` objects pointing to
      JSON config files in *configs_dir* (defaults to ``<repo_root>/configs``).

    Use :func:`load_config` to load a saved config file and pass it directly to
    :func:`run_simulation` or :func:`run_and_load`.

    Parameters
    ----------
    include_saved:
        When ``True``, also include saved JSON configs from *configs_dir*.
    configs_dir:
        Directory to search for saved JSON configs.  Defaults to
        ``<repo_root>/configs``.

    Example
    -------
    ::

        # Built-in paramsets only
        list_paramsets()
        # ['GammaSignature', 'GammaSignature_DifferentOdor', ...]

        # Both built-in and saved configs
        sources = list_paramsets(include_saved=True)
        # {
        #   'builtin': ['GammaSignature', 'PureMCs', ...],
        #   'saved':   [PosixPath('configs/my_experiment.json'), ...]
        # }
        cfg = load_config(sources['saved'][0])
    """
    import olfactorybulb.model as obmodel
    from olfactorybulb.paramsets.base import SilentNetwork

    names = sorted(
        name
        for name, obj in vars(obmodel).items()
        if isinstance(obj, type)
        and issubclass(obj, SilentNetwork)
        and obj is not SilentNetwork
    )

    if not include_saved:
        return names

    return {
        "builtin": names,
        "saved": list_saved_configs(configs_dir),
    }


def config_diff(
    config1: dict[str, Any],
    config2: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare two run configs at the effective-params level.

    Resolves the full paramset snapshot for each config and returns a list of
    changed paths.  Each entry has the keys ``path``, ``before``, and
    ``after``.  Only parameters that differ between the two configs appear in
    the result.

    Parameters
    ----------
    config1:
        The "before" config dict.
    config2:
        The "after" config dict.

    Example
    -------
    ::

        base = build_run_config(paramset="GammaSignature")
        tweaked = build_run_config(paramset="GammaSignature", gaba_tau2_ms=50)
        changes = config_diff(base, tweaked)
        print_diff_section("Changes", changes)
    """
    snap1 = resolve_effective_params(config1)["full_param_snapshot"]
    snap2 = resolve_effective_params(config2)["full_param_snapshot"]
    return diff_values(snap1, snap2)


if __name__ == "__main__":
    config = build_run_config(paramset="OneMsTest", tstop_ms=1.0, label_prefix="helper_smoke")
    run, result = run_and_load(config)
    print_run_summary(run, result)
