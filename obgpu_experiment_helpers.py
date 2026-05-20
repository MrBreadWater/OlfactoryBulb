"""Notebook-facing helpers for running, loading, and analyzing OBGPU simulations.

This module is the maintained convenience layer for the interactive notebooks in
``notebooks/``. It keeps heavy NEURON work in subprocesses when possible so
notebook reruns do not corrupt the live HOC state.
"""

from __future__ import annotations

import atexit
import json
import os
import pickle
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import builtins
import warnings
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
try:
    import paramiko
except ImportError:  # pragma: no cover - optional runtime dependency
    paramiko = None
try:
    from tqdm.std import tqdm as _tqdm_plain
except ImportError:  # pragma: no cover - optional runtime dependency
    try:
        from tqdm import tqdm as _tqdm_plain
    except ImportError:  # pragma: no cover - optional runtime dependency
        _tqdm_plain = None
try:
    from tqdm.notebook import tqdm as _tqdm_notebook
except ImportError:  # pragma: no cover - optional runtime dependency
    _tqdm_notebook = None

if paramiko is None:  # pragma: no cover - optional runtime dependency
    _PARAMIKO_PARTIAL_AUTH_EXC: tuple[type[BaseException], ...] = ()
else:
    _PARAMIKO_PARTIAL_AUTH_EXC = tuple(
        exc
        for exc in (
            getattr(paramiko, "PartialAuthentication", None),
            getattr(getattr(paramiko, "ssh_exception", None), "PartialAuthentication", None),
        )
        if exc is not None
    )
tqdm = _tqdm_plain or _tqdm_notebook
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, hilbert, lfilter, spectrogram, welch
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
    "input_stimuli": "Custom InputSpec-driven stimuli keyed by onset ms. Cannot be combined with input_odors.",
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
    "ketamine_block": "Semantic NMDA block multiplier on AmpaNmdaSyn NMDA current.",
    "ampa_block": "AMPA current multiplier on AmpaNmdaSyn AMPA current.",
    "gaba_gmax": "Global GabaSyn gmax.",
    "gaba_tau2_ms": "Global GabaSyn tau2.",
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
    "enable_reciprocal_synapses": "Toggle GC<->MC/TC reciprocal synapses.",
    "extra_overrides": "Any raw paramset overrides not exposed above.",
    "spectrogram_signal": "Signal for spectrogram plots, e.g. 'lfp', 'mean_MC_voltage', or 'MC5[0].soma'.",
    "wavelet_signal": "Signal for wavelet plots, e.g. 'lfp', 'mean_TC_voltage', or a soma label.",
    "runner_backend": "Execution backend: 'local', 'sol_slurm', or 'slurm_remote'.",
    "use_corenrn": "Local-run CoreNEURON toggle. Remote Slurm runs infer this from the Slurm resource request unless you explicitly override it after applying the remote config.",
    "use_gpu": "Local-run GPU toggle. Remote Slurm runs infer this from slurm_gpus unless you explicitly override it after applying the remote config.",
    "mpi_exec": "MPI launcher for local notebook runs, e.g. 'mpiexec' or 'srun --mpi=pmi2'.",
    "remote_mpi_exec": "MPI launcher on the remote host, e.g. 'srun' or 'mpiexec'.",
    "remote_host": "SSH target used by the Sol backend, e.g. 'user@sol.asu.edu'.",
    "remote_repo_root": "Absolute repo path on Sol.",
    "remote_results_root": "Remote root directory where timestamped notebook runs are written.",
    "remote_conda_activate_cmd": "Shell snippet used on the remote cluster before launching the benchmark command. Generic remote runs default to 'source tools/setup/activate_obgpu.sh'; Sol uses 'source tools/setup/activate_sol_obgpu.sh'.",
    "remote_runtime_profiles": "Optional ordered runtime-profile selectors. Each profile can match node arch/features and choose an activation command plus mechanism profile.",
    "remote_fallback_conda_activate_cmd": "Optional shell snippet used when the allocated Slurm nodes do not all match remote_fast_node_feature.",
    "remote_fast_node_feature": "Optional Slurm node feature required for the primary remote environment, e.g. 'cascadelake'.",
    "remote_mechanism_profile": "Mechanism build/cache profile for the primary remote environment. 'default' uses remote_repo_root/x86_64.",
    "remote_fallback_mechanism_profile": "Mechanism build/cache profile for the fallback remote environment. Non-default profiles use .obgpu-mechanisms/<profile>.",
    "remote_repo_mode": "How Sol should choose the repo tree for a run: 'shared' temporarily checks out the requested commit in remote_repo_root and restores it afterward, while 'snapshot' stages a detached per-run worktree.",
    "remote_git_ref": "Optional git commit, tag, or branch for Sol runs. Defaults to the current local HEAD commit so notebook runs can auto-publish exact code.",
    "remote_git_fetch": "When True, fetch the configured remote on Sol before using remote_git_ref.",
    "remote_git_remote": "Git remote name on Sol used when remote_git_fetch=True. Defaults to 'origin'.",
    "slurm_allocation_job_id": "Optional existing Slurm allocation/job id to reuse for notebook runs instead of submitting a fresh sbatch job.",
    "slurm_reuse_allocation": "When True, cache one reusable Slurm allocation in the notebook runtime and launch runs as srun steps inside it.",
    "slurm_allocation_time": "Optional walltime for the cached reusable allocation. Defaults to slurm_time when unset.",
    "slurm_allocation_name": "Optional job-name prefix for cached reusable allocations. Defaults to 'obgpu_notebook_alloc'.",
    "remote_poll_interval_s": "Polling interval in seconds for remote Slurm jobs.",
    "remote_live_status": "When True, print live remote Slurm state updates in the notebook while polling.",
    "remote_live_logs": "When True, stream remote bootstrap/stdout/stderr/slurm log updates into the notebook while polling.",
    "remote_heartbeat_timeout_s": "Remote Slurm watchdog timeout in seconds. Notebook-managed jobs and reusable allocations self-terminate if the notebook stops refreshing their heartbeat for longer than this.",
    "remote_cleanup_stale_allocations": "When True, cancel stale or pre-heartbeat notebook-managed reusable allocations on the remote before submitting a new run.",
    "remote_sync_compress": "When True, compress the remote result directory before downloading it back to the notebook.",
    "slurm_partition": "Optional Slurm partition for remote submission. Set it explicitly when needed; None omits --partition entirely.",
    "slurm_account": "Optional Slurm account for remote submission.",
    "slurm_time": "Optional Slurm walltime, e.g. '02:00:00'.",
    "slurm_gpus": "Optional GPU count requested from Slurm.",
    "slurm_cpus_per_task": "Optional CPU count requested per Slurm task.",
    "slurm_mem": "Optional Slurm memory request, e.g. '32G'.",
    "slurm_extra_args": "Optional extra sbatch arguments passed as raw strings.",
    "ssh_binary": "SSH client executable used by the remote backend.",
    "ssh_options": "Extra SSH options, e.g. ['-J', 'jumphost'].",
    "ssh_transport": "Remote transport for the Sol backend: 'auto', 'paramiko', or 'openssh'.",
    "ssh_multiplex": "When True, reuse a persistent SSH control socket so Sol auth happens once instead of on every submit/poll/sync step.",
    "ssh_allow_interactive_auth": "When True, open the initial SSH control master through an interactive PTY so notebook-launched Sol runs can prompt for password and 2FA once.",
    "ssh_keepalive_s": "Paramiko keepalive interval in seconds for notebook-managed SSH sessions. Higher values reduce background traffic; lower values make idle sessions less likely to die between runs.",
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


def _format_bytes(num_bytes: int | float) -> str:
    """Return a compact human-readable byte count."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def _render_progress_bar(current: int | float, total: int | float, width: int = 24) -> str:
    """Render a compact ASCII progress bar."""
    if total <= 0:
        return "[" + ("?" * width) + "]"
    progress = max(0.0, min(float(current) / float(total), 1.0))
    filled = int(round(progress * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_progress_value(value: int | float, unit: str, unit_scale: bool) -> str:
    """Format one progress value using either byte or plain-unit rendering."""
    if unit_scale and unit == "B":
        return _format_bytes(value)
    if unit:
        return f"{float(value):.1f} {unit}" if isinstance(value, float) or isinstance(value, np.floating) else f"{int(value)} {unit}"
    return str(value)


def _progress_write(message: str) -> None:
    """Write one progress message without corrupting active tqdm bars."""
    global tqdm
    if tqdm is not None:
        try:
            tqdm.write(message)
            return
        except Exception:
            tqdm = _tqdm_plain
            if tqdm is not None:
                try:
                    tqdm.write(message)
                    return
                except Exception:
                    pass
    print(message, flush=True)


def _make_tqdm_bar(**kwargs: Any) -> Any | None:
    """Create one tqdm instance, falling back to plain tqdm when notebook widgets fail."""
    global tqdm
    if tqdm is None:
        return None
    try:
        return tqdm(**kwargs)
    except Exception:
        tqdm = _tqdm_plain
        if tqdm is not None:
            try:
                return tqdm(**kwargs)
            except Exception:
                return None
    return None


def _is_permission_listing_line(line: str) -> bool:
    """Return whether one line looks like `ls -l` file-listing noise."""
    text = str(line or "").strip()
    if len(text) < 10:
        return False
    return (
        text[:1] in {"d", "-", "l"}
        and all(char in "rwxstST-" for char in text[1:10])
        and text[10:11] == " "
    )


def _filter_live_remote_log_line(kind: str, line: str) -> str | None:
    """Return a cleaned live-log line, or None when the line is routine noise."""
    text = str(line or "").rstrip()
    stripped = text.strip()
    if not stripped:
        return None

    if kind == "stdout":
        if stripped.startswith("Sim ["):
            return None
        if stripped.startswith("numprocs="):
            return None
        if stripped.startswith("Rank Complexity "):
            return None
        if stripped in {"{", "}", "[", "]", "},", "],"}:
            return None
        if re.match(r'^"[^"]+":\s*[{[]?$', stripped):
            return None
        if re.match(r'^"[^"]+":\s*".*"[,\s]*$', stripped):
            return None
        if re.match(r'^"[^"]+":\s*-?\d+(\.\d+)?[,\s]*$', stripped):
            return None
        if re.match(r'^"[^"]+":\s*(true|false|null)[,\s]*$', stripped, re.IGNORECASE):
            return None
        return stripped

    if kind == "bootstrap":
        if stripped.startswith("Updating files:"):
            return None
        if stripped.startswith("HEAD is now at"):
            return None
        if stripped.startswith("Previous HEAD position was"):
            return None
        if stripped.startswith("total "):
            return None
        if _is_permission_listing_line(stripped):
            return None
        return stripped

    if kind == "stderr":
        if stripped.startswith("A requested component was not found"):
            return None
        if stripped.startswith("This means that this component is either not installed"):
            return None
        if stripped.startswith("means that this component is either not installed"):
            return None
        if stripped.startswith("used on your system"):
            return None
        if stripped.startswith("that the component requires are unable to be found/loaded"):
            return None
        if stripped.startswith("PMIx stopped checking at the first component"):
            return None
        if stripped.startswith("Host:"):
            return None
        if stripped.startswith("Framework: psec"):
            return None
        if stripped.startswith("Component: munge"):
            return None
        if stripped == "--------------------------------------------------------------------------":
            return None
        if stripped.startswith("NEURON -- VERSION"):
            return None
        if stripped.startswith("Duke, Yale, and the BlueBrain Project"):
            return None
        if stripped.startswith("See http://neuron.yale.edu/neuron/credits"):
            return None
        if stripped.startswith("Additional mechanisms from files"):
            return None
        if stripped.startswith('"prev_ob_models/') or stripped.startswith('" "prev_ob_models/'):
            return None
        return stripped

    return stripped


def _summarize_remote_status(status: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a compact JSON-safe remote status summary without duplicated tails."""
    if not status:
        return None
    return {
        "state": status.get("state"),
        "reason": status.get("reason"),
        "location": status.get("location"),
        "done": bool(status.get("done")),
        "ok": bool(status.get("ok")),
        "summary_exists": bool(status.get("summary_exists")),
        "stdout_exists": bool(status.get("stdout_exists")),
        "stderr_exists": bool(status.get("stderr_exists")),
        "bootstrap_exists": bool(status.get("bootstrap_exists")),
        "command_exists": bool(status.get("command_exists")),
        "slurm_log_exists": bool(status.get("slurm_log_exists")),
        "progress_percent": status.get("progress_percent"),
        "progress_current_ms": status.get("progress_current_ms"),
        "progress_total_ms": status.get("progress_total_ms"),
    }


def _summarize_remote_submit_response(submission: dict[str, Any]) -> dict[str, Any]:
    """Return a compact remote submission summary for run_info."""
    return {
        "job_id": submission.get("job_id"),
        "result_dir": submission.get("result_dir"),
        "wrapper_dir": submission.get("wrapper_dir"),
        "batch_script": submission.get("batch_script"),
        "worktree_path": submission.get("worktree_path"),
        "heartbeat_path": submission.get("heartbeat_path"),
        "heartbeat_timeout_s": submission.get("heartbeat_timeout_s"),
    }


def _compact_remote_poll_events(poll_transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compress raw remote polling samples into state changes and new log deltas."""
    events: list[dict[str, Any]] = []
    last_signature: tuple[Any, ...] | None = None
    last_tails = {"bootstrap": "", "stdout": "", "stderr": "", "slurm": ""}
    last_progress_bucket: int | None = None
    for status in poll_transcript:
        event: dict[str, Any] = {}
        signature = (
            status.get("state"),
            status.get("reason"),
            status.get("location"),
            bool(status.get("summary_exists")),
            bool(status.get("stdout_exists")),
            bool(status.get("stderr_exists")),
            bool(status.get("bootstrap_exists")),
            bool(status.get("command_exists")),
            bool(status.get("slurm_log_exists")),
            bool(status.get("done")),
            bool(status.get("ok")),
        )
        if signature != last_signature:
            event.update(_summarize_remote_status(status) or {})
            last_signature = signature

        progress_percent = status.get("progress_percent")
        if progress_percent not in (None, ""):
            progress_bucket = int(progress_percent) // 5
            if progress_bucket != last_progress_bucket or status.get("done"):
                event["progress_percent"] = int(progress_percent)
                event["progress_current_ms"] = status.get("progress_current_ms")
                event["progress_total_ms"] = status.get("progress_total_ms")
                last_progress_bucket = progress_bucket

        new_logs: dict[str, list[str]] = {}
        for kind in ("bootstrap", "stdout", "stderr", "slurm"):
            tail_text = str(status.get(f"{kind}_tail") or "")
            previous = last_tails[kind]
            if tail_text and tail_text != previous:
                delta_text = tail_text[len(previous):] if previous and tail_text.startswith(previous) else tail_text
                lines: list[str] = []
                for line in delta_text.replace("\r", "\n").splitlines():
                    cleaned = _filter_live_remote_log_line(kind, line)
                    if cleaned:
                        lines.append(cleaned)
                if lines:
                    new_logs[kind] = lines
            last_tails[kind] = tail_text
        if new_logs:
            event["new_logs"] = new_logs

        if event:
            events.append(event)
    return events


class _ProgressBar:
    """Small wrapper around tqdm with a plain-print fallback."""

    def __init__(
        self,
        *,
        total: int | None,
        desc: str,
        unit: str = "B",
        unit_scale: bool = False,
        display_step: int = 1,
    ):
        self.total = None if total is None else int(total)
        self.current = 0
        self.desc = desc
        self.unit = unit
        self.unit_scale = unit_scale
        self.display_step = max(int(display_step), 1)
        self._last_step = -1
        self._bar = None
        self._fallback_active = False
        self._display_current = 0
        self._bar = _make_tqdm_bar(
            total=max(self.total, 0) if self.total is not None else None,
            desc=desc,
            unit=unit,
            unit_scale=unit_scale,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.1,
        )

    def update_to(self, current: int) -> None:
        current = max(0, int(current))
        self.current = current
        should_render = (current - self._display_current) >= self.display_step
        if self.total is not None and current >= self.total:
            should_render = True
        if not should_render:
            return
        if self._bar is not None:
            delta = current - self._display_current
            try:
                if delta > 0:
                    self._bar.update(delta)
                self._display_current = current
                return
            except Exception:
                try:
                    self._bar.close()
                except Exception:
                    pass
                self._bar = None

        if self.total is None:
            step = self.display_step
            progress_step = self.current // step
            if progress_step == self._last_step:
                return
            self._last_step = progress_step
            self._fallback_active = True
            sys.stdout.write(
                "\r" + f"{self.desc} {_format_progress_value(self.current, self.unit, self.unit_scale)}"
            )
            sys.stdout.flush()
            self._display_current = current
            return
        if self.total <= 0:
            return
        progress_step = int((self.current * 100.0) / self.total) // 5
        if progress_step == self._last_step and self.current < self.total:
            return
        self._last_step = progress_step
        self._fallback_active = True
        sys.stdout.write(
            "\r"
            + f"{self.desc} {_render_progress_bar(self.current, self.total)} "
            + f"{_format_progress_value(self.current, self.unit, self.unit_scale)} / "
            + f"{_format_progress_value(self.total, self.unit, self.unit_scale)}"
        )
        sys.stdout.flush()
        self._display_current = current

    def tick(self, delta: int = 1) -> None:
        """Advance one indeterminate progress bar."""
        self.update_to(self.current + max(0, int(delta)))

    def close(self) -> None:
        if self._display_current < self.current:
            if self._bar is not None:
                delta = self.current - self._display_current
                try:
                    if delta > 0:
                        self._bar.update(delta)
                except Exception:
                    try:
                        self._bar.close()
                    except Exception:
                        pass
                    self._bar = None
            elif self._fallback_active:
                if self.total is None:
                    sys.stdout.write(
                        "\r" + f"{self.desc} {_format_progress_value(self.current, self.unit, self.unit_scale)}"
                    )
                else:
                    sys.stdout.write(
                        "\r"
                        + f"{self.desc} {_render_progress_bar(self.current, self.total)} "
                        + f"{_format_progress_value(self.current, self.unit, self.unit_scale)} / "
                        + f"{_format_progress_value(self.total, self.unit, self.unit_scale)}"
                    )
                sys.stdout.flush()
            self._display_current = self.current
        if self._bar is not None:
            try:
                self._bar.close()
            except Exception:
                pass
        elif self._fallback_active:
            sys.stdout.write("\r" + (" " * 120) + "\r")
            sys.stdout.flush()
            self._fallback_active = False


_LIVE_INSPECTION_MODEL = None
_LIVE_INSPECTION_SIGNATURE = None
if not hasattr(builtins, "_OBGPU_NOTEBOOK_RUNTIME"):
    builtins._OBGPU_NOTEBOOK_RUNTIME = {}
_NOTEBOOK_RUNTIME = builtins._OBGPU_NOTEBOOK_RUNTIME
_NOTEBOOK_RUNTIME.setdefault("ssh_masters", {})
_NOTEBOOK_RUNTIME.setdefault("paramiko_connections", {})
_NOTEBOOK_RUNTIME.setdefault("slurm_allocations", {})
_NOTEBOOK_RUNTIME.setdefault("remote_git_refs", {})
_NOTEBOOK_RUNTIME.setdefault("slurm_allocation_atexit_registered", False)
_LIVE_SSH_MASTERS: dict[str, Any] = _NOTEBOOK_RUNTIME["ssh_masters"]
_LIVE_PARAMIKO_CONNECTIONS: dict[str, Any] = _NOTEBOOK_RUNTIME["paramiko_connections"]
_LIVE_SLURM_ALLOCATIONS: dict[str, Any] = _NOTEBOOK_RUNTIME["slurm_allocations"]
_LIVE_REMOTE_GIT_REFS: dict[str, set[str]] = _NOTEBOOK_RUNTIME["remote_git_refs"]


def _slurm_allocation_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the SSH/runtime subset needed to rediscover or cancel one allocation."""
    keys = (
        "remote_host",
        "remote_results_root",
        "remote_heartbeat_timeout_s",
        "runner_backend",
        "ssh_binary",
        "ssh_options",
        "ssh_transport",
        "ssh_multiplex",
        "ssh_allow_interactive_auth",
        "ssh_keepalive_s",
        "ssh_control_path",
        "ssh_control_persist_s",
    )
    return {key: deepcopy(config.get(key)) for key in keys if key in config}


def _cleanup_notebook_remote_allocations() -> None:
    """Best-effort shutdown cleanup for notebook-managed reusable Slurm allocations."""
    allocations = list(_LIVE_SLURM_ALLOCATIONS.items())
    _LIVE_SLURM_ALLOCATIONS.clear()
    for _cache_key, allocation in allocations:
        if allocation.get("manual", False):
            continue
        job_id = allocation.get("job_id")
        runtime_config = allocation.get("config")
        if job_id in (None, "") or not isinstance(runtime_config, dict):
            continue
        try:
            _run_ssh_shell(runtime_config, _build_remote_cancel_command(job_id=str(job_id)))
        except Exception:
            continue


if not _NOTEBOOK_RUNTIME["slurm_allocation_atexit_registered"]:
    atexit.register(_cleanup_notebook_remote_allocations)
    _NOTEBOOK_RUNTIME["slurm_allocation_atexit_registered"] = True


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
        "use_corenrn": None,
        "use_gpu": None,
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
        "input_stimuli": None,
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
        "ketamine_block": None,
        "ampa_block": None,
        "gaba_gmax": None,
        "gaba_tau2_ms": None,
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
        "remote_poll_interval_s": 1.0,
        "remote_live_status": True,
        "remote_live_logs": True,
        "remote_heartbeat_timeout_s": 120,
        "remote_cleanup_stale_allocations": True,
        "slurm_partition": None,
        "slurm_account": None,
        "slurm_time": None,
        "slurm_gpus": None,
        "slurm_cpus_per_task": None,
        "slurm_mem": None,
        "slurm_extra_args": [],
        "ssh_binary": "ssh",
        "ssh_options": [],
        "ssh_transport": "auto",
        "ssh_multiplex": True,
        "ssh_allow_interactive_auth": True,
        "ssh_keepalive_s": 30,
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
    slurm_mem: str | None = None,
    remote_poll_interval_s: float = 1.0,
    remote_live_status: bool = True,
    remote_live_logs: bool = True,
    remote_heartbeat_timeout_s: int = 120,
    remote_cleanup_stale_allocations: bool = True,
    remote_repo_mode: str = "shared",
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = False,
    remote_git_remote: str = "origin",
    slurm_allocation_job_id: str | None = None,
    slurm_reuse_allocation: bool = False,
    slurm_allocation_time: str | None = None,
    slurm_allocation_name: str | None = None,
    ssh_options: list[str] | None = None,
    rsync_options: list[str] | None = None,
    slurm_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Return a generic remote Slurm config for notebook-driven runs.

    Slurm arguments are only emitted when explicitly provided.
    """
    _warn_remote_execution_mode_reset()
    remote_repo_root = str(remote_repo_root)
    if remote_results_root is None:
        remote_results_root = str(PurePosixPath(remote_repo_root) / "results" / "notebook_runs")

    config = {
        "runner_backend": "slurm_remote",
        "use_corenrn": None,
        "use_gpu": None,
        "remote_host": str(remote_host),
        "remote_repo_root": remote_repo_root,
        "remote_results_root": str(remote_results_root),
        "remote_conda_activate_cmd": str(remote_conda_activate_cmd),
        "remote_runtime_profiles": list(remote_runtime_profiles or []),
        "remote_fallback_conda_activate_cmd": None
        if remote_fallback_conda_activate_cmd in (None, "")
        else str(remote_fallback_conda_activate_cmd),
        "remote_fast_node_feature": None if remote_fast_node_feature in (None, "") else str(remote_fast_node_feature),
        "remote_mechanism_profile": str(remote_mechanism_profile or "default"),
        "remote_fallback_mechanism_profile": str(remote_fallback_mechanism_profile or "portable"),
        "remote_mpi_exec": str(remote_mpi_exec or default_remote_mpi_exec()),
        "remote_poll_interval_s": float(remote_poll_interval_s),
        "remote_live_status": bool(remote_live_status),
        "remote_live_logs": bool(remote_live_logs),
        "remote_heartbeat_timeout_s": int(remote_heartbeat_timeout_s),
        "remote_cleanup_stale_allocations": bool(remote_cleanup_stale_allocations),
        "remote_sync_compress": True,
        "disable_status_report": False,
        "remote_repo_mode": str(remote_repo_mode),
        "remote_git_ref": remote_git_ref,
        "remote_git_fetch": bool(remote_git_fetch),
        "remote_git_remote": str(remote_git_remote),
        "slurm_allocation_job_id": None if slurm_allocation_job_id in (None, "") else str(slurm_allocation_job_id),
        "slurm_reuse_allocation": bool(slurm_reuse_allocation),
        "slurm_allocation_time": None if slurm_allocation_time in (None, "") else str(slurm_allocation_time),
        "slurm_allocation_name": None if slurm_allocation_name in (None, "") else str(slurm_allocation_name),
        "slurm_partition": None if slurm_partition in (None, "") else str(slurm_partition),
        "slurm_account": None if slurm_account in (None, "") else str(slurm_account),
        "slurm_time": None if slurm_time in (None, "") else str(slurm_time),
        "slurm_gpus": None if slurm_gpus in (None, "") else int(slurm_gpus),
        "slurm_cpus_per_task": None if slurm_cpus_per_task in (None, "") else int(slurm_cpus_per_task),
        "slurm_mem": None if slurm_mem in (None, "") else str(slurm_mem),
        "slurm_extra_args": list(slurm_extra_args or []),
        "ssh_options": list(ssh_options or []),
        "ssh_transport": "auto",
        "ssh_keepalive_s": 30,
        "rsync_options": list(rsync_options or ["-az"]),
    }
    return config


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
    slurm_mem: str | None = None,
    remote_poll_interval_s: float = 1.0,
    remote_live_status: bool = True,
    remote_live_logs: bool = True,
    remote_heartbeat_timeout_s: int = 120,
    remote_cleanup_stale_allocations: bool = True,
    remote_repo_mode: str = "shared",
    remote_git_ref: str | None = None,
    remote_git_fetch: bool = False,
    remote_git_remote: str = "origin",
    slurm_allocation_job_id: str | None = None,
    slurm_reuse_allocation: bool = False,
    slurm_allocation_time: str | None = None,
    slurm_allocation_name: str | None = None,
    ssh_options: list[str] | None = None,
    rsync_options: list[str] | None = None,
    slurm_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Return a Sol-specific remote runner config with Sol activation defaults.

    Slurm arguments are only emitted when explicitly provided.
    """
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
        slurm_mem=slurm_mem,
        remote_poll_interval_s=remote_poll_interval_s,
        remote_live_status=remote_live_status,
        remote_live_logs=remote_live_logs,
        remote_heartbeat_timeout_s=remote_heartbeat_timeout_s,
        remote_cleanup_stale_allocations=remote_cleanup_stale_allocations,
        remote_repo_mode=remote_repo_mode,
        remote_git_ref=remote_git_ref,
        remote_git_fetch=remote_git_fetch,
        remote_git_remote=remote_git_remote,
        slurm_allocation_job_id=slurm_allocation_job_id,
        slurm_reuse_allocation=slurm_reuse_allocation,
        slurm_allocation_time=slurm_allocation_time,
        slurm_allocation_name=slurm_allocation_name,
        ssh_options=ssh_options,
        rsync_options=rsync_options,
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
    """Return ordered runtime profiles for Sol's Grace Hopper, ARM, and x86 nodes.

    The remote batch script selects the first profile whose node-info predicates
    match every allocated node. Mechanism profiles keep same-architecture builds
    separate when Sol has more than one CPU/GPU target under one repo checkout.
    The conda environment defaults to the shared Sol OBGPU env that existing
    notebook runs use; callers can still pass architecture-specific env names.
    """
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


def _warn_remote_execution_mode_reset() -> None:
    """Warn that remote configs clear local acceleration toggles and infer mode from Slurm."""
    warnings.warn(
        "Remote Slurm configs reset use_corenrn/use_gpu to auto. "
        "If you apply them via RUN_CONFIG.update(...), any previous local values for those keys "
        "will be cleared. Remote execution mode will then be inferred from slurm_gpus unless you "
        "explicitly set use_corenrn/use_gpu again after applying the remote config.",
        stacklevel=2,
    )


def _resolve_execution_mode(config: dict[str, Any]) -> dict[str, Any]:
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
    if config.get("input_stimuli") is not None:
        from olfactorybulb.inputs import serialize_input_stimuli
        raw = config["input_stimuli"]
        # Normalize onset-time keys (JSON round-trips string keys)
        normalized = {}
        for k, v in raw.items():
            try:
                nk = int(float(k)) if float(k).is_integer() else float(k)
            except (TypeError, ValueError):
                nk = k
            normalized[nk] = v
        json_safe, dill_blob = serialize_input_stimuli(normalized)
        if dill_blob is not None:
            # Callable specs are written to a temp file; the path is stored in
            # the config so build_run_command can pass --input-spec-file.
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".inputspec.dill", prefix="ob_"
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
        for key in (
            "ampa_nmda_gmax",
            "ampa_nmda_nmdafactor",
            "ketamine_block",
            "ampa_block",
            "gaba_gmax",
            "gaba_tau2_ms",
        )
    ):
        overrides.setdefault("synapse_properties", {})
    if any(
        config.get(key) is not None
        for key in ("ampa_nmda_gmax", "ampa_nmda_nmdafactor", "ketamine_block", "ampa_block")
    ):
        overrides["synapse_properties"].setdefault("AmpaNmdaSyn", {})
        if config.get("ampa_nmda_gmax") is not None:
            overrides["synapse_properties"]["AmpaNmdaSyn"]["gmax"] = float(config["ampa_nmda_gmax"])
        if config.get("ampa_nmda_nmdafactor") is not None:
            overrides["synapse_properties"]["AmpaNmdaSyn"]["nmdafactor"] = float(
                config["ampa_nmda_nmdafactor"]
            )
        if config.get("ketamine_block") is not None:
            overrides["synapse_properties"]["AmpaNmdaSyn"]["ketamine_block"] = float(
                config["ketamine_block"]
            )
        if config.get("ampa_block") is not None:
            overrides["synapse_properties"]["AmpaNmdaSyn"]["ampa_block"] = float(
                config["ampa_block"]
            )
    if config.get("gaba_gmax") is not None or config.get("gaba_tau2_ms") is not None:
        overrides["synapse_properties"].setdefault("GabaSyn", {})
        if config.get("gaba_gmax") is not None:
            overrides["synapse_properties"]["GabaSyn"]["gmax"] = float(config["gaba_gmax"])
        if config.get("gaba_tau2_ms") is not None:
            overrides["synapse_properties"]["GabaSyn"]["tau2"] = float(config["gaba_tau2_ms"])
    scalar_param_map = {
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
    for config_key, param_key in scalar_param_map.items():
        if config.get(config_key) is not None:
            overrides[param_key] = float(config[config_key])
    if config.get("enable_gc_kar") is not None:
        overrides["enable_gc_kar"] = bool(config["enable_gc_kar"])
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
    include_mpi_launcher: bool = True,
) -> list[str]:
    """Build the benchmark subprocess command for a notebook run."""
    repo_root = repo_root or REPO_ROOT
    results_base = results_base or config.get("results_base", DEFAULT_RESULTS_BASE)
    benchmark_script = Path(repo_root) / "tools" / "benchmarks" / "benchmark_ob.py"
    execution_mode = _resolve_execution_mode(config)
    command: list[str] = []
    if include_mpi_launcher:
        mpi_exec = mpi_exec or str(config.get("mpi_exec", default_local_mpi_exec()))
        command.extend(
            [
                *shlex.split(mpi_exec),
                "-n",
                str(int(config["nranks"])),
            ]
        )

    param_overrides = build_param_overrides(config)
    input_spec_file = param_overrides.pop("_input_spec_file", None)
    command.extend(
        [
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
            json.dumps(param_overrides, sort_keys=True),
        ]
    )
    if input_spec_file is not None:
        command.extend(["--input-spec-file", str(input_spec_file)])

    if config.get("tstop_ms") is not None:
        command.extend(["--tstop-override", str(float(config["tstop_ms"]))])

    if execution_mode["use_corenrn"]:
        command.append("--coreneuron")
    if execution_mode["use_gpu"]:
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
        raise ValueError("remote Slurm runner requires remote_repo_root")
    return PurePosixPath(str(remote_repo_root))


def _remote_results_root(config: dict[str, Any]) -> PurePosixPath:
    """Return the configured results root on the remote Sol host."""
    configured = config.get("remote_results_root")
    if configured:
        return PurePosixPath(str(configured))
    return _remote_repo_root(config) / "results" / "notebook_runs"


_REMOTE_SLURM_TERMINAL_OK = {"COMPLETED"}
_REMOTE_SLURM_TERMINAL_FAIL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED_WITH_ERRORS",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


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


def _resolve_local_git_branch() -> str | None:
    """Return the current local branch name or ``None`` when detached."""
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    branch = (completed.stdout or "").strip()
    return branch or None


def _resolve_local_git_upstream_ref() -> str | None:
    """Return the current branch upstream ref, or ``None`` when unavailable."""
    completed = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    upstream = (completed.stdout or "").strip()
    return upstream or None


def _git_rev_parse(ref_name: str) -> str | None:
    """Resolve one local git ref to a commit SHA."""
    completed = subprocess.run(
        ["git", "rev-parse", ref_name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    sha = (completed.stdout or "").strip()
    return sha or None


def _git_ref_points_to_commit(ref_name: str, commit_sha: str) -> bool:
    """Return whether one local git ref currently resolves to the requested commit."""
    return _git_rev_parse(ref_name) == commit_sha


def _git_ref_is_ancestor(ancestor_ref: str, descendant_ref: str) -> bool:
    """Return whether one git ref is an ancestor of another."""
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_ref, descendant_ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _local_git_sync_base_candidates(commit_sha: str, *, max_count: int = 500) -> list[str]:
    """Return local ancestor SHAs to test as possible remote bundle bases."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(ref_name: str | None) -> None:
        if not ref_name:
            return
        sha = _git_rev_parse(ref_name)
        if not sha or sha == commit_sha or sha in seen:
            return
        if not _git_ref_is_ancestor(sha, commit_sha):
            return
        seen.add(sha)
        candidates.append(sha)

    completed = subprocess.run(
        ["git", "rev-list", "--first-parent", f"--max-count={int(max_count)}", f"{commit_sha}^"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        for line in (completed.stdout or "").splitlines():
            add_candidate(line.strip())

    add_candidate(_resolve_local_git_upstream_ref())
    return candidates


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
        raise ValueError("remote Slurm runner requires remote_host")
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


def _remote_transport(config: dict[str, Any]) -> str:
    """Return the active transport implementation for the Sol backend."""
    configured = str(config.get("ssh_transport", "auto")).strip().lower()
    if configured not in {"auto", "paramiko", "openssh"}:
        raise ValueError(f"Unsupported ssh_transport={configured!r}")
    if configured == "auto":
        if paramiko is not None:
            return "paramiko"
        return "openssh"
    if configured == "paramiko" and paramiko is None:
        raise RuntimeError(
            "ssh_transport='paramiko' requires the optional 'paramiko' dependency "
            "in the notebook environment."
        )
    return configured


def _remote_endpoint(config: dict[str, Any]) -> tuple[str, int, str]:
    """Resolve hostname, port, and username from the remote config."""
    host = _require_remote_host(config)
    if "@" in host:
        username, hostname = host.split("@", 1)
    else:
        username = os.environ.get("USER") or os.environ.get("USERNAME") or ""
        hostname = host
    if not username:
        raise ValueError(f"Could not infer SSH username from remote_host={host!r}")

    port = 22
    options = [str(option) for option in config.get("ssh_options", [])]
    index = 0
    while index < len(options):
        option = options[index]
        if option == "-p" and index + 1 < len(options):
            port = int(options[index + 1])
            index += 2
            continue
        if option.startswith("-p") and option != "-p":
            port = int(option[2:])
            index += 1
            continue
        if option == "-o" and index + 1 < len(options):
            key_value = options[index + 1]
            if key_value.lower().startswith("port="):
                port = int(key_value.split("=", 1)[1])
            index += 2
            continue
        index += 1
    return hostname, port, username


def _paramiko_connection_key(config: dict[str, Any]) -> str:
    """Build the cache key for one persistent Paramiko connection."""
    hostname, port, username = _remote_endpoint(config)
    return f"{username}@{hostname}:{port}"


def _remote_git_ref_cache_key(config: dict[str, Any], remote_repo_root: PurePosixPath) -> str:
    """Build the runtime cache key for remote git-object presence checks."""
    return f"{_paramiko_connection_key(config)}::{remote_repo_root.as_posix()}"


def _build_remote_git_repo_probe_command(remote_repo_root: PurePosixPath) -> str:
    """Build a remote shell command that verifies the configured repo exists."""
    repo_root = remote_repo_root.as_posix()
    quoted_repo = shlex.quote(repo_root)
    missing_message = shlex.quote(f"remote_repo_root does not exist: {repo_root}")
    not_git_message = shlex.quote(f"remote_repo_root is not a git work tree: {repo_root}")
    return (
        f"if ! test -d {quoted_repo}; then printf '%s\\n' {missing_message} >&2; exit 2; fi; "
        f"if ! git -C {quoted_repo} rev-parse --is-inside-work-tree >/dev/null 2>&1; "
        f"then printf '%s\\n' {not_git_message} >&2; exit 3; fi"
    )


def _normalize_slurm_state(raw_state: str) -> str:
    """Normalize Slurm state tokens by removing suffixes such as '+'."""
    return raw_state.split()[0].split("+", 1)[0].strip().upper()


def _remote_heartbeat_timeout_s(config: dict[str, Any]) -> int:
    """Return the notebook heartbeat timeout used by remote Slurm watchdogs."""
    value = config.get("remote_heartbeat_timeout_s", 120)
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return 120


def _slurm_allocation_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return the cache signature for one reusable remote Slurm allocation."""
    hostname, port, username = _remote_endpoint(config)
    return {
        "remote_host": f"{username}@{hostname}:{port}",
        "remote_results_root": _remote_results_root(config).as_posix(),
        "partition": None if config.get("slurm_partition") in (None, "") else str(config.get("slurm_partition")),
        "account": None if config.get("slurm_account") in (None, "") else str(config.get("slurm_account")),
        "time": str(config.get("slurm_allocation_time") or config.get("slurm_time") or ""),
        "gpus": None if config.get("slurm_gpus") in (None, "") else int(config.get("slurm_gpus")),
        "cpus_per_task": None if config.get("slurm_cpus_per_task") in (None, "") else int(config.get("slurm_cpus_per_task")),
        "mem": None if config.get("slurm_mem") in (None, "") else str(config.get("slurm_mem")),
        "extra_args": [str(arg) for arg in config.get("slurm_extra_args", [])],
        "remote_conda_activate_cmd": str(config.get("remote_conda_activate_cmd") or ""),
        "remote_runtime_profiles": _json_ready(config.get("remote_runtime_profiles") or []),
        "remote_fallback_conda_activate_cmd": str(config.get("remote_fallback_conda_activate_cmd") or ""),
        "remote_fast_node_feature": str(config.get("remote_fast_node_feature") or ""),
        "remote_mechanism_profile": str(config.get("remote_mechanism_profile") or "default"),
        "remote_fallback_mechanism_profile": str(config.get("remote_fallback_mechanism_profile") or "portable"),
        "name": str(config.get("slurm_allocation_name") or "obgpu_notebook_alloc"),
    }


def _slurm_allocation_cache_key(config: dict[str, Any]) -> str:
    """Return the runtime cache key for one reusable remote Slurm allocation."""
    payload = json.dumps(_slurm_allocation_signature(config), sort_keys=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def _paramiko_prompt_response(prompt_text: str) -> str:
    """Prompt the notebook user for one interactive SSH auth field."""
    prompt = prompt_text.strip() or "SSH authentication:"
    lowered = prompt.lower()
    if "password" in lowered or "passphrase" in lowered:
        return getpass(prompt + " ")
    return input(prompt + " ")


def _drop_paramiko_connection(config: dict[str, Any]) -> None:
    """Close and forget one cached Paramiko connection."""
    cached = _LIVE_PARAMIKO_CONNECTIONS.pop(_paramiko_connection_key(config), None)
    if cached is None:
        return
    sftp = cached.get("sftp")
    if sftp is not None:
        try:
            sftp.close()
        except Exception:
            pass
    transport = cached.get("transport")
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


def _connect_paramiko(config: dict[str, Any]) -> Any:
    """Open or reuse one persistent Paramiko transport for the Sol backend."""
    if paramiko is None:
        raise RuntimeError("Paramiko transport requested but the 'paramiko' package is not installed.")

    cache_key = _paramiko_connection_key(config)
    cached = _LIVE_PARAMIKO_CONNECTIONS.get(cache_key)
    if cached is not None:
        transport = cached.get("transport")
        if transport is not None and transport.is_active() and transport.is_authenticated():
            return cached
        _LIVE_PARAMIKO_CONNECTIONS.pop(cache_key, None)

    hostname, port, username = _remote_endpoint(config)
    raw_sock = None
    transport = None
    try:
        import socket

        _progress_write(f"[Sol remote] Opening SSH session to {username}@{hostname}:{port}...")
        raw_sock = socket.create_connection((hostname, port), timeout=30.0)
        transport = paramiko.Transport(raw_sock)
        transport.start_client(timeout=30.0)
        keepalive_seconds = int(config.get("ssh_keepalive_s", 30) or 0)
        if keepalive_seconds > 0:
            transport.set_keepalive(keepalive_seconds)

        auth_methods: list[str] = []
        try:
            transport.auth_none(username)
        except paramiko.BadAuthenticationType as exc:
            auth_methods = list(exc.allowed_types)
        except _PARAMIKO_PARTIAL_AUTH_EXC as exc:  # pragma: no cover - defensive
            auth_methods = list(exc.allowed_types)
        except paramiko.AuthenticationException:
            auth_methods = []

        authenticated = False
        if "keyboard-interactive" in auth_methods or not auth_methods:
            _progress_write(f"[Sol remote] Waiting for interactive SSH authentication...")
            def handler(title: str, instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
                responses: list[str] = []
                if title:
                    print(title)
                if instructions:
                    print(instructions)
                for prompt_text, _echo in prompt_list:
                    responses.append(_paramiko_prompt_response(prompt_text))
                return responses

            try:
                transport.auth_interactive(username, handler)
                authenticated = transport.is_authenticated()
            except paramiko.AuthenticationException:
                authenticated = False

        if not authenticated and "password" in auth_methods:
            try:
                _progress_write(f"[Sol remote] Waiting for password authentication...")
                transport.auth_password(
                    username,
                    _paramiko_prompt_response(f"Password for {username}@{hostname}:"),
                )
                authenticated = transport.is_authenticated()
            except _PARAMIKO_PARTIAL_AUTH_EXC as exc:
                auth_methods = list(exc.allowed_types)
                authenticated = False

        if not authenticated and "keyboard-interactive" in auth_methods:
            _progress_write(f"[Sol remote] Waiting for interactive SSH authentication...")
            def handler(title: str, instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
                responses: list[str] = []
                if title:
                    print(title)
                if instructions:
                    print(instructions)
                for prompt_text, _echo in prompt_list:
                    responses.append(_paramiko_prompt_response(prompt_text))
                return responses

            transport.auth_interactive(username, handler)
            authenticated = transport.is_authenticated()

        if not authenticated:
            raise RuntimeError(
                "Paramiko could not authenticate to the Sol backend.\n"
                f"Host: {username}@{hostname}:{port}\n"
                f"Auth methods: {auth_methods}"
            )

        _progress_write(f"[Sol remote] SSH authentication complete; opening SFTP channel...")
        connection = {
            "transport": transport,
            "sftp": paramiko.SFTPClient.from_transport(transport),
            "hostname": hostname,
            "port": port,
            "username": username,
        }
        _LIVE_PARAMIKO_CONNECTIONS[cache_key] = connection
        _progress_write(f"[Sol remote] SSH session ready for {username}@{hostname}:{port}.")
        return connection
    except Exception:
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        if raw_sock is not None:
            try:
                raw_sock.close()
            except Exception:
                pass
        raise


def _run_paramiko_shell(
    config: dict[str, Any],
    remote_shell_command: str,
) -> subprocess.CompletedProcess[str]:
    """Run one shell command over a persistent Paramiko transport."""
    last_exc: Exception | None = None
    for attempt in range(2):
        connection = _connect_paramiko(config)
        transport = connection["transport"]
        channel = None
        try:
            channel = transport.open_session()
            channel.exec_command(f"bash -lc {shlex.quote(remote_shell_command)}")
            stdout_data = channel.makefile("rb").read().decode("utf-8", errors="replace")
            stderr_data = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace")
            return subprocess.CompletedProcess(
                args=["paramiko", connection["hostname"], remote_shell_command],
                returncode=channel.recv_exit_status(),
                stdout=stdout_data,
                stderr=stderr_data,
            )
        except Exception as exc:
            last_exc = exc
            _drop_paramiko_connection(config)
            if attempt == 0:
                continue
            raise
        finally:
            if channel is not None:
                channel.close()
    raise RuntimeError(f"Paramiko shell command failed unexpectedly: {last_exc}")


def _sftp_copy_tree(sftp: Any, remote_dir: str, local_dir: Path) -> None:
    """Recursively copy one remote directory tree through SFTP with progress output."""

    def collect_files(current_remote_dir: str, current_local_dir: Path) -> list[tuple[str, Path, int]]:
        current_local_dir.mkdir(parents=True, exist_ok=True)
        files: list[tuple[str, Path, int]] = []
        for entry in sftp.listdir_attr(current_remote_dir):
            remote_path = f"{current_remote_dir.rstrip('/')}/{entry.filename}"
            local_path = current_local_dir / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                files.extend(collect_files(remote_path, local_path))
                continue
            files.append((remote_path, local_path, int(getattr(entry, "st_size", 0))))
        return files

    transfer_plan = collect_files(remote_dir, local_dir)
    total_files = len(transfer_plan)
    total_bytes = sum(size for _remote_path, _local_path, size in transfer_plan)
    transferred_bytes = 0
    progress = _ProgressBar(total=total_bytes, desc="[OBGPU load] Sync from Sol", unit="B", unit_scale=True)

    if total_files:
        _progress_write(
            f"[OBGPU load] Syncing {total_files} files from Sol ({_format_bytes(total_bytes)})...",
        )

    for index, (remote_path, local_path, file_size) in enumerate(transfer_plan, start=1):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _progress_write(
            f"[OBGPU load] Syncing {index}/{total_files}: {local_path.name} ({_format_bytes(file_size)})",
        )
        base_bytes = transferred_bytes

        def callback(current_file_bytes: int, _current_file_total: int) -> None:
            overall_bytes = base_bytes + current_file_bytes
            progress.update_to(overall_bytes)

        sftp.get(remote_path, str(local_path), callback=callback)
        transferred_bytes += file_size
        progress.update_to(transferred_bytes)

    if total_files:
        _progress_write(
            f"[OBGPU load] Sync complete {_render_progress_bar(total_bytes, total_bytes)} "
            f"{_format_bytes(total_bytes)} / {_format_bytes(total_bytes)}",
        )
    progress.close()


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
    if _remote_transport(config) == "paramiko":
        completed = _run_paramiko_shell(config, remote_shell_command)
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed

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


def _build_remote_archive_command(remote_result_dir: PurePosixPath) -> str:
    """Build one remote shell command that packs the result dir into a compressed tar archive."""
    archive_dir = PurePosixPath(remote_result_dir.parent) / ".obgpu-transfer"
    archive_base = archive_dir / remote_result_dir.name
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        f"archive_dir={shlex.quote(archive_dir.as_posix())} && "
        f"archive_base={shlex.quote(archive_base.as_posix())} && "
        "mkdir -p \"$archive_dir\" && "
        "rm -f \"${archive_base}.tar.zst\" \"${archive_base}.tar.gz\" \"${archive_base}.tar.xz\" && "
        "raw_bytes=$(du -sb \"$result_dir\" 2>/dev/null | awk '{print $1}') && "
        "if command -v zstd >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.zst\"; "
        "  compressor=zstd; "
        "  tar -C \"$result_dir\" -cf - . | zstd -T0 -15 -q -o \"$archive_path\"; "
        "elif command -v pigz >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.gz\"; "
        "  compressor=pigz; "
        "  tar -C \"$result_dir\" -cf - . | pigz -6 > \"$archive_path\"; "
        "elif command -v gzip >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.gz\"; "
        "  compressor=gzip; "
        "  tar -C \"$result_dir\" -cf - . | gzip -6 > \"$archive_path\"; "
        "elif command -v xz >/dev/null 2>&1; then "
        "  archive_path=\"${archive_base}.tar.xz\"; "
        "  compressor=xz; "
        "  tar -C \"$result_dir\" -cf - . | xz -6 -T0 > \"$archive_path\"; "
        "else "
        "  printf '%s\\n' 'No supported compressor found on remote host' >&2; "
        "  exit 1; "
        "fi && "
        "archive_bytes=$(wc -c < \"$archive_path\") && "
        "printf '%s\\n%s\\n%s\\n%s\\n' \"$archive_path\" \"$compressor\" \"${raw_bytes:-0}\" \"$archive_bytes\""
    )


def _build_remote_archive_probe_command(remote_result_dir: PurePosixPath) -> str:
    """Build one remote shell command that selects a compressor and reports stream metadata."""
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        "raw_bytes=$(du -sb \"$result_dir\" 2>/dev/null | awk '{print $1}') && "
        "if command -v zstd >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'zstd' \"${raw_bytes:-0}\" '.tar.zst'; "
        "elif command -v pigz >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'pigz' \"${raw_bytes:-0}\" '.tar.gz'; "
        "elif command -v gzip >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'gzip' \"${raw_bytes:-0}\" '.tar.gz'; "
        "elif command -v xz >/dev/null 2>&1; then "
        "  printf '%s\\n%s\\n%s\\n' 'xz' \"${raw_bytes:-0}\" '.tar.xz'; "
        "else "
        "  printf '%s\\n' 'No supported compressor found on remote host' >&2; "
        "  exit 1; "
        "fi"
    )


def _build_remote_stream_archive_command(
    remote_result_dir: PurePosixPath,
    *,
    compressor: str,
) -> str:
    """Build one remote shell command that streams a compressed tar archive to stdout."""
    compressor = str(compressor)
    compressor_commands = {
        "zstd": 'tar -C "$result_dir" -cf - . | zstd -T0 -15 -q -c',
        "pigz": 'tar -C "$result_dir" -cf - . | pigz -6',
        "gzip": 'tar -C "$result_dir" -cf - . | gzip -6',
        "xz": 'tar -C "$result_dir" -cf - . | xz -6 -T0',
    }
    if compressor not in compressor_commands:
        raise ValueError(f"Unsupported archive compressor {compressor!r}")
    return (
        "set -euo pipefail && "
        f"result_dir={shlex.quote(remote_result_dir.as_posix())} && "
        + compressor_commands[compressor]
    )


def _remove_remote_file(config: dict[str, Any], remote_path: str) -> None:
    """Best-effort remote file removal used for temporary sync archives."""
    remote_shell = "rm -f {}".format(shlex.quote(remote_path))
    if _remote_transport(config) == "paramiko":
        try:
            _run_paramiko_shell(config, remote_shell)
        except Exception:
            pass
        return
    try:
        _run_ssh_shell(config, remote_shell)
    except Exception:
        pass


def _extract_local_archive(local_archive_path: Path, local_result_dir: Path) -> subprocess.CompletedProcess[str]:
    """Extract one downloaded result archive into the local result directory."""
    local_result_dir.mkdir(parents=True, exist_ok=True)
    suffixes = local_archive_path.suffixes
    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-2:] == [".tar", ".xz"]:
        import tarfile

        mode = "r:gz" if suffixes[-1] == ".gz" else "r:xz"
        try:
            with tarfile.open(local_archive_path, mode) as handle:
                handle.extractall(local_result_dir)
        except Exception as exc:
            return subprocess.CompletedProcess(
                args=["tarfile", str(local_archive_path), str(local_result_dir)],
                returncode=1,
                stdout="",
                stderr=str(exc),
            )
        return subprocess.CompletedProcess(
            args=["tarfile", str(local_archive_path), str(local_result_dir)],
            returncode=0,
            stdout="",
            stderr="",
        )

    if suffixes[-2:] == [".tar", ".zst"]:
        completed = subprocess.run(
            [
                "tar",
                "--use-compress-program=zstd -d -q",
                "-xf",
                str(local_archive_path),
                "-C",
                str(local_result_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed

    return subprocess.CompletedProcess(
        args=["extract", str(local_archive_path), str(local_result_dir)],
        returncode=1,
        stdout="",
        stderr=f"Unsupported archive format for {local_archive_path.name}",
    )


def _local_archive_decompress_command(compressor: str) -> list[str]:
    """Return a local decompressor command for one archive stream."""
    compressor = str(compressor)
    if compressor == "zstd":
        return [str(shutil.which("zstd") or "zstd"), "-d", "-q"]
    if compressor in {"pigz", "gzip"}:
        return [str(shutil.which("gzip") or "gzip"), "-d"]
    if compressor == "xz":
        return [str(shutil.which("xz") or "xz"), "-d"]
    raise ValueError(f"Unsupported archive compressor {compressor!r}")


def _stream_paramiko_archive_to_local(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_archive_path: Path,
    compressor: str,
    raw_bytes: int,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote compressed tar archive over Paramiko into a local file."""
    connection = _connect_paramiko(config)
    transport = connection["transport"]
    channel = None
    stderr_chunks: list[bytes] = []
    bytes_written = 0
    completed_ok = False
    progress = _ProgressBar(
        total=None,
        desc="[OBGPU load] Download compressed stream",
        unit="B",
        unit_scale=True,
        display_step=10 * 1024 * 1024,
    )
    stream_command = _build_remote_stream_archive_command(
        remote_result_dir,
        compressor=compressor,
    )
    local_archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive_path = local_archive_path.with_suffix(local_archive_path.suffix + ".part")
    try:
        channel = transport.open_session()
        channel.exec_command(f"bash -lc {shlex.quote(stream_command)}")
        with open(tmp_archive_path, "wb") as handle:
            while True:
                if channel.recv_ready():
                    data = channel.recv(1024 * 1024)
                    if data:
                        handle.write(data)
                        bytes_written += len(data)
                        progress.update_to(bytes_written)
                if channel.recv_stderr_ready():
                    stderr_chunks.append(channel.recv_stderr(65536))
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    break
                if not channel.recv_ready() and not channel.recv_stderr_ready():
                    time.sleep(0.05)
        returncode = channel.recv_exit_status()
        if bytes_written:
            progress.update_to(bytes_written)
        progress.close()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        if returncode == 0:
            tmp_archive_path.replace(local_archive_path)
            completed_ok = True
        else:
            try:
                tmp_archive_path.unlink(missing_ok=True)
            except Exception:
                pass
        return subprocess.CompletedProcess(
            args=["paramiko-stream", remote_result_dir.as_posix(), str(local_archive_path)],
            returncode=returncode,
            stdout="",
            stderr=stderr_text,
        )
    finally:
        progress.close()
        if not completed_ok:
            try:
                tmp_archive_path.unlink(missing_ok=True)
            except Exception:
                pass
        if channel is not None:
            channel.close()


def _stream_paramiko_archive_to_local_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    compressor: str,
    raw_bytes: int,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote compressed tar archive over Paramiko directly into local extraction."""
    connection = _connect_paramiko(config)
    transport = connection["transport"]
    channel = None
    stderr_chunks: list[bytes] = []
    bytes_written = 0
    progress = _ProgressBar(
        total=None,
        desc="[OBGPU load] Stream download/extract",
        unit="B",
        unit_scale=True,
        display_step=10 * 1024 * 1024,
    )
    local_result_dir.mkdir(parents=True, exist_ok=True)
    stream_command = _build_remote_stream_archive_command(
        remote_result_dir,
        compressor=compressor,
    )
    decompress_cmd = _local_archive_decompress_command(compressor)
    decompress_stderr = tempfile.NamedTemporaryFile(prefix="obgpu-decompress-", suffix=".log", delete=False)
    tar_stderr = tempfile.NamedTemporaryFile(prefix="obgpu-tar-", suffix=".log", delete=False)
    decompress_proc = None
    tar_proc = None
    decompress_stderr_handle = None
    tar_stderr_handle = None
    try:
        decompress_stderr.close()
        tar_stderr.close()
        decompress_stderr_handle = open(decompress_stderr.name, "wb")
        tar_stderr_handle = open(tar_stderr.name, "wb")
        decompress_proc = subprocess.Popen(
            decompress_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=decompress_stderr_handle,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-xf", "-", "-C", str(local_result_dir)],
            stdin=decompress_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=tar_stderr_handle,
        )
        if decompress_proc.stdout is not None:
            decompress_proc.stdout.close()
        if decompress_proc.stdin is None:
            raise RuntimeError("Could not open decompressor stdin for streaming extraction.")

        channel = transport.open_session()
        channel.exec_command(f"bash -lc {shlex.quote(stream_command)}")
        while True:
            if channel.recv_ready():
                data = channel.recv(1024 * 1024)
                if data:
                    decompress_proc.stdin.write(data)
                    bytes_written += len(data)
                    progress.update_to(bytes_written)
                    continue
            if channel.recv_stderr_ready():
                stderr_chunks.append(channel.recv_stderr(65536))
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            time.sleep(0.05)

        if decompress_proc.stdin is not None:
            decompress_proc.stdin.close()
        remote_returncode = channel.recv_exit_status()
        decompress_returncode = decompress_proc.wait()
        tar_returncode = tar_proc.wait()
        progress.close()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        if Path(decompress_stderr.name).exists():
            stderr_text += Path(decompress_stderr.name).read_text(errors="replace")
        if Path(tar_stderr.name).exists():
            stderr_text += Path(tar_stderr.name).read_text(errors="replace")
        returncode = 0 if remote_returncode == 0 and decompress_returncode == 0 and tar_returncode == 0 else 1
        return subprocess.CompletedProcess(
            args=["paramiko-stream-extract", remote_result_dir.as_posix(), str(local_result_dir)],
            returncode=returncode,
            stdout="",
            stderr=stderr_text,
        )
    finally:
        progress.close()
        if channel is not None:
            channel.close()
        for handle in (decompress_stderr_handle, tar_stderr_handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        if decompress_proc is not None and decompress_proc.poll() is None:
            try:
                decompress_proc.kill()
            except Exception:
                pass
        if tar_proc is not None and tar_proc.poll() is None:
            try:
                tar_proc.kill()
            except Exception:
                pass
        for path in (decompress_stderr.name, tar_stderr.name):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def _sync_remote_result_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Sync one remote result directory back into the local notebook results tree."""
    local_result_dir.mkdir(parents=True, exist_ok=True)
    if _remote_transport(config) == "paramiko":
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                connection = _connect_paramiko(config)
                if bool(config.get("remote_sync_compress", True)):
                    probe_completed = _run_paramiko_shell(
                        config,
                        _build_remote_archive_probe_command(remote_result_dir),
                    )
                    if probe_completed.returncode != 0:
                        return subprocess.CompletedProcess(
                            args=["paramiko-probe", remote_result_dir.as_posix(), str(local_result_dir)],
                            returncode=1,
                            stdout=probe_completed.stdout or "",
                            stderr=probe_completed.stderr or "",
                        )
                    probe_lines = [line.strip() for line in (probe_completed.stdout or "").splitlines() if line.strip()]
                    if len(probe_lines) < 3:
                        return subprocess.CompletedProcess(
                            args=["paramiko-probe", remote_result_dir.as_posix(), str(local_result_dir)],
                            returncode=1,
                            stdout=probe_completed.stdout or "",
                            stderr="Remote archive probe did not return the expected metadata",
                        )
                    compressor, raw_bytes_text, _archive_suffix = probe_lines[:3]
                    raw_bytes = int(raw_bytes_text or "0")
                    stream_completed = _stream_paramiko_archive_to_local_dir(
                        config,
                        remote_result_dir=remote_result_dir,
                        local_result_dir=local_result_dir,
                        compressor=compressor,
                        raw_bytes=raw_bytes,
                    )
                    if stream_completed.returncode != 0:
                        _progress_write(
                            "[OBGPU load] Streamed archive sync failed; retrying the same result dir over SFTP..."
                        )
                        _sftp_copy_tree(connection["sftp"], remote_result_dir.as_posix(), local_result_dir)
                        return subprocess.CompletedProcess(
                            args=["paramiko-stream-extract-fallback", remote_result_dir.as_posix(), str(local_result_dir)],
                            returncode=0,
                            stdout=stream_completed.stdout or "",
                            stderr=(stream_completed.stderr or "")
                            + "\n[OBGPU load] Stream archive sync failed, but SFTP fallback completed successfully.\n",
                        )
                else:
                    _sftp_copy_tree(connection["sftp"], remote_result_dir.as_posix(), local_result_dir)
            except Exception as exc:
                last_exc = exc
                _drop_paramiko_connection(config)
                if attempt == 0:
                    continue
                return subprocess.CompletedProcess(
                    args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
                    returncode=1,
                    stdout="",
                    stderr=str(exc),
                )
            return subprocess.CompletedProcess(
                args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
                returncode=0,
                stdout="",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=["paramiko-sftp", remote_result_dir.as_posix(), str(local_result_dir)],
            returncode=1,
            stdout="",
            stderr=str(last_exc) if last_exc is not None else "unknown paramiko sftp failure",
        )

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


def _local_result_dir_has_loadable_payload(result_dir: str | Path) -> bool:
    """Return True when the local result directory already has standard notebook payloads."""
    result_dir = Path(result_dir)
    for filename in ("input_times.pkl", "soma_vs.pkl", "gc_output_events.pkl", "lfp.pkl"):
        if (result_dir / filename).exists():
            return True
    return False


def _synthesize_partial_sync_summary(
    result_dir: str | Path,
    *,
    label: str,
    timestamp: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create a minimal summary when the payload files arrived but metadata did not."""
    result_dir = Path(result_dir)
    files = {}
    for filename in ("input_times.pkl", "soma_vs.pkl", "gc_output_events.pkl", "lfp.pkl"):
        path = result_dir / filename
        if path.exists():
            files[filename] = {"size_bytes": int(path.stat().st_size)}
    return {
        "label": label,
        "requested_label": label,
        "timestamp": timestamp,
        "paramset": config.get("paramset"),
        "nranks": config.get("nranks"),
        "files": files,
        "partial_sync": True,
    }


def _build_remote_allocation_submit_command(
    config: dict[str, Any],
) -> tuple[str, PurePosixPath, str]:
    """Build the remote helper invocation that submits one reusable allocation."""
    remote_helper = REPO_ROOT / "tools" / "remote" / "submit_slurm_allocation.py"
    helper_b64 = b64encode(remote_helper.read_bytes()).decode("ascii")
    python_exec = (
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"'
        ' && test -n "$REMOTE_PYTHON"'
        ' && exec "$REMOTE_PYTHON" -c '
        + shlex.quote(
            'import base64,sys; '
            'script_b64=sys.argv[1]; '
            'script_path=sys.argv[2]; '
            'sys.argv=sys.argv[2:]; '
            'namespace={"__name__":"__main__","__file__":script_path}; '
            'exec(compile(base64.b64decode(script_b64).decode("utf-8"), script_path, "exec"), namespace)'
        )
    )
    allocation_key = _slurm_allocation_cache_key(config)
    allocation_root = _remote_results_root(config) / ".obgpu-allocations" / allocation_key
    allocation_name_base = str(config.get("slurm_allocation_name") or "obgpu_notebook_alloc")
    allocation_name = f"{allocation_name_base[:100]}_{allocation_key[:8]}"
    allocation_time = config.get("slurm_allocation_time") or config.get("slurm_time")
    command = [
        helper_b64,
        str(remote_helper),
        "--alloc-root",
        allocation_root.as_posix(),
        "--name",
        allocation_name,
    ]
    for key, flag in (
        ("slurm_partition", "--partition"),
        ("slurm_account", "--account"),
        ("slurm_mem", "--mem"),
    ):
        value = config.get(key)
        if value not in (None, ""):
            command.extend([flag, str(value)])
    if allocation_time not in (None, ""):
        command.extend(["--time", str(allocation_time)])
    command.extend(["--heartbeat-timeout-s", str(_remote_heartbeat_timeout_s(config))])
    for key, flag in (
        ("slurm_gpus", "--gpus"),
        ("slurm_cpus_per_task", "--cpus-per-task"),
    ):
        value = config.get(key)
        if value not in (None, ""):
            command.extend([flag, str(int(value))])
    for extra in config.get("slurm_extra_args", []):
        command.append("--sbatch-arg={}".format(str(extra)))
    return python_exec + " " + _shell_join(command), allocation_root, allocation_name


def _build_remote_touch_command(path_value: str | PurePosixPath) -> str:
    """Build a remote command that refreshes one heartbeat path."""
    path = PurePosixPath(str(path_value))
    return (
        f"mkdir -p {shlex.quote(path.parent.as_posix())} && "
        f"touch {shlex.quote(path.as_posix())}"
    )


def _refresh_remote_heartbeat(
    config: dict[str, Any],
    heartbeat_path: str | PurePosixPath | None,
    *,
    warn: bool = False,
) -> bool:
    """Best-effort refresh of a remote notebook heartbeat file."""
    if heartbeat_path in (None, ""):
        return False
    try:
        completed = _run_ssh_shell(config, _build_remote_touch_command(str(heartbeat_path)))
    except Exception as exc:
        if warn:
            _progress_write(f"[Sol remote] Heartbeat refresh failed: {exc}")
        return False
    if completed.returncode != 0:
        if warn:
            stderr = (completed.stderr or "").strip()
            _progress_write(f"[Sol remote] Heartbeat refresh failed: {stderr or 'unknown error'}")
        return False
    return True


def _build_remote_cleanup_allocations_command(config: dict[str, Any]) -> str:
    """Build a remote command that cancels stale notebook-managed allocations."""
    cleanup_root = _remote_results_root(config) / ".obgpu-allocations"
    script = r'''
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
default_timeout_s = int(sys.argv[2])
now = time.time()
actions = []
if root.exists():
    for allocation_json in sorted(root.glob("*/allocation.json")):
        try:
            payload = json.loads(allocation_json.read_text())
        except Exception as exc:
            actions.append({"allocation_json": str(allocation_json), "action": "skip", "reason": "invalid_json", "error": str(exc)})
            continue
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            continue
        heartbeat_path = str(payload.get("heartbeat_path") or "").strip()
        try:
            timeout_s = int(payload.get("heartbeat_timeout_s") or default_timeout_s)
        except Exception:
            timeout_s = default_timeout_s
        reason = ""
        if not heartbeat_path:
            reason = "legacy_no_heartbeat"
        else:
            heartbeat = Path(heartbeat_path)
            if not heartbeat.exists():
                reason = "missing_heartbeat"
            elif timeout_s > 0 and now - heartbeat.stat().st_mtime > timeout_s:
                reason = "expired_heartbeat"
        if not reason:
            continue
        completed = subprocess.run(
            ["scancel", job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        actions.append({
            "job_id": job_id,
            "action": "cancel_requested",
            "reason": reason,
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "").strip(),
        })
print(json.dumps(actions, sort_keys=True))
'''
    script_b64 = b64encode(script.encode("utf-8")).decode("ascii")
    python_exec = (
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"'
        ' && test -n "$REMOTE_PYTHON"'
        ' && exec "$REMOTE_PYTHON" -c '
        + shlex.quote(
            'import base64,sys; '
            'script_b64=sys.argv[1]; '
            'sys.argv=sys.argv[1:]; '
            'exec(compile(base64.b64decode(script_b64).decode("utf-8"), "<obgpu-cleanup-allocations>", "exec"))'
        )
    )
    return python_exec + " " + _shell_join(
        [script_b64, cleanup_root.as_posix(), str(_remote_heartbeat_timeout_s(config))]
    )


def _cleanup_stale_remote_slurm_allocations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Cancel stale remote notebook-managed reusable allocations before a new run."""
    completed = _run_ssh_shell(config, _build_remote_cleanup_allocations_command(config))
    if completed.returncode != 0:
        _progress_write(f"[Sol remote] Stale allocation cleanup failed: {(completed.stderr or '').strip()}")
        return []
    try:
        actions = json.loads((completed.stdout or "").strip() or "[]")
    except json.JSONDecodeError:
        _progress_write("[Sol remote] Stale allocation cleanup returned invalid JSON.")
        return []
    if not isinstance(actions, list):
        return []
    cancelled = [action for action in actions if isinstance(action, dict) and action.get("action") == "cancel_requested"]
    for action in cancelled:
        job_id = action.get("job_id")
        reason = action.get("reason")
        _progress_write(f"[Sol remote] Cancelled stale reusable allocation {job_id} ({reason}).")
    return [action for action in actions if isinstance(action, dict)]


def _build_remote_allocation_discovery_command(
    config: dict[str, Any],
) -> tuple[str, PurePosixPath, str]:
    """Build the remote command that returns allocation metadata for one config, if present."""
    _submit_command, allocation_root, allocation_name = _build_remote_allocation_submit_command(config)
    allocation_json = allocation_root / "allocation.json"
    command = (
        f"if test -f {shlex.quote(allocation_json.as_posix())}; then "
        f"cat {shlex.quote(allocation_json.as_posix())}; "
        "fi"
    )
    return command, allocation_root, allocation_name


def _build_remote_submit_command(
    config: dict[str, Any],
    *,
    label: str,
    remote_repo_root: PurePosixPath,
    remote_results_root: PurePosixPath,
    benchmark_command: list[str],
    remote_mpi_exec: str,
    remote_git_ref: str | None,
) -> str:
    """Build the remote `submit_sol_run.py` invocation shell line."""
    remote_helper = REPO_ROOT / "tools" / "remote" / "submit_sol_run.py"
    benchmark_b64 = b64encode(json.dumps(benchmark_command).encode("utf-8")).decode("ascii")
    helper_b64 = b64encode(remote_helper.read_bytes()).decode("ascii")
    python_exec = (
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"'
        ' && test -n "$REMOTE_PYTHON"'
        ' && exec "$REMOTE_PYTHON" -c '
        + shlex.quote(
            'import base64,sys; '
            'script_b64=sys.argv[1]; '
            'script_path=sys.argv[2]; '
            'sys.argv=sys.argv[2:]; '
            'namespace={"__name__":"__main__","__file__":script_path}; '
            'exec(compile(base64.b64decode(script_b64).decode("utf-8"), script_path, "exec"), namespace)'
        )
    )
    command = [
        helper_b64,
        str(remote_helper),
        "--repo-root",
        remote_repo_root.as_posix(),
        "--results-base",
        remote_results_root.as_posix(),
        "--label",
        label,
        "--benchmark-command-b64",
        benchmark_b64,
        "--repo-mode",
        str(config.get("remote_repo_mode", "shared")),
        "--mpi-exec",
        str(remote_mpi_exec),
        "--conda-activate-cmd",
        str(config.get("remote_conda_activate_cmd")),
        "--heartbeat-timeout-s",
        str(_remote_heartbeat_timeout_s(config)),
    ]

    fallback_conda_activate_cmd = config.get("remote_fallback_conda_activate_cmd")
    runtime_profiles = config.get("remote_runtime_profiles") or []
    if runtime_profiles:
        profiles_b64 = b64encode(json.dumps(runtime_profiles, sort_keys=True).encode("utf-8")).decode("ascii")
        command.extend(["--runtime-profiles-b64", profiles_b64])
    if fallback_conda_activate_cmd not in (None, ""):
        command.extend(["--fallback-conda-activate-cmd", str(fallback_conda_activate_cmd)])
    fast_node_feature = config.get("remote_fast_node_feature")
    if fast_node_feature not in (None, ""):
        command.extend(["--fast-node-feature", str(fast_node_feature)])
    mechanism_profile = config.get("remote_mechanism_profile")
    if mechanism_profile not in (None, ""):
        command.extend(["--mechanism-profile", str(mechanism_profile)])
    fallback_mechanism_profile = config.get("remote_fallback_mechanism_profile")
    if fallback_mechanism_profile not in (None, ""):
        command.extend(["--fallback-mechanism-profile", str(fallback_mechanism_profile)])

    if remote_git_ref:
        command.extend(["--git-ref", remote_git_ref])
    if bool(config.get("remote_git_fetch", False)):
        command.append("--git-fetch")
        command.extend(["--git-remote", str(config.get("remote_git_remote", "origin"))])
    allocation_job_id = config.get("slurm_allocation_job_id")
    if allocation_job_id not in (None, ""):
        command.extend(["--allocation-job-id", str(allocation_job_id)])

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
        command.append("--sbatch-arg={}".format(str(extra)))

    return python_exec + " " + _shell_join(command)


def _build_remote_poll_command(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    remote_result_dir: PurePosixPath,
    job_id: str,
    wrapper_dir: str | None = None,
    worktree_path: str | None = None,
) -> str:
    """Build the remote `poll_sol_run.py` invocation shell line."""
    remote_helper = REPO_ROOT / "tools" / "remote" / "poll_sol_run.py"
    helper_b64 = b64encode(remote_helper.read_bytes()).decode("ascii")
    python_exec = (
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"'
        ' && test -n "$REMOTE_PYTHON"'
        ' && exec "$REMOTE_PYTHON" -c '
        + shlex.quote(
            'import base64,sys; '
            'script_b64=sys.argv[1]; '
            'script_path=sys.argv[2]; '
            'sys.argv=sys.argv[2:]; '
            'namespace={"__name__":"__main__","__file__":script_path}; '
            'exec(compile(base64.b64decode(script_b64).decode("utf-8"), script_path, "exec"), namespace)'
        )
    )
    command = [
        helper_b64,
        str(remote_helper),
        "--job-id",
        str(job_id),
        "--result-dir",
        remote_result_dir.as_posix(),
    ]
    if wrapper_dir not in (None, ""):
        command.extend(["--wrapper-dir", str(wrapper_dir)])
    if worktree_path not in (None, ""):
        command.extend(
            [
                "--repo-root",
                remote_repo_root.as_posix(),
                "--worktree-path",
                str(worktree_path),
            ]
        )
    return python_exec + " " + _shell_join(command)


def _build_remote_preflight_command(
    *,
    remote_repo_root: PurePosixPath,
) -> str:
    """Build one remote shell command that validates Sol-side prerequisites."""
    checks = [
        f'test -d {shlex.quote(remote_repo_root.as_posix())}',
        'REMOTE_PYTHON="$(command -v python3 || command -v python || true)"',
        'test -n "$REMOTE_PYTHON"',
        'command -v bash >/dev/null',
        'command -v git >/dev/null',
        'command -v sbatch >/dev/null',
        'command -v sacct >/dev/null',
        'command -v scancel >/dev/null',
        'command -v squeue >/dev/null',
        'command -v srun >/dev/null',
    ]
    return " && ".join(checks)


def _build_remote_result_listing_command(
    *,
    remote_result_dir: PurePosixPath,
) -> str:
    """Build one remote shell command that lists the synced result directory contents."""
    quoted_dir = shlex.quote(remote_result_dir.as_posix())
    return (
        f"if test -d {quoted_dir}; then "
        f"find {quoted_dir} -maxdepth 1 -type f -printf '%f\\t%s\\n' | sort; "
        "fi"
    )


def _build_remote_cancel_command(*, job_id: str) -> str:
    """Build one remote shell command that cancels a submitted Slurm job."""
    return f"scancel {shlex.quote(str(job_id))}"


def _query_remote_slurm_job_state(config: dict[str, Any], job_id: str) -> dict[str, str]:
    """Query one remote Slurm job state without requiring a result directory."""
    query_command = (
        f"squeue -j {shlex.quote(str(job_id))} -h -o '%T|%R' || true; "
        "printf '%s\\n' '__SACCT__'; "
        f"sacct -j {shlex.quote(str(job_id))} --format=JobIDRaw,State --parsable2 --noheader || true"
    )
    completed = _run_ssh_shell(config, query_command)
    if completed.returncode != 0:
        raise RuntimeError(
            "Remote Slurm job-state query failed.\n"
            f"Job id: {job_id}\n"
            f"Stdout:\n{completed.stdout}\n\nStderr:\n{completed.stderr}"
        )

    squeue_text, _marker, sacct_text = (completed.stdout or "").partition("__SACCT__\n")
    squeue_output = squeue_text.strip()
    sacct_output = sacct_text.strip()
    squeue_reason = ""
    squeue_location = ""

    if squeue_output:
        first_line = squeue_output.splitlines()[0]
        parts = first_line.split("|", 1)
        if len(parts) == 2:
            squeue_state = _normalize_slurm_state(parts[0])
            squeue_detail = parts[1].strip()
            if squeue_state == "PENDING":
                squeue_reason = squeue_detail
            else:
                squeue_location = squeue_detail
        else:
            squeue_state = _normalize_slurm_state(first_line)
        if squeue_state == "PENDING":
            return {"state": squeue_state, "reason": squeue_reason, "location": squeue_location}

    if sacct_output:
        for line in sacct_output.splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            raw_job_id, raw_state = parts
            if raw_job_id.strip() == str(job_id):
                state = _normalize_slurm_state(raw_state)
                if state:
                    return {"state": state, "reason": squeue_reason, "location": squeue_location}
        for line in sacct_output.splitlines():
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            state = _normalize_slurm_state(parts[1])
            if state:
                return {"state": state, "reason": squeue_reason, "location": squeue_location}

    if squeue_output:
        return {
            "state": _normalize_slurm_state(squeue_output.split("|", 1)[0]),
            "reason": squeue_reason,
            "location": squeue_location,
        }
    return {"state": "UNKNOWN", "reason": "", "location": ""}


def _ensure_cached_remote_slurm_allocation(config: dict[str, Any]) -> dict[str, Any]:
    """Acquire or reuse one notebook-cached remote Slurm allocation."""
    manual_job_id = config.get("slurm_allocation_job_id")
    if manual_job_id not in (None, ""):
        return {
            "job_id": str(manual_job_id),
            "cached": False,
            "manual": True,
            "state": "",
            "reason": "",
            "location": "",
        }
    if not bool(config.get("slurm_reuse_allocation", False)):
        return {
            "job_id": None,
            "cached": False,
            "manual": False,
            "state": "",
            "reason": "",
            "location": "",
        }

    cache_key = _slurm_allocation_cache_key(config)
    allocation = _LIVE_SLURM_ALLOCATIONS.get(cache_key)
    created_now = False
    runtime_config = _slurm_allocation_runtime_config(config)

    if allocation is not None:
        if allocation.get("heartbeat_path") in (None, ""):
            _LIVE_SLURM_ALLOCATIONS.pop(cache_key, None)
            allocation = None
        else:
            _refresh_remote_heartbeat(config, str(allocation["heartbeat_path"]), warn=True)

    if allocation is not None:
        status = _query_remote_slurm_job_state(config, str(allocation["job_id"]))
        state = status.get("state", "UNKNOWN")
        if state in _REMOTE_SLURM_TERMINAL_OK or state in _REMOTE_SLURM_TERMINAL_FAIL or state == "UNKNOWN":
            _LIVE_SLURM_ALLOCATIONS.pop(cache_key, None)
            allocation = None
        else:
            print(f"[Sol remote] Reusing cached allocation {allocation['job_id']}.", flush=True)

    if allocation is None:
        discover_command, allocation_root, allocation_name = _build_remote_allocation_discovery_command(config)
        discover_completed = _run_ssh_shell(config, discover_command)
        if discover_completed.returncode != 0:
            raise RuntimeError(
                "Remote Slurm allocation discovery failed.\n"
                f"Stdout:\n{discover_completed.stdout}\n\nStderr:\n{discover_completed.stderr}"
            )
        discovered_text = (discover_completed.stdout or "").strip()
        if discovered_text:
            try:
                discovered = json.loads(discovered_text)
            except json.JSONDecodeError:
                discovered = None
            if isinstance(discovered, dict) and discovered.get("job_id") not in (None, ""):
                discovered_job_id = str(discovered["job_id"])
                status = _query_remote_slurm_job_state(config, discovered_job_id)
                state = status.get("state", "UNKNOWN")
                if state not in _REMOTE_SLURM_TERMINAL_OK and state not in _REMOTE_SLURM_TERMINAL_FAIL and state != "UNKNOWN":
                    heartbeat_path = str(discovered.get("heartbeat_path") or "")
                    if not heartbeat_path:
                        print(
                            f"[Sol remote] Cancelling legacy reusable allocation {discovered_job_id} without heartbeat lease.",
                            flush=True,
                        )
                        _run_ssh_shell(config, _build_remote_cancel_command(job_id=discovered_job_id))
                    else:
                        _refresh_remote_heartbeat(config, heartbeat_path, warn=True)
                        allocation = {
                            "job_id": discovered_job_id,
                            "cache_key": cache_key,
                            "allocation_root": str(discovered.get("allocation_root") or allocation_root.as_posix()),
                            "batch_script": str(discovered.get("batch_script") or ""),
                            "heartbeat_path": heartbeat_path,
                            "heartbeat_timeout_s": discovered.get("heartbeat_timeout_s"),
                            "slurm_log_pattern": str(discovered.get("slurm_log_pattern") or ""),
                            "name": str(discovered.get("name") or allocation_name),
                            "cached": True,
                            "manual": False,
                            "config": runtime_config,
                        }
                        _LIVE_SLURM_ALLOCATIONS[cache_key] = allocation
                        print(f"[Sol remote] Reusing discovered allocation {allocation['job_id']}.", flush=True)

    if allocation is None:
        print("[Sol remote] Requesting reusable Slurm allocation...", flush=True)
        submit_command, allocation_root, allocation_name = _build_remote_allocation_submit_command(config)
        submit_completed = _run_ssh_shell(config, submit_command)
        if submit_completed.returncode != 0:
            raise RuntimeError(
                "Remote Slurm allocation submission failed.\n"
                f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
            )
        try:
            submission = json.loads((submit_completed.stdout or "").strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Remote Slurm allocation submission did not return valid JSON.\n"
                f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
            ) from exc
        allocation = {
            "job_id": str(submission["job_id"]),
            "cache_key": cache_key,
            "allocation_root": str(submission.get("allocation_root") or allocation_root.as_posix()),
            "batch_script": str(submission.get("batch_script") or ""),
            "heartbeat_path": str(submission.get("heartbeat_path") or allocation_root / "notebook-heartbeat.txt"),
            "heartbeat_timeout_s": submission.get("heartbeat_timeout_s"),
            "slurm_log_pattern": str(submission.get("slurm_log_pattern") or ""),
            "name": str(submission.get("name") or allocation_name),
            "cached": True,
            "manual": False,
            "config": runtime_config,
        }
        _LIVE_SLURM_ALLOCATIONS[cache_key] = allocation
        created_now = True

    last_signature: tuple[str, str, str] | None = None
    try:
        while True:
            _refresh_remote_heartbeat(config, str(allocation.get("heartbeat_path") or ""), warn=False)
            status = _query_remote_slurm_job_state(config, str(allocation["job_id"]))
            state = status.get("state", "UNKNOWN")
            reason = str(status.get("reason") or "").strip()
            location = str(status.get("location") or "").strip()
            status_signature = (state, reason, location)
            if status_signature != last_signature:
                detail = ""
                if state == "PENDING" and reason:
                    detail = f" reason={reason}"
                elif location and state not in {"UNKNOWN", "PENDING"}:
                    detail = f" where={location}"
                print(
                    f"[Sol remote] Allocation {allocation['job_id']}: {state}{detail}",
                    flush=True,
                )
                last_signature = status_signature

            if state == "RUNNING":
                allocation.update(
                    {
                        "state": state,
                        "reason": reason,
                        "location": location,
                        "config": runtime_config,
                    }
                )
                _LIVE_SLURM_ALLOCATIONS[cache_key] = allocation
                return allocation
            if state in _REMOTE_SLURM_TERMINAL_OK or state in _REMOTE_SLURM_TERMINAL_FAIL:
                _LIVE_SLURM_ALLOCATIONS.pop(cache_key, None)
                raise RuntimeError(
                    "Reusable Slurm allocation terminated before it became runnable.\n"
                    f"Job id: {allocation['job_id']}\n"
                    f"State: {state}\n"
                    f"Reason: {reason}\n"
                    f"Location: {location}"
                )
            time.sleep(5.0)
    except KeyboardInterrupt:
        if created_now:
            print(
                f"[Sol remote] Interrupt received; cancelling new allocation {allocation['job_id']}...",
                flush=True,
            )
            _run_ssh_shell(config, _build_remote_cancel_command(job_id=str(allocation["job_id"])))
            _LIVE_SLURM_ALLOCATIONS.pop(cache_key, None)
        raise


def release_remote_slurm_allocation(config: dict[str, Any]) -> bool:
    """Cancel and forget the cached or remotely-discovered reusable Slurm allocation."""
    cache_key = _slurm_allocation_cache_key(config)
    allocation = _LIVE_SLURM_ALLOCATIONS.pop(cache_key, None)
    if allocation is None:
        discover_command, _allocation_root, _allocation_name = _build_remote_allocation_discovery_command(config)
        discover_completed = _run_ssh_shell(config, discover_command)
        if discover_completed.returncode != 0:
            print(f"[Sol remote] Allocation discovery stderr: {(discover_completed.stderr or '').strip()}", flush=True)
            return False
        discovered_text = (discover_completed.stdout or "").strip()
        if discovered_text:
            try:
                discovered = json.loads(discovered_text)
            except json.JSONDecodeError:
                discovered = None
            if isinstance(discovered, dict) and discovered.get("job_id") not in (None, ""):
                allocation = {"job_id": str(discovered["job_id"])}
        if allocation is None:
            print("[Sol remote] No cached or discovered reusable allocation for this config.", flush=True)
            return False
    job_id = str(allocation["job_id"])
    print(f"[Sol remote] Releasing reusable allocation {job_id}...", flush=True)
    completed = _run_ssh_shell(config, _build_remote_cancel_command(job_id=job_id))
    if completed.returncode != 0:
        print(f"[Sol remote] scancel stderr: {(completed.stderr or '').strip()}", flush=True)
        return False
    print(f"[Sol remote] Cancellation requested for allocation {job_id}.", flush=True)
    return True


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
    allocation_job_id = config.get("slurm_allocation_job_id")
    include_mpi_launcher = True
    if allocation_job_id not in (None, ""):
        include_mpi_launcher = int(config.get("nranks", 1)) != 1
    remote_command = build_run_command(
        config,
        label,
        repo_root=remote_repo_root,
        results_base=remote_results_root,
        mpi_exec=str(remote_mpi_exec),
        include_mpi_launcher=include_mpi_launcher,
    )
    submit_command = _build_remote_submit_command(
        config,
        label=label,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        benchmark_command=remote_command,
        remote_mpi_exec=str(remote_mpi_exec),
        remote_git_ref=remote_git_ref,
    )
    return (
        remote_repo_root,
        remote_results_root,
        remote_command,
        {
            "runner_backend": str(config.get("runner_backend", "slurm_remote")),
            "remote_host": _require_remote_host(config),
            "remote_repo_root": remote_repo_root.as_posix(),
            "remote_results_root": remote_results_root.as_posix(),
            "remote_mpi_exec": str(remote_mpi_exec),
            "remote_repo_mode": str(config.get("remote_repo_mode", "shared")),
            "remote_git_ref": remote_git_ref,
            "remote_git_fetch": bool(config.get("remote_git_fetch", False)),
            "remote_git_remote": str(config.get("remote_git_remote", "origin")),
            "slurm_allocation_job_id": None if allocation_job_id in (None, "") else str(allocation_job_id),
        },
        submit_command,
    )


def _remote_status_has_artifacts(status: dict[str, Any] | None) -> bool:
    """Return whether the remote poll status saw any useful output artifacts."""
    if not status:
        return False
    return any(
        bool(status.get(key))
        for key in (
            "summary_exists",
            "stdout_exists",
            "stderr_exists",
            "bootstrap_exists",
            "command_exists",
            "slurm_log_exists",
        )
    )


def _create_git_bundle_for_commit(commit_sha: str, *, exclude_ref: str | None = None) -> tuple[Path, str]:
    """Create a temporary git bundle for the requested commit."""
    branch_name = _resolve_local_git_branch()
    temp_ref: str | None = None
    source_ref: str

    if branch_name and _git_ref_points_to_commit(branch_name, commit_sha):
        source_ref = f"refs/heads/{branch_name}"
    else:
        temp_ref = f"refs/obgpu-notebook-sync/{commit_sha}"
        updated = subprocess.run(
            ["git", "update-ref", temp_ref, commit_sha],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if updated.returncode != 0:
            raise RuntimeError(
                "Could not create a temporary git ref for the Sol sync bundle.\n"
                f"Commit: {commit_sha}\n"
                f"Stderr:\n{updated.stderr}"
            )
        source_ref = temp_ref

    bundle_handle = tempfile.NamedTemporaryFile(prefix="obgpu-sol-sync-", suffix=".bundle", delete=False)
    bundle_path = Path(bundle_handle.name)
    bundle_handle.close()

    try:
        bundle_args = ["git", "bundle", "create", str(bundle_path), source_ref]
        if (
            exclude_ref
            and not _git_ref_points_to_commit(exclude_ref, commit_sha)
            and _git_ref_is_ancestor(exclude_ref, commit_sha)
        ):
            bundle_args.append(f"^{exclude_ref}")
        created = subprocess.run(
            bundle_args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if created.returncode != 0:
            raise RuntimeError(
                "Could not create a git bundle for the Sol backend.\n"
                f"Source ref: {source_ref}\n"
                f"Stderr:\n{created.stderr}"
            )
        return bundle_path, source_ref
    except Exception:
        try:
            bundle_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        if temp_ref is not None:
            subprocess.run(
                ["git", "update-ref", "-d", temp_ref],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )


def _find_remote_git_bundle_base(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    candidate_shas: list[str],
) -> str | None:
    """Return the newest local ancestor SHA already present in the remote repo."""
    candidates = [sha for sha in candidate_shas if sha]
    if not candidates:
        return None

    quoted_repo = shlex.quote(remote_repo_root.as_posix())
    quoted_candidates = " ".join(shlex.quote(sha) for sha in candidates)
    command = (
        f"for sha in {quoted_candidates}; do "
        f"if git -C {quoted_repo} cat-file -e \"$sha^{{commit}}\" 2>/dev/null; "
        "then printf '%s\\n' \"$sha\"; exit 0; fi; "
        "done; exit 1"
    )
    completed = _run_ssh_shell(config, command)
    if completed.returncode != 0:
        return None
    selected = (completed.stdout or "").strip().splitlines()
    if not selected:
        return None
    base_sha = selected[-1].strip()
    return base_sha if base_sha in candidates else None


def _ensure_remote_git_ref_available(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    remote_git_ref: str | None,
) -> None:
    """Ensure the current local commit exists in the remote repo without requiring manual git push."""
    if not remote_git_ref:
        return

    cache_key = _remote_git_ref_cache_key(config, remote_repo_root)
    cached_refs = _LIVE_REMOTE_GIT_REFS.setdefault(cache_key, set())
    if remote_git_ref in cached_refs:
        _progress_write(f"[Sol remote] Remote git cache hit for commit {remote_git_ref[:12]}.")
        return

    repo_probe_completed = _run_ssh_shell(
        config,
        _build_remote_git_repo_probe_command(remote_repo_root),
    )
    if repo_probe_completed.returncode != 0:
        raise RuntimeError(
            "The configured Sol remote_repo_root is not an accessible git repo, so the notebook "
            "cannot publish the local commit there.\n"
            f"Remote repo: {remote_repo_root.as_posix()}\n"
            f"Commit: {remote_git_ref}\n"
            f"Stdout:\n{repo_probe_completed.stdout}\n\n"
            f"Stderr:\n{repo_probe_completed.stderr}"
        )

    _progress_write(f"[Sol remote] Checking whether remote repo already has commit {remote_git_ref[:12]}...")
    check_command = (
        f"git -C {shlex.quote(remote_repo_root.as_posix())} "
        f"cat-file -e {shlex.quote(remote_git_ref + '^{commit}')}"
    )
    check_completed = _run_ssh_shell(config, check_command)
    if check_completed.returncode == 0:
        cached_refs.add(remote_git_ref)
        _progress_write(f"[Sol remote] Remote repo already has commit {remote_git_ref[:12]}.")
        return

    if _remote_transport(config) != "paramiko":
        raise RuntimeError(
            "The requested git ref is not available in the remote Sol repo, and automatic sync "
            "requires ssh_transport='paramiko'.\n"
            f"Remote ref: {remote_git_ref}\n"
            "Push the branch manually or enable the Paramiko transport."
        )

    connection = _connect_paramiko(config)
    sftp = connection["sftp"]
    bundle_base = _find_remote_git_bundle_base(
        config,
        remote_repo_root=remote_repo_root,
        candidate_shas=_local_git_sync_base_candidates(remote_git_ref),
    )
    if bundle_base:
        _progress_write(
            f"[Sol remote] Building incremental git bundle for commit "
            f"{remote_git_ref[:12]} from remote base {bundle_base[:12]}..."
        )
    else:
        _progress_write(
            f"[Sol remote] Building self-contained git bundle for commit {remote_git_ref[:12]} "
            "because no tested remote ancestor was found..."
        )
    bundle_path, source_ref = _create_git_bundle_for_commit(remote_git_ref, exclude_ref=bundle_base)
    remote_bundle_path = f"/tmp/obgpu-sync-{remote_git_ref[:12]}-{os.getpid()}.bundle"
    remote_private_ref = f"refs/obgpu-notebook-sync/{remote_git_ref}"
    remote_git_lock = shlex.quote((remote_repo_root / ".obgpu-git.lock").as_posix())
    fetch_body = (
        f"git -C {shlex.quote(remote_repo_root.as_posix())} fetch --force --no-tags "
        f"{shlex.quote(remote_bundle_path)} "
        f"{shlex.quote(source_ref)}:{shlex.quote(remote_private_ref)}"
        f" && git -C {shlex.quote(remote_repo_root.as_posix())} "
        f"cat-file -e {shlex.quote(remote_git_ref + '^{commit}')}"
        f" && rm -f {shlex.quote(remote_bundle_path)}"
    )
    fetch_command = (
        f"if command -v flock >/dev/null 2>&1; then "
        f"touch {remote_git_lock} && flock {remote_git_lock} bash -lc {shlex.quote(fetch_body)}; "
        f"else {fetch_body}; fi"
    )

    try:
        _progress_write(f"[Sol remote] Uploading git bundle for commit {remote_git_ref[:12]}...")
        sftp.put(str(bundle_path), remote_bundle_path)
        _progress_write(f"[Sol remote] Publishing local commit {remote_git_ref[:12]} to remote repo...")
        fetch_completed = _run_paramiko_shell(config, fetch_command)
        if fetch_completed.returncode != 0:
            raise RuntimeError(
                "Could not publish the current local git commit to the Sol repo over the notebook SSH transport.\n"
                f"Remote repo: {remote_repo_root.as_posix()}\n"
                f"Commit: {remote_git_ref}\n"
                f"Stdout:\n{fetch_completed.stdout}\n\n"
                f"Stderr:\n{fetch_completed.stderr}"
            )
        cached_refs.add(remote_git_ref)
        _progress_write(f"[Sol remote] Remote repo now has commit {remote_git_ref[:12]}.")
    finally:
        try:
            bundle_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            sftp.remove(remote_bundle_path)
        except Exception:
            pass


def _run_remote_simulation(
    config: dict[str, Any],
    *,
    label: str,
    timestamp: str,
    local_result_dir: Path,
) -> RunRecord:
    """Submit one Sol Slurm job, wait for completion, sync results, and return a run record."""
    effective_config = dict(config)
    remote_repo_root = _remote_repo_root(effective_config)
    remote_git_ref = _resolve_remote_git_ref(effective_config)
    (
        _remote_repo_root_value,
        _remote_results_root_value,
        remote_benchmark_command,
        remote_metadata,
        submit_shell,
    ) = _remote_submission_payload(effective_config, label=label)
    _ensure_remote_git_ref_available(
        effective_config,
        remote_repo_root=remote_repo_root,
        remote_git_ref=remote_git_ref,
    )
    _progress_write("[Sol remote] Running remote preflight checks...")

    preflight_completed = _run_ssh_shell(
        effective_config,
        _build_remote_preflight_command(remote_repo_root=remote_repo_root),
    )
    if preflight_completed.returncode != 0:
        local_result_dir.mkdir(parents=True, exist_ok=True)
        completed = SimpleNamespace(
            returncode=preflight_completed.returncode,
            stdout=preflight_completed.stdout or "",
            stderr=preflight_completed.stderr or "",
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
            "Remote Sol preflight failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Stdout:\n{preflight_completed.stdout}\n\n"
            f"Stderr:\n{preflight_completed.stderr}"
        )

    if bool(effective_config.get("remote_cleanup_stale_allocations", True)):
        _cleanup_stale_remote_slurm_allocations(effective_config)

    allocation_info = _ensure_cached_remote_slurm_allocation(effective_config)
    allocation_heartbeat_path = None
    if allocation_info.get("job_id") not in (None, ""):
        effective_config["slurm_allocation_job_id"] = str(allocation_info["job_id"])
        allocation_heartbeat_path = allocation_info.get("heartbeat_path")
        (
            _remote_repo_root_value,
            _remote_results_root_value,
            remote_benchmark_command,
            remote_metadata,
            submit_shell,
        ) = _remote_submission_payload(effective_config, label=label)
        remote_metadata["auto_reused_allocation"] = bool(
            effective_config.get("slurm_reuse_allocation", False)
            and not allocation_info.get("manual", False)
        )
        remote_metadata["allocation_state"] = allocation_info.get("state", "")
        remote_metadata["allocation_reason"] = allocation_info.get("reason", "")
        remote_metadata["allocation_location"] = allocation_info.get("location", "")
        remote_metadata["allocation_heartbeat_path"] = allocation_heartbeat_path

    _progress_write("[Sol remote] Submitting Slurm job...")
    submit_completed = _run_ssh_shell(effective_config, submit_shell)
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
            config=effective_config,
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
    remote_job_heartbeat_path = submission.get("heartbeat_path")
    remote_metadata["job_heartbeat_path"] = remote_job_heartbeat_path
    remote_metadata["heartbeat_timeout_s"] = submission.get(
        "heartbeat_timeout_s",
        _remote_heartbeat_timeout_s(effective_config),
    )
    _progress_write(f"[Sol remote] Submitted job {submission['job_id']}.")
    poll_interval_s = max(float(effective_config.get("remote_poll_interval_s", 1.0)), 1.0)
    live_status = bool(effective_config.get("remote_live_status", True))
    live_logs = bool(effective_config.get("remote_live_logs", True))
    poll_transcript: list[dict[str, Any]] = []
    final_status: dict[str, Any] | None = None
    missing_artifact_retries = 0
    last_status_signature: tuple[Any, ...] | None = None
    last_live_tails = {
        "bootstrap": "",
        "stdout": "",
        "stderr": "",
        "slurm": "",
    }
    last_live_lines = {
        "bootstrap": None,
        "stdout": None,
        "stderr": None,
        "slurm": None,
    }
    last_live_partials = {
        "bootstrap": "",
        "stdout": "",
        "stderr": "",
        "slurm": "",
    }
    sim_progress_bar: _ProgressBar | None = None
    sim_progress_total_ms: int | None = None
    sim_last_progress_ms: int | None = None
    sim_waiting_for_progress_logged = False
    sim_progress_complete = False
    sim_finalizing_logged = False

    def refresh_remote_leases(*, warn: bool = False) -> None:
        _refresh_remote_heartbeat(effective_config, remote_job_heartbeat_path, warn=warn)
        _refresh_remote_heartbeat(effective_config, allocation_heartbeat_path, warn=warn)

    def poll_remote_status_once(*, refresh_heartbeat: bool = True) -> dict[str, Any]:
        if refresh_heartbeat:
            refresh_remote_leases()
        poll_shell = _build_remote_poll_command(
            effective_config,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_result_dir,
            job_id=str(submission["job_id"]),
            wrapper_dir=str(submission.get("wrapper_dir") or ""),
            worktree_path=str(submission.get("worktree_path") or ""),
        )
        poll_completed = _run_ssh_shell(effective_config, poll_shell)
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
        return status

    def emit_live_remote_updates(status: dict[str, Any]) -> None:
        nonlocal last_status_signature
        nonlocal sim_progress_bar
        nonlocal sim_progress_total_ms
        nonlocal sim_last_progress_ms
        nonlocal sim_waiting_for_progress_logged
        nonlocal sim_progress_complete
        nonlocal sim_finalizing_logged
        status_signature = (
            status.get("state"),
            bool(status.get("summary_exists")),
            bool(status.get("stdout_exists")),
            bool(status.get("stderr_exists")),
            bool(status.get("bootstrap_exists")),
            bool(status.get("command_exists")),
            bool(status.get("slurm_log_exists")),
        )
        if live_status and status_signature != last_status_signature:
            state = str(status.get("state", "UNKNOWN"))
            reason = str(status.get("reason") or "").strip()
            location = str(status.get("location") or "").strip()
            flags = []
            if status.get("bootstrap_exists"):
                flags.append("bootstrap")
            if status.get("command_exists"):
                flags.append("command")
            if status.get("stdout_exists"):
                flags.append("stdout")
            if status.get("stderr_exists"):
                flags.append("stderr")
            if status.get("slurm_log_exists"):
                flags.append("slurm")
            if status.get("summary_exists"):
                flags.append("summary")
            flag_text = ", ".join(flags) if flags else "no artifacts yet"
            if state == "PENDING" and reason:
                flag_text = f"{flag_text}; reason={reason}"
            elif location and state not in {"PENDING", "UNKNOWN"}:
                flag_text = f"{flag_text}; where={location}"
            _progress_write(f"[Sol remote] Job {submission['job_id']}: {state} ({flag_text})")
            last_status_signature = status_signature
        progress_total_ms = status.get("progress_total_ms")
        progress_current_ms = status.get("progress_current_ms")
        if (
            not sim_progress_complete
            and progress_total_ms not in (None, "", 0)
            and progress_current_ms is not None
        ):
            total_ms = max(int(float(progress_total_ms)), 0)
            current_ms = max(0, min(int(float(progress_current_ms)), total_ms))
            if total_ms > 0:
                if sim_progress_bar is None or sim_progress_total_ms != total_ms:
                    if sim_progress_bar is not None:
                        sim_progress_bar.close()
                    sim_progress_total_ms = total_ms
                    sim_progress_bar = _ProgressBar(
                        total=total_ms,
                        desc="Sim",
                        unit="ms",
                        unit_scale=False,
                    )
                    sim_waiting_for_progress_logged = False
                sim_progress_bar.update_to(current_ms)
                sim_last_progress_ms = current_ms
        state = str(status.get("state", "UNKNOWN"))
        if (
            state == "RUNNING"
            and sim_progress_bar is None
            and not sim_waiting_for_progress_logged
            and not sim_progress_complete
            and not status.get("summary_exists")
        ):
            _progress_write("[Sol remote] Simulation started; waiting for first progress update...")
            sim_waiting_for_progress_logged = True
        if (
            sim_progress_bar is not None
            and sim_progress_total_ms is not None
            and sim_last_progress_ms is not None
            and sim_last_progress_ms >= sim_progress_total_ms
        ) or status.get("summary_exists"):
            if sim_progress_bar is not None:
                sim_progress_bar.close()
                sim_progress_bar = None
                sim_progress_total_ms = None
            if status.get("summary_exists") and not sim_finalizing_logged:
                _progress_write("[Sol remote] Remote simulation finished; finalizing artifacts...")
                sim_finalizing_logged = True
            sim_progress_complete = True
        if live_logs:
            for kind in ("bootstrap", "stdout", "stderr", "slurm"):
                tail_text = str(status.get(f"{kind}_tail") or "")
                if not tail_text or tail_text == last_live_tails[kind]:
                    continue
                previous = last_live_tails[kind]
                if previous and tail_text.startswith(previous):
                    delta_text = tail_text[len(previous):]
                else:
                    delta_text = tail_text
                delta_text = last_live_partials[kind] + delta_text.replace("\r", "\n")
                if delta_text:
                    segments = delta_text.split("\n")
                    if delta_text.endswith("\n"):
                        last_live_partials[kind] = ""
                    else:
                        last_live_partials[kind] = segments.pop() if segments else delta_text
                    for line in segments:
                        filtered = _filter_live_remote_log_line(kind, line)
                        if filtered is None:
                            continue
                        if filtered == last_live_lines[kind]:
                            continue
                        _progress_write(f"[Sol remote][{kind}] {filtered}")
                        last_live_lines[kind] = filtered
                else:
                    last_live_partials[kind] = ""
                last_live_tails[kind] = tail_text
        if status.get("done"):
            if sim_progress_bar is not None:
                sim_progress_bar.close()
                sim_progress_bar = None
                sim_progress_total_ms = None

    def close_live_progress_bars() -> None:
        nonlocal sim_progress_bar, sim_progress_total_ms
        if sim_progress_bar is not None:
            sim_progress_bar.close()
            sim_progress_bar = None
            sim_progress_total_ms = None

    def cancel_remote_job_and_sync(reason_text: str) -> None:
        nonlocal final_status
        close_live_progress_bars()
        _progress_write(
            f"[Sol remote] {reason_text}; beginning shutdown for job {submission['job_id']}..."
        )
        try:
            cancel_completed = _run_ssh_shell(
                effective_config,
                _build_remote_cancel_command(job_id=str(submission["job_id"])),
            )
            if cancel_completed.returncode != 0 and (cancel_completed.stderr or "").strip():
                _progress_write(f"[Sol remote] scancel stderr: {(cancel_completed.stderr or '').strip()}")
            else:
                _progress_write(f"[Sol remote] Cancellation requested; waiting for remote cleanup...")
        except Exception as exc:
            _progress_write(f"[Sol remote] Failed to request cancellation: {exc}")

        cancel_deadline = time.time() + 30.0
        cancel_confirmed = False
        while time.time() < cancel_deadline:
            try:
                status = poll_remote_status_once(refresh_heartbeat=False)
            except Exception as exc:
                _progress_write(f"[Sol remote] Remote shutdown poll failed: {exc}")
                break
            try:
                emit_live_remote_updates(status)
            except Exception as exc:
                _progress_write(f"[Sol remote] Remote shutdown status rendering failed: {exc}")
            if status.get("done"):
                final_status = status
                cancel_confirmed = True
                _progress_write(
                    f"[Sol remote] Job {submission['job_id']} reached terminal state {status.get('state', 'UNKNOWN')}."
                )
                break
            time.sleep(1.0)

        if not cancel_confirmed:
            _progress_write(
                f"[Sol remote] Remote shutdown not yet confirmed; syncing partial artifacts anyway..."
            )
        else:
            _progress_write(f"[Sol remote] Syncing partial remote artifacts...")
        try:
            sync_completed = _sync_remote_result_dir(
                effective_config,
                remote_result_dir=remote_result_dir,
                local_result_dir=local_result_dir,
            )
            (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
            (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
            if sync_completed.returncode == 0:
                _progress_write(f"[Sol remote] Partial artifacts synced to {local_result_dir}")
            else:
                _progress_write(
                    f"[Sol remote] Partial artifact sync failed (rc={sync_completed.returncode})."
                )
        except Exception as exc:
            _progress_write(f"[Sol remote] Partial artifact sync failed: {exc}")

    try:
        while True:
            refresh_remote_leases(warn=True)
            status = poll_remote_status_once(refresh_heartbeat=False)
            emit_live_remote_updates(status)
            if status.get("done"):
                if not status.get("ok") and not _remote_status_has_artifacts(status) and missing_artifact_retries < 3:
                    missing_artifact_retries += 1
                    time.sleep(3.0)
                    continue
                final_status = status
                break
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        cancel_remote_job_and_sync("Interrupt received")
        raise KeyboardInterrupt(
            f"Interrupted remote Sol run and requested cancellation for job {submission['job_id']}."
        )
    except Exception:
        cancel_remote_job_and_sync("Local notebook error while monitoring remote run")
        raise

    sync_completed = _sync_remote_result_dir(
        effective_config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
    )
    (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
    (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
    sync_warning = None
    if sync_completed.returncode != 0:
        if _local_result_dir_has_loadable_payload(local_result_dir):
            sync_warning = (
                "Remote Sol result sync reported an error, but standard payload files were already present locally. "
                "Proceeding with the partial local copy.\n"
                f"{sync_completed.stderr}"
            )
            _progress_write(f"[OBGPU load] {sync_warning}")
        else:
            raise RuntimeError(
                "Remote Sol result sync failed.\n"
                f"Result dir: {local_result_dir}\n"
                f"rsync stderr:\n{sync_completed.stderr}"
            )
    _progress_write(f"[OBGPU load] Remote sync finished: {local_result_dir}")

    stdout_text = (local_result_dir / "stdout.txt").read_text() if (local_result_dir / "stdout.txt").exists() else ""
    stderr_text = (local_result_dir / "stderr.txt").read_text() if (local_result_dir / "stderr.txt").exists() else ""
    bootstrap_text = (
        (local_result_dir / "bootstrap.log").read_text()
        if (local_result_dir / "bootstrap.log").exists()
        else ""
    )
    slurm_logs = sorted(local_result_dir.glob("slurm-*.out"))
    slurm_text = slurm_logs[-1].read_text() if slurm_logs else ""
    if final_status and not final_status.get("ok") and not any((stdout_text, stderr_text, bootstrap_text, slurm_text)):
        time.sleep(3.0)
        sync_completed = _sync_remote_result_dir(
            effective_config,
            remote_result_dir=remote_result_dir,
            local_result_dir=local_result_dir,
        )
        (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
        (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
        if sync_completed.returncode == 0:
            stdout_text = (local_result_dir / "stdout.txt").read_text() if (local_result_dir / "stdout.txt").exists() else ""
            stderr_text = (local_result_dir / "stderr.txt").read_text() if (local_result_dir / "stderr.txt").exists() else ""
            bootstrap_text = (
                (local_result_dir / "bootstrap.log").read_text()
                if (local_result_dir / "bootstrap.log").exists()
                else ""
            )
            slurm_logs = sorted(local_result_dir.glob("slurm-*.out"))
            slurm_text = slurm_logs[-1].read_text() if slurm_logs else ""
    remote_listing_text = ""
    if final_status and not final_status.get("ok") and not any((stdout_text, stderr_text, bootstrap_text, slurm_text)):
        listing_completed = _run_ssh_shell(
            effective_config,
            _build_remote_result_listing_command(remote_result_dir=remote_result_dir),
        )
        remote_listing_text = (listing_completed.stdout or "").strip()
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
    elif sync_warning is not None and _local_result_dir_has_loadable_payload(local_result_dir):
        summary = _synthesize_partial_sync_summary(
            local_result_dir,
            label=label,
            timestamp=timestamp,
            config=effective_config,
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    compact_poll_events = _compact_remote_poll_events(poll_transcript)
    poll_events_path = None
    if compact_poll_events:
        poll_events_path = local_result_dir / "remote_poll_events.json"
        poll_events_path.write_text(json.dumps(_json_ready(compact_poll_events), indent=2, sort_keys=True))

    _write_notebook_run_info(
        local_result_dir,
        config=effective_config,
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
                "submit_response": _summarize_remote_submit_response(submission),
                "final_status": _summarize_remote_status(final_status),
                "sync_warning": sync_warning,
                "poll_sample_count": len(poll_transcript),
                "poll_event_count": len(compact_poll_events),
                "poll_events_file": poll_events_path.name if poll_events_path is not None else None,
                "resolved_git_ref": remote_git_ref,
                "resolved_git_commit": remote_git_commit,
            }
        },
    )

    if returncode != 0:
        stderr_tail = stderr_text.strip()[-4000:]
        stdout_tail = stdout_text.strip()[-2000:]
        bootstrap_tail = bootstrap_text.strip()[-4000:]
        slurm_tail = slurm_text.strip()[-4000:]
        remote_listing_tail = remote_listing_text.strip()[-4000:]
        raise RuntimeError(
            "Remote Sol simulation failed.\n"
            f"Result dir: {local_result_dir}\n"
            f"Command: {_shell_join(remote_benchmark_command)}\n"
            f"Stdout tail:\n{stdout_tail}\n\n"
            f"Stderr tail:\n{stderr_tail}\n\n"
            f"Bootstrap tail:\n{bootstrap_tail}\n\n"
            f"Slurm tail:\n{slurm_tail}\n\n"
            f"Remote files:\n{remote_listing_tail}"
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
    payload["resolved_execution_mode"] = _json_ready(_resolve_execution_mode(config))

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

    if runner_backend in {"sol_slurm", "slurm_remote"}:
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
    result_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=result_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

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
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Run a parameter sweep by repeatedly calling :func:`run_and_load`.

    Single-axis form (original)::

        sweep = run_parameter_sweep(config, 'gaba_gmax', [0, 1, 2, 4])

    Joint form — pairs parameters by list index::

        sweep = run_parameter_sweep(
            config,
            {'gaba_gmax': [0, 1, 2], 'gap_mc': [16, 32, 64]},
        )
        # Runs 3 simulations: (gaba_gmax=0, gap_mc=16), (1, 32), (2, 64)

    The returned dict always has the same shape:
    ``{"path": ..., "values": [...], "items": [...], "paramset": ...}``.
    For joint sweeps ``path`` is the param dict and each item's ``value`` is a
    sub-dict of ``{path: value}`` pairs.
    """
    base_config = build_run_config(**deepcopy(base_config))

    if isinstance(sweep_path, dict):
        # Joint sweep — validate lengths match
        paths = list(sweep_path.keys())
        all_values = list(sweep_path.values())
        lengths = [len(v) for v in all_values]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All parameter lists must have the same length for a joint sweep; "
                f"got lengths {dict(zip(paths, lengths))}"
            )
        paired_values = [dict(zip(paths, combo)) for combo in zip(*all_values)]
        items = []
        for value_dict in paired_values:
            sweep_config = deepcopy(base_config)
            for path, val in value_dict.items():
                set_path_value(sweep_config, path, val)
            run, result = run_and_load(sweep_config)
            items.append({"value": value_dict, "config": sweep_config, "run": run, "result": result})
        sweep = {
            "path": sweep_path,
            "values": paired_values,
            "items": items,
            "paramset": base_config.get("paramset"),
        }
    else:
        if values is None:
            raise ValueError("values must be provided for single-axis sweeps")
        items = []
        for value in values:
            sweep_config = deepcopy(base_config)
            set_path_value(sweep_config, sweep_path, value)
            run, result = run_and_load(sweep_config)
            items.append({"value": value, "config": sweep_config, "run": run, "result": result})
        sweep = {
            "path": sweep_path,
            "values": list(values),
            "items": items,
            "paramset": base_config.get("paramset"),
        }

    save_sweep(sweep)
    return sweep


def run_grid_sweep(
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
) -> dict[str, Any]:
    """Run every combination (cartesian product) of the provided parameter grid.

    Example::

        sweep = run_grid_sweep(config, {'gaba_gmax': [0, 1, 2], 'gap_mc': [16, 32]})
        # 6 runs: (0,16), (0,32), (1,16), (1,32), (2,16), (2,32)

    Items are ordered row-major (first parameter varies slowest).  Each item's
    ``value`` is a ``{path: value}`` dict, matching the joint-sweep convention.
    """
    from itertools import product as _product

    base_config = build_run_config(**deepcopy(base_config))
    paths = list(param_grid.keys())
    value_lists = list(param_grid.values())
    items = []
    all_values = []
    for combo in _product(*value_lists):
        value_dict = dict(zip(paths, combo))
        sweep_config = deepcopy(base_config)
        for path, val in value_dict.items():
            set_path_value(sweep_config, path, val)
        run, result = run_and_load(sweep_config)
        items.append({"value": value_dict, "config": sweep_config, "run": run, "result": result})
        all_values.append(value_dict)

    sweep = {
        "path": param_grid,
        "values": all_values,
        "items": items,
        "paramset": base_config.get("paramset"),
        "grid": {p: list(v) for p, v in param_grid.items()},
    }
    save_sweep(sweep)
    return sweep


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


class LazyResult(dict):
    """Result dict that loads selected heavy payloads on first access."""

    def __init__(self, *args: Any, lazy_loaders: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lazy_loaders = dict(lazy_loaders or {})

    def _ensure_loaded(self, key: str) -> None:
        if key not in self._lazy_loaders:
            return
        loader = self._lazy_loaders.pop(key)
        _progress_write(f"[OBGPU load] Lazy-loading {key}...")
        started = time.perf_counter()
        value = loader()
        dict.__setitem__(self, key, value)
        elapsed_s = time.perf_counter() - started
        _progress_write(f"[OBGPU load] Loaded {key} in {elapsed_s:.1f}s")

    def __getitem__(self, key: str) -> Any:
        self._ensure_loaded(key)
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._lazy_loaders:
            self._ensure_loaded(key)
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or key in self._lazy_loaders


def load_result(
    run_or_dir: RunRecord | str | Path,
    *,
    lazy_soma_vs: bool = True,
) -> dict[str, Any]:
    """Load the standard saved outputs for a notebook run directory."""
    result_dir = Path(run_or_dir.result_dir if isinstance(run_or_dir, RunRecord) else run_or_dir)
    summary = _read_json_if_present(result_dir / "summary.json")
    run_info = _read_json_if_present(result_dir / "run_info.json")

    result = LazyResult({
        "result_dir": result_dir,
        "summary": summary,
        "run_info": run_info,
        "input_times": [],
        "soma_vs": [],
        "gc_output_events": [],
        "lfp_t": np.array([]),
        "lfp": np.array([]),
    })

    load_plan: list[tuple[str, Path]] = []
    input_path = result_dir / "input_times.pkl"
    if input_path.exists():
        load_plan.append(("input_times", input_path))
    soma_path = result_dir / "soma_vs.pkl"
    if soma_path.exists() and not lazy_soma_vs:
        load_plan.append(("soma_vs", soma_path))
    gc_output_path = result_dir / "gc_output_events.pkl"
    if gc_output_path.exists():
        load_plan.append(("gc_output_events", gc_output_path))
    lfp_path = result_dir / "lfp.pkl"
    if lfp_path.exists():
        load_plan.append(("lfp", lfp_path))

    total_bytes = sum(path.stat().st_size for _key, path in load_plan)
    loaded_bytes = 0
    progress = _ProgressBar(total=total_bytes, desc="[OBGPU load] Load result files", unit="B", unit_scale=True)
    if load_plan:
        _progress_write(
            f"[OBGPU load] Loading {len(load_plan)} local result files ({_format_bytes(total_bytes)})...",
        )

    for index, (key, path) in enumerate(load_plan, start=1):
        file_size = path.stat().st_size
        _progress_write(
            f"[OBGPU load] Loading {index}/{len(load_plan)}: {path.name} ({_format_bytes(file_size)})",
        )
        started = time.perf_counter()
        loaded = load_pickle(path)
        elapsed_s = time.perf_counter() - started
        if key == "lfp":
            lfp_t, lfp = loaded
            result["lfp_t"] = np.asarray(lfp_t, dtype=float)
            result["lfp"] = np.asarray(lfp, dtype=float)
        else:
            result[key] = loaded
        loaded_bytes += file_size
        progress.update_to(loaded_bytes)
        _progress_write(
            f"[OBGPU load] {_render_progress_bar(loaded_bytes, total_bytes)} "
            f"{_format_bytes(loaded_bytes)} / {_format_bytes(total_bytes)} "
            f"(loaded {path.name} in {elapsed_s:.1f}s)",
        )
    progress.close()

    if soma_path.exists() and lazy_soma_vs:
        result["soma_vs_file"] = soma_path
        result._lazy_loaders["soma_vs"] = lambda path=soma_path: load_pickle(path)
        _progress_write(
            f"[OBGPU load] Deferred soma traces ({_format_bytes(soma_path.stat().st_size)}) until result['soma_vs'] is accessed."
        )

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
    print("[OBGPU load] Starting simulation run...", flush=True)
    run = run_simulation(config)
    print(f"[OBGPU load] Simulation complete. Loading results from {run.result_dir}...", flush=True)
    result = load_result(run)
    print("[OBGPU load] Result load complete.", flush=True)
    return run, result


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
        "ssh_binary",
        "ssh_options",
        "ssh_multiplex",
        "ssh_allow_interactive_auth",
        "ssh_keepalive_s",
        "ssh_control_path",
        "ssh_control_persist_s",
        "rsync_binary",
        "rsync_options",
    ]
    snapshot = {key: _json_ready(config.get(key)) for key in runtime_keys if key in config}
    snapshot["resolved_execution_mode"] = _json_ready(_resolve_execution_mode(config))
    return snapshot


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


DEFAULT_HFO_BANDS = {
    "hfo_80_130": (80.0, 130.0),
    "hfo_130_180": (130.0, 180.0),
}


def compute_band_power_summary(
    signal_t: np.ndarray | list[float],
    signal_y: np.ndarray | list[float],
    *,
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float | None = 0.1,
    nperseg: int | None = None,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
) -> dict[str, Any]:
    """Compute integrated Welch band powers for HFO-style summaries."""
    bands = dict(bands or DEFAULT_HFO_BANDS)
    t, y = uniform_trace(signal_t, signal_y, dt_ms=dt_ms)
    if len(t) < 4:
        return {
            "freqs": np.array([]),
            "psd": np.array([]),
            "band_power": {name: 0.0 for name in bands},
            "relative_band_power": {name: 0.0 for name in bands},
            "relative_band": relative_band,
        }

    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    fs_hz = 1000.0 / float(np.median(np.diff(t)))
    if nperseg is None:
        nperseg = min(2048, len(y))
    else:
        nperseg = min(int(nperseg), len(y))
    freqs, psd = welch(y, fs=fs_hz, nperseg=nperseg)

    if relative_band is None:
        denominator = float(np.trapz(psd, freqs))
    else:
        relative_mask = (freqs >= relative_band[0]) & (freqs <= relative_band[1])
        denominator = float(np.trapz(psd[relative_mask], freqs[relative_mask])) if np.any(relative_mask) else 0.0

    band_power = {}
    relative_power = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= float(lo)) & (freqs <= float(hi))
        power_value = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        band_power[name] = power_value
        relative_power[name] = power_value / denominator if denominator > 0 else 0.0

    return {
        "freqs": freqs,
        "psd": psd,
        "band_power": band_power,
        "relative_band_power": relative_power,
        "relative_band": relative_band,
    }


def compute_hfo_power_summary(
    result: dict[str, Any],
    *,
    signal: str = "lfp",
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float = 0.1,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
) -> dict[str, Any]:
    """Compute HFO band-power metrics for a named saved signal."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    summary = compute_band_power_summary(
        signal_t,
        signal_y,
        bands=bands,
        dt_ms=dt_ms,
        relative_band=relative_band,
    )
    summary["signal"] = signal
    return summary


def compute_spike_phase_locking(
    result: dict[str, Any],
    *,
    signal: str = "lfp",
    band: tuple[float, float] = (80.0, 130.0),
    cell_types: tuple[str, ...] | list[str] = ("MC", "TC"),
    threshold: float | None = None,
    dt_ms: float = 0.1,
) -> dict[str, Any]:
    """Measure soma-spike phase locking to a band-passed LFP-like signal."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    if len(signal_t) < 4:
        return {
            "signal": signal,
            "band": band,
            "cell_types": list(cell_types),
            "n_spikes": 0,
            "vector_strength": 0.0,
            "mean_phase_rad": np.nan,
            "per_cell": [],
        }

    fs_hz = 1000.0 / float(np.median(np.diff(signal_t)))
    bandpassed = butter_bandpass_filter(signal_y, band[0], band[1], fs_hz, order=4)
    phase = np.angle(hilbert(bandpassed))
    unwrapped_phase = np.unwrap(phase)
    allowed_types = tuple(str(cell_type) for cell_type in cell_types)

    all_vectors = []
    per_cell = []
    for label, t, v in result["soma_vs"]:
        if not label.startswith(allowed_types):
            continue
        spikes = detect_spikes(t, v, threshold=threshold)
        spikes = spikes[(spikes >= signal_t[0]) & (spikes <= signal_t[-1])]
        if len(spikes) == 0:
            continue
        spike_phase = np.angle(np.exp(1j * np.interp(spikes, signal_t, unwrapped_phase)))
        vectors = np.exp(1j * spike_phase)
        cell_vector = np.mean(vectors)
        per_cell.append(
            {
                "label": label,
                "n_spikes": int(len(spikes)),
                "vector_strength": float(np.abs(cell_vector)),
                "mean_phase_rad": float(np.angle(cell_vector)),
            }
        )
        all_vectors.append(vectors)

    if all_vectors:
        vectors = np.concatenate(all_vectors)
        mean_vector = np.mean(vectors)
        vector_strength = float(np.abs(mean_vector))
        mean_phase = float(np.angle(mean_vector))
        n_spikes = int(len(vectors))
    else:
        vector_strength = 0.0
        mean_phase = np.nan
        n_spikes = 0

    return {
        "signal": signal,
        "band": tuple(float(value) for value in band),
        "cell_types": list(cell_types),
        "n_spikes": n_spikes,
        "vector_strength": vector_strength,
        "mean_phase_rad": mean_phase,
        "per_cell": per_cell,
    }


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


def _adaptive_spike_peak_floor(v: np.ndarray) -> float:
    """Estimate a conservative spike-height floor from one voltage trace."""
    finite_v = np.asarray(v, dtype=float)
    finite_v = finite_v[np.isfinite(finite_v)]
    if finite_v.size == 0:
        return np.inf
    baseline = float(np.percentile(finite_v, 5.0))
    upper = float(np.percentile(finite_v, 95.0))
    dynamic_span = max(0.0, upper - baseline)
    return baseline + max(20.0, 0.5 * dynamic_span)


def detect_spikes(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float | None = None,
    *,
    min_prominence_mv: float = 3.0,
    refractory_ms: float = 1.0,
) -> np.ndarray:
    """Detect spike peaks from a soma trace using prominence and a refractory window.

    The previous detector only looked for upward crossings of a fixed voltage
    level, which misses sustained suprathreshold limit cycles and spikes that
    peak below 0 mV. This version finds local maxima, applies a minimum
    prominence, and uses either an explicit peak threshold or an adaptive floor
    derived from the trace itself.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 3:
        return np.array([])

    finite_mask = np.isfinite(t) & np.isfinite(v)
    if not np.all(finite_mask):
        t = t[finite_mask]
        v = v[finite_mask]
    if len(t) < 3:
        return np.array([])

    dt_ms = float(np.median(np.diff(t)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        dt_ms = 0.1
    min_distance = max(1, int(round(float(refractory_ms) / dt_ms)))

    peaks, _properties = find_peaks(
        v,
        prominence=float(min_prominence_mv),
        distance=min_distance,
    )
    if len(peaks) == 0:
        return np.array([])

    peak_floor = float(threshold) if threshold is not None else _adaptive_spike_peak_floor(v)
    keep = v[peaks] >= peak_floor
    return t[peaks[keep]]


def calculate_instantaneous_frequency(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float | None = None,
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
    threshold: float | None = None,
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
    threshold: float | None = None,
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


def plot_hfo_power_summary(
    result: dict[str, Any],
    *,
    signal: str = "lfp",
    bands: dict[str, tuple[float, float]] | None = None,
    dt_ms: float = 0.1,
    relative_band: tuple[float, float] | None = (30.0, 250.0),
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot absolute and relative HFO band power for a named signal."""
    summary = compute_hfo_power_summary(
        result,
        signal=signal,
        bands=bands,
        dt_ms=dt_ms,
        relative_band=relative_band,
    )
    names = list(summary["band_power"].keys())
    absolute = [summary["band_power"][name] for name in names]
    relative = [summary["relative_band_power"][name] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    axes[0].bar(names, absolute, color="tab:blue")
    axes[0].set_title(f"{signal} HFO Band Power")
    axes[0].set_ylabel("Integrated PSD")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, relative, color="tab:green")
    axes[1].set_title("Relative Band Power")
    axes[1].set_ylabel("Fraction")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig, axes, summary


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


def _fig_to_rgb_array(fig: Any) -> np.ndarray:
    """Render a matplotlib figure to an H×W×3 uint8 numpy array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)


def animate_sweep(
    sweep: dict[str, Any],
    plot_fn: Any,
    figsize: tuple[float, float] = (12, 5),
    interval: int = 1000,
    title_fn: Any = None,
    close_frames: bool = True,
) -> animation.FuncAnimation:
    """Animate any plot function across a parameter sweep.

    ``plot_fn(result) -> matplotlib.Figure`` is called once per sweep item.
    The figure is rendered to a pixel array so *any* plotting code works —
    multi-panel layouts, seaborn, custom axes, etc.  No changes to existing
    plot functions are needed.

    Parameters
    ----------
    sweep:
        Dict returned by :func:`run_parameter_sweep` or :func:`run_grid_sweep`.
    plot_fn:
        Callable that accepts a result dict and returns (or leaves as current)
        a ``matplotlib.Figure``.  If it returns None, ``plt.gcf()`` is used.
    figsize:
        Size of the *display* figure used for the animation.  Does not affect
        the rendered frames (those use whatever size ``plot_fn`` creates).
    interval:
        Milliseconds between frames.
    title_fn:
        Optional ``title_fn(value) -> str`` for per-frame titles.  When None
        the title is taken from the sweep path and value.
    close_frames:
        When True (default), close each frame figure after rendering to avoid
        leaking matplotlib figures.

    Example
    -------
    ::

        anim = animate_sweep(
            sweep,
            lambda r: plot_lfp_overview(r, dt_ms=0.1),
        )
        gif = save_animation(anim, 'my_sweep', sweep=sweep)
    """
    frames_rgb: list[np.ndarray] = []
    frame_titles: list[str] = []

    for item in sweep["items"]:
        result = item["result"]
        value = item["value"]

        returned = plot_fn(result)
        fig = returned if returned is not None else plt.gcf()

        frames_rgb.append(_fig_to_rgb_array(fig))

        if title_fn is not None:
            frame_titles.append(str(title_fn(value)))
        else:
            path = sweep.get("path", "")
            if isinstance(path, dict):
                title = ", ".join(f"{k}={_format_sweep_value(v)}" for k, v in value.items())
            else:
                title = f"{path} = {_format_sweep_value(value)}"
            frame_titles.append(title)

        if close_frames:
            plt.close(fig)

    if not frames_rgb:
        raise ValueError("sweep has no items to animate")

    display_fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    display_fig.tight_layout(pad=0)
    im = ax.imshow(frames_rgb[0])
    title_obj = ax.set_title(frame_titles[0])

    def _update(i: int) -> list:
        im.set_data(frames_rgb[i])
        title_obj.set_text(frame_titles[i])
        return [im, title_obj]

    anim = animation.FuncAnimation(
        display_fig,
        _update,
        frames=len(frames_rgb),
        interval=interval,
        repeat=True,
    )
    plt.close(display_fig)
    return anim


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


SWEEPS_BASE = REPO_ROOT / "results" / "sweeps"


def save_sweep(
    sweep: dict[str, Any],
    name: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Persist a completed sweep to an organised directory tree.

    Creates::

        <base_dir>/<name>_<timestamp>/
            sweep_info.json
            runs/
                00_<val>/run_info.json  (copy of each run's run_info.json)
            animations/               (empty; filled by save_animation)
            figures/                  (empty; filled by save_figure)

    The sweep dict is updated in-place with ``sweep["sweep_dir"]``.
    """
    base_dir = Path(base_dir or SWEEPS_BASE)
    timestamp = make_timestamp()
    path_label = sweep.get("path", "sweep")
    if isinstance(path_label, dict):
        path_label = "_".join(str(k) for k in path_label.keys())
    auto_name = _safe_name(f"{path_label}_{timestamp}")
    sweep_dir = base_dir / (name or auto_name)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "animations").mkdir(exist_ok=True)
    (sweep_dir / "figures").mkdir(exist_ok=True)
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Write per-run pointers
    run_dirs = []
    for i, item in enumerate(sweep.get("items", [])):
        val = item.get("value")
        run = item.get("run")
        result = item.get("result")
        result_dir = None
        if run is not None and hasattr(run, "result_dir"):
            result_dir = Path(run.result_dir)
        elif isinstance(result, dict) and "result_dir" in result:
            result_dir = Path(result["result_dir"])

        val_str = _safe_name(str(val)) if val is not None else str(i)
        slot = runs_dir / f"{i:02d}_{val_str}"
        slot.mkdir(exist_ok=True)

        if result_dir is not None:
            # Write a small pointer file so load_sweep can find the original dir
            (slot / "result_dir.txt").write_text(str(result_dir))
            # Copy run_info.json if present for quick inspection
            src = result_dir / "run_info.json"
            if src.exists():
                import shutil as _shutil
                _shutil.copy2(src, slot / "run_info.json")

        run_dirs.append(str(result_dir) if result_dir else None)

    # Write sweep_info.json
    git_ref = None
    try:
        git_ref = _resolve_local_git_head()
    except Exception:
        pass

    sweep_info = {
        "path": sweep.get("path"),
        "values": [_json_ready(v) for v in sweep.get("values", [])],
        "paramset": sweep.get("paramset"),
        "timestamp": timestamp,
        "git_ref": git_ref,
        "run_dirs": run_dirs,
        "n_items": len(sweep.get("items", [])),
    }
    (sweep_dir / "sweep_info.json").write_text(
        json.dumps(sweep_info, indent=2, sort_keys=True)
    )
    sweep["sweep_dir"] = sweep_dir
    return sweep_dir


def load_sweep(path: str | Path) -> dict[str, Any]:
    """Reconstruct a sweep dict from a directory created by save_sweep.

    Results are loaded lazily (same as load_result) so re-animating old
    sweeps does not require loading all soma traces upfront.
    """
    sweep_dir = Path(path)
    info_path = sweep_dir / "sweep_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No sweep_info.json found in {sweep_dir}")

    info = json.loads(info_path.read_text())
    items = []
    runs_dir = sweep_dir / "runs"
    for i, (value, run_dir_str) in enumerate(
        zip(info.get("values", []), info.get("run_dirs", []))
    ):
        result = None
        if run_dir_str is not None:
            run_dir = Path(run_dir_str)
            if run_dir.exists():
                result = load_result(run_dir)
            else:
                # Try the pointer file in the runs/ slot
                slot = runs_dir / f"{i:02d}_{_safe_name(str(value))}"
                ptr = slot / "result_dir.txt"
                if ptr.exists():
                    alt = Path(ptr.read_text().strip())
                    if alt.exists():
                        result = load_result(alt)
        items.append({"value": value, "config": None, "run": None, "result": result})

    return {
        "path": info.get("path"),
        "values": info.get("values", []),
        "items": items,
        "sweep_dir": sweep_dir,
        "sweep_info": info,
        "paramset": info.get("paramset"),
    }


def list_sweeps(
    prefix: str | None = None,
    base_dir: str | Path | None = None,
) -> list[Path]:
    """Return saved sweep directories sorted from oldest to newest."""
    base_dir = Path(base_dir or SWEEPS_BASE)
    if not base_dir.exists():
        return []
    dirs = [
        d for d in sorted(base_dir.iterdir())
        if d.is_dir() and (d / "sweep_info.json").exists()
        and (prefix is None or d.name.startswith(prefix))
    ]
    return dirs


def save_animation(
    anim: animation.FuncAnimation,
    name: str,
    output_dir: str | Path | None = None,
    sweep: dict[str, Any] | None = None,
    fps: int = 2,
) -> Path:
    """Save an animation as a GIF and return the written path.

    When ``sweep`` is provided and has a ``sweep_dir``, the GIF is saved to
    ``sweep_dir/animations/`` automatically (``output_dir`` is ignored).
    """
    if output_dir is None and sweep is not None and "sweep_dir" in sweep:
        output_dir = Path(sweep["sweep_dir"]) / "animations"
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
    sweep: dict[str, Any] | None = None,
    dpi: int = 200,
    close: bool = False,
) -> Path:
    """Save a Matplotlib figure near a run directory or in a timestamped folder.

    When ``sweep`` is provided and has a ``sweep_dir``, the figure is saved to
    ``sweep_dir/figures/`` automatically (other location hints are ignored).
    """
    fig = fig or plt.gcf()

    if output_dir is None and sweep is not None and "sweep_dir" in sweep:
        output_dir = Path(sweep["sweep_dir"]) / "figures"
    elif output_dir is None and run_or_result is not None:
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
