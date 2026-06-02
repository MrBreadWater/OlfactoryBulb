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
import socket
import subprocess
import sys
import tempfile
import time
import builtins
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from getpass import getpass
from hashlib import sha1, sha256
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable, MutableMapping

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
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

tqdm = _tqdm_plain or _tqdm_notebook
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert, welch
from scipy.stats import gaussian_kde
from modify_model import (
    add_synaptic_connection,
    modify_synaptic_connection,
    perform_cell_type_swaps,
    build_synapse_map
)
from olfactorybulb.result_artifacts import (
    DEFAULT_SOMA_TRACE_DTYPE,
    DEFAULT_SOMA_TRACE_FORMAT,
    DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
    DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
    DEFAULT_SOMA_SPIKE_THRESHOLD_MV,
    SOMA_SPIKES_FILENAME_NPZ,
    SOMA_TRACE_FILENAME_NPZ,
    SOMA_TRACE_FILENAME_PKL,
    VOLTAGE_SUMMARY_FILENAME_NPZ,
    adaptive_soma_spike_peak_floor,
    detect_soma_spikes,
    find_soma_trace_artifact,
    load_saved_result_artifact,
    preferred_soma_trace_artifact_name,
    soma_trace_artifact_candidates,
)
from olfactorybulb.hfo_features import (
    apply_hfo_runtime_overrides,
    hfo_control_help,
    hfo_run_config_defaults,
)
from neuroinfra.remote.helper_bundle import (
    HelperBundleEntry,
    bundle_entries_by_path,
    helper_bundle_signature,
)
from neuroinfra.remote.helper_cache import (
    helper_cache_dir as _neuroinfra_helper_cache_dir,
    helper_cache_manifest_path as _neuroinfra_helper_cache_manifest_path,
    helper_cache_mkdir_targets as _neuroinfra_helper_cache_mkdir_targets,
    helper_cache_probe_command as _neuroinfra_helper_cache_probe_command,
    helper_cache_probe_matches as _neuroinfra_helper_cache_probe_matches,
    helper_cache_runtime_key as _neuroinfra_helper_cache_runtime_key,
    helper_cache_upload_payload as _neuroinfra_helper_cache_upload_payload,
)
from neuroinfra.remote.allocation_cache import (
    allocation_cache_key as _neuroinfra_allocation_cache_key,
    allocation_runtime_config as _neuroinfra_allocation_runtime_config,
    allocation_signature as _neuroinfra_allocation_signature,
)
from neuroinfra.remote.allocation_runtime import (
    RemoteAllocationRuntimeContext as _NeuroinfraRemoteAllocationRuntimeContext,
)
from neuroinfra.remote.slurm_state import (
    REMOTE_SLURM_TERMINAL_FAIL as _NEUROINFRA_REMOTE_SLURM_TERMINAL_FAIL,
    REMOTE_SLURM_TERMINAL_OK as _NEUROINFRA_REMOTE_SLURM_TERMINAL_OK,
    build_remote_cancel_command as _neuroinfra_build_remote_cancel_command,
    build_remote_preflight_command as _neuroinfra_build_remote_preflight_command,
    build_remote_result_listing_command as _neuroinfra_build_remote_result_listing_command,
    normalize_slurm_state as _neuroinfra_normalize_slurm_state,
    query_remote_slurm_job_state as _neuroinfra_query_remote_slurm_job_state,
    remote_preflight_cache_key as _neuroinfra_remote_preflight_cache_key,
    remote_status_has_artifacts as _neuroinfra_remote_status_has_artifacts,
    run_remote_preflight_cached as _neuroinfra_run_remote_preflight_cached,
)
from neuroinfra.remote.command_launch import (
    build_remote_python_file_command as _neuroinfra_build_remote_python_file_command,
    build_remote_python_inline_command as _neuroinfra_build_remote_python_inline_command,
    build_remote_touch_command as _neuroinfra_build_remote_touch_command,
    remote_helper_script_path as _neuroinfra_remote_helper_script_path,
    remote_python_exec_prefix as _neuroinfra_remote_python_exec_prefix,
)
from neuroinfra.remote.config import (
    build_remote_slurm_config as _neuroinfra_build_remote_slurm_config,
    connect_retry_backoff_s as _neuroinfra_connect_retry_backoff_s,
    connect_retry_count as _neuroinfra_connect_retry_count,
    heartbeat_timeout_s as _neuroinfra_heartbeat_timeout_s,
    poll_command_timeout_s as _neuroinfra_poll_command_timeout_s,
    remote_connection_key as _neuroinfra_remote_connection_key,
    require_remote_host as _neuroinfra_require_remote_host,
    resolve_remote_endpoint as _neuroinfra_resolve_remote_endpoint,
    ssh_command_timeout_s as _neuroinfra_ssh_command_timeout_s,
    ssh_exec_timeout_s as _neuroinfra_ssh_exec_timeout_s,
    ssh_upload_timeout_s as _neuroinfra_ssh_upload_timeout_s,
)
from neuroinfra.remote.notebook_runtime import (
    cache_prompt_response as _neuroinfra_cache_prompt_response,
    cached_prompt_responses as _neuroinfra_cached_prompt_responses,
    can_reconnect as _neuroinfra_can_reconnect,
    ensure_notebook_remote_runtime as _neuroinfra_ensure_notebook_remote_runtime,
    get_cached_prompt_response as _neuroinfra_get_cached_prompt_response,
    has_cached_auth as _neuroinfra_has_cached_auth,
    midrun_reauth_error as _neuroinfra_midrun_reauth_error,
    prompt_key as _neuroinfra_prompt_key,
    transport_is_usable as _neuroinfra_transport_is_usable,
)
from neuroinfra.remote.paramiko_transport import (
    ParamikoTransportContext as _NeuroinfraParamikoTransportContext,
    SSHCommandTimeoutError as _NeuroinfraSSHCommandTimeoutError,
    connect_error_is_retryable as _neuroinfra_connect_error_is_retryable,
)
from neuroinfra.remote.archive_stream import (
    build_remote_archive_command as _neuroinfra_build_remote_archive_command,
    build_remote_archive_probe_command as _neuroinfra_build_remote_archive_probe_command,
    build_remote_selected_archive_probe_command as _neuroinfra_build_remote_selected_archive_probe_command,
    build_remote_selected_stream_archive_command as _neuroinfra_build_remote_selected_stream_archive_command,
    build_remote_stream_archive_command as _neuroinfra_build_remote_stream_archive_command,
    build_remote_sweep_compact_stream_archive_command as _neuroinfra_build_remote_sweep_compact_stream_archive_command,
    local_archive_decompress_command as _neuroinfra_local_archive_decompress_command,
    paramiko_channel_stream_finished as _neuroinfra_paramiko_channel_stream_finished,
)
from neuroinfra.remote.sftp_sync import (
    SFTPSyncHooks as _NeuroinfraSFTPSyncHooks,
    sftp_copy_files as _neuroinfra_sftp_copy_files,
    sftp_copy_tree as _neuroinfra_sftp_copy_tree,
)
from neuroinfra.remote.stream_sync import (
    ParamikoStreamSyncHooks as _NeuroinfraParamikoStreamSyncHooks,
    probe_selected_sync_files as _neuroinfra_probe_selected_sync_files,
    stream_archive_to_local as _neuroinfra_stream_archive_to_local,
    stream_archive_to_local_dir as _neuroinfra_stream_archive_to_local_dir,
    stream_file_to_local_path as _neuroinfra_stream_file_to_local_path,
)
from neuroinfra.remote.result_sync import (
    RemoteResultSyncHooks as _NeuroinfraRemoteResultSyncHooks,
    combine_sync_attempt_stderr as _neuroinfra_combine_sync_attempt_stderr,
    sync_remote_result_dir as _neuroinfra_sync_remote_result_dir,
    sync_remote_result_dir_resilient as _neuroinfra_sync_remote_result_dir_resilient,
)
from neuroinfra.remote.status_poll import (
    RemoteJSONPollHooks as _NeuroinfraRemoteJSONPollHooks,
    poll_remote_json_status as _neuroinfra_poll_remote_json_status,
)
from neuroinfra.remote.run_artifacts import (
    RemoteRunArtifactHooks as _NeuroinfraRemoteRunArtifactHooks,
    finalize_remote_run_artifacts as _neuroinfra_finalize_remote_run_artifacts,
)
from neuroinfra.remote.run_monitor import (
    RemoteRunMonitorHooks as _NeuroinfraRemoteRunMonitorHooks,
    monitor_remote_run as _neuroinfra_monitor_remote_run,
)
from neuroinfra.remote.sweep_monitor import (
    RemoteSweepMonitorHooks as _NeuroinfraRemoteSweepMonitorHooks,
    monitor_remote_sweep as _neuroinfra_monitor_remote_sweep,
)
from neuroinfra.remote.sweep_artifacts import (
    RemoteSweepArtifactHooks as _NeuroinfraRemoteSweepArtifactHooks,
    finalize_remote_sweep_artifacts as _neuroinfra_finalize_remote_sweep_artifacts,
)
from neuroinfra.remote.deferred_artifacts import (
    DeferredArtifactSyncHooks as _NeuroinfraDeferredArtifactSyncHooks,
    sync_deferred_remote_artifact as _neuroinfra_sync_deferred_remote_artifact,
    sync_deferred_remote_artifact_direct as _neuroinfra_sync_deferred_remote_artifact_direct,
)
from neuroinfra.remote.slurm_launch import (
    build_allocation_discovery_command as _neuroinfra_build_allocation_discovery_command,
    build_cleanup_stale_allocations_argv as _neuroinfra_build_cleanup_stale_allocations_argv,
    build_poll_sol_run_argv as _neuroinfra_build_poll_sol_run_argv,
    build_remote_helper_launch_command as _neuroinfra_build_remote_helper_launch_command,
    build_submit_slurm_allocation_argv as _neuroinfra_build_submit_slurm_allocation_argv,
    build_submit_sol_run_argv as _neuroinfra_build_submit_sol_run_argv,
)
from neuroinfra.remote.git_sync import (
    build_remote_git_bundle_fetch_command as _neuroinfra_build_remote_git_bundle_fetch_command,
    build_remote_git_repo_probe_command as _neuroinfra_build_remote_git_repo_probe_command,
    create_git_bundle_for_commit as _neuroinfra_create_git_bundle_for_commit,
    git_merged_ref_shas as _neuroinfra_git_merged_ref_shas,
    git_ref_is_ancestor as _neuroinfra_git_ref_is_ancestor,
    git_ref_points_to_commit as _neuroinfra_git_ref_points_to_commit,
    git_rev_parse as _neuroinfra_git_rev_parse,
    local_git_sync_base_candidates as _neuroinfra_local_git_sync_base_candidates,
    remote_notebook_tracking_ref_for_source as _neuroinfra_remote_notebook_tracking_ref_for_source,
    resolve_local_git_branch as _neuroinfra_resolve_local_git_branch,
    resolve_local_git_head as _neuroinfra_resolve_local_git_head,
    resolve_local_git_upstream_ref as _neuroinfra_resolve_local_git_upstream_ref,
)
from neuroinfra.artifacts.loading import (
    ArtifactLoadingHooks as _NeuroinfraArtifactLoadingHooks,
    LazyResult as _NeuroinfraLazyResult,
    load_local_artifact_plan as _neuroinfra_load_local_artifact_plan,
)
from neuroinfra.artifacts.result_view import (
    ResultArtifactBinding as _NeuroinfraResultArtifactBinding,
    ResultFieldSpec as _NeuroinfraResultFieldSpec,
    ResultViewHooks as _NeuroinfraResultViewHooks,
    ResultViewSchema as _NeuroinfraResultViewSchema,
    attach_lazy_artifact_loaders as _neuroinfra_attach_lazy_artifact_loaders,
    plan_result_view as _neuroinfra_plan_result_view,
)
from neuroinfra.analysis.signals import (
    ResultSignalProvider as _NeuroinfraResultSignalProvider,
    list_available_result_signals as _neuroinfra_list_available_result_signals,
    resolve_result_signal as _neuroinfra_resolve_result_signal,
)
from neuroinfra.analysis.catalog import (
    CategoryCatalogHooks as _NeuroinfraCategoryCatalogHooks,
    group_rows_by_category as _neuroinfra_group_rows_by_category,
    list_available_categories as _neuroinfra_list_available_categories,
    list_unique_labels as _neuroinfra_list_unique_labels,
)
from neuroinfra.analysis.overview import (
    build_result_overview as _neuroinfra_build_result_overview,
    build_result_overview_context as _neuroinfra_build_result_overview_context,
    first_result_file_metadata as _neuroinfra_first_result_file_metadata,
    metadata_value_or_result_length as _neuroinfra_metadata_value_or_result_length,
)
from neuroinfra.analysis.events import (
    binned_event_rate as _neuroinfra_binned_event_rate,
    calculate_event_frequency as _neuroinfra_calculate_event_frequency,
    ensure_raster_axis as _neuroinfra_ensure_raster_axis,
    fit_raster_labels as _neuroinfra_fit_raster_labels,
    plot_event_raster_rows as _neuroinfra_plot_event_raster_rows,
    rate_series_label as _neuroinfra_rate_series_label,
    recommended_raster_fontsize as _neuroinfra_recommended_raster_fontsize,
    recommended_raster_height as _neuroinfra_recommended_raster_height,
    smooth_rate_series as _neuroinfra_smooth_rate_series,
    style_raster_axis as _neuroinfra_style_raster_axis,
)
from neuroinfra.analysis.plotting import (
    plot_band_power_summary as _neuroinfra_plot_band_power_summary,
    plot_named_time_series as _neuroinfra_plot_named_time_series,
    plot_time_frequency_map as _neuroinfra_plot_time_frequency_map,
    plot_time_series as _neuroinfra_plot_time_series,
)
from neuroinfra.analysis.sweeps import (
    SweepPlotSpec,
    animate_sweep as _neuroinfra_animate_sweep,
    animate_sweep_plots as _neuroinfra_animate_sweep_plots,
    build_sweep_plot_callable as _neuroinfra_build_sweep_plot_callable,
    compose_sweep_display_frame as _neuroinfra_compose_sweep_display_frame,
    default_sweep_animation_worker_count as _neuroinfra_default_sweep_animation_worker_count,
    describe_unavailable_sweep_item as _neuroinfra_describe_unavailable_sweep_item,
    extract_figure_from_plot_result as _neuroinfra_extract_figure_from_plot_result,
    fig_to_rgb_array as _neuroinfra_fig_to_rgb_array,
    format_sweep_frame_title as _neuroinfra_format_sweep_frame_title,
    iter_parallel_sweep_display_frames as _neuroinfra_iter_parallel_sweep_display_frames,
    iter_sweep_animation_frames as _neuroinfra_iter_sweep_animation_frames,
    is_deprecated_sweep_animation_spec as _neuroinfra_is_deprecated_sweep_animation_spec,
    list_sweeps as _neuroinfra_list_sweeps,
    load_sweep as _neuroinfra_load_sweep,
    make_sweep_placeholder_figure as _neuroinfra_make_sweep_placeholder_figure,
    make_sweep_plot_spec as _neuroinfra_make_sweep_plot_spec,
    normalize_sweep_plot_spec as _neuroinfra_normalize_sweep_plot_spec,
    render_sweep_frame as _neuroinfra_render_sweep_frame,
    save_animation as _neuroinfra_save_animation,
    save_sweep as _neuroinfra_save_sweep,
    save_sweep_animation_stream as _neuroinfra_save_sweep_animation_stream,
    write_sweep_info as _neuroinfra_write_sweep_info,
)
from neuroinfra.analysis.spectral import (
    DEFAULT_HFO_BANDS,
    butter_bandpass_filter,
    compute_band_power_summary,
    compute_spectrogram,
    compute_wavelet_band_power,
    compute_wavelet_map,
    normalize_time_modulus as _normalize_time_modulus,
    uniform_trace,
)

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "benchmarks" / "benchmark_ob.py"
DEFAULT_RESULTS_BASE = REPO_ROOT / "results" / "notebook_runs"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
PRIMARY_CELL_TYPE_ORDER = ("MC", "TC", "GC", "EPLI")
PLOT_DISPLAY_CELL_GROUPS = ("MT", "GC", "EPLI", "other")
DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ = (130.0, 230.0)
DEFAULT_PSD_TEMPLATE_FLOOR = 1e-5
CELL_TYPE_ALIASES = {
    # The optional EPLI population currently uses the synthetic PVCRH_FSI1
    # model class. Saved section labels expose that class name, but notebook
    # summaries should report the runtime population, not a second cell type.
    "PVCRH": "EPLI",
}
CELL_TYPE_COLORS = {
    "MC": "tab:blue",
    "TC": "tab:red",
    "GC": "tab:orange",
    "EPLI": "tab:green",
    "other": "tab:gray",
}
CONTROL_HELP = {
    "mode": "Use 'fast' for 1-rank exploration or 'parity' for 2-rank comparison runs.",
    "nranks": "MPI rank count for the run. 1 is faster on this machine",
    "tstop_ms": "Simulation duration in ms. Use None to keep the paramset default.",
    "sim_dt_ms": "Requested simulation dt in ms.",
    "recording_period_ms": "Saved sample period for LFP and soma traces.",
    "soma_trace_format": "Saved soma-trace artifact format. 'npz' stores compressed array-native traces; 'pkl' keeps the legacy Python-object format.",
    "soma_trace_dtype": "Saved soma-trace numeric dtype. Use 'float32' by default; 'int16' stores lossy per-trace linear-quantized voltages.",
    "soma_spike_threshold_mv": "Optional absolute soma spike peak threshold in mV. None uses an adaptive per-trace peak floor.",
    "soma_spike_min_prominence_mv": "Minimum peak prominence in mV for runtime soma spike detection.",
    "soma_spike_refractory_ms": "Minimum inter-peak spacing in ms for runtime soma spike detection.",
    "save_soma_traces": "When False, save soma spike detections without writing raw soma voltage traces.",
    "save_voltage_summary": "When False, skip voltage-summary arrays after soma spike detection.",
    "lfp_electrode_location": "Probe location as [x, y, z] in microns.",
    "lfp_include_cell_types": "Optional LFP source filter. When set, compute LFP only from these cell types.",
    "lfp_exclude_cell_types": "Optional LFP source filter. Cell types to exclude from LFP without changing the circuit.",
    "rnd_seed": "Random seed for odor input generation.",
    "record_from_somas": "Which cell types to record from, e.g. ['MC', 'TC', 'GC']. When EPLIs are enabled their configured cell type is appended automatically unless already present.",
    "enable_epl_interneurons": "Enable the opt-in EPLI slice population when the slice export and paramset support it.",
    "max_epl_interneurons": "Maximum EPLI cells to instantiate. Must be > 0 together with enable_epl_interneurons.",
    "epl_interneuron_cell_type": "Runtime cell-type label for the opt-in EPLI population, e.g. 'EPLI'.",
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
    "enable_reciprocal_synapses": "Toggle GC<->MC/TC reciprocal synapses.",
    "extra_overrides": "Any raw paramset overrides not exposed above.",
    "spectrogram_signal": "Signal for spectrogram plots, e.g. 'lfp', 'mean_MC_voltage', or 'MC5[0].soma'.",
    "spectrogram_max_freq_hz": "Maximum frequency (Hz) shown in spectrogram output.",
    "spectrogram_nperseg": "Spectrogram STFT window length in samples.",
    "spectrogram_noverlap": "Number of samples of overlap between spectrogram windows.",
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
    "sweep_engine": "Sweep execution engine: 'auto', 'legacy', or 'remote_batch'.",
    "sweep_parallelism": "Maximum concurrent sweep items inside one remote batch job. None/0 uses an automatic best-effort choice.",
    "sweep_sync_live": "When True, sync completed remote sweep items back locally while later items are still running. Remote config builders default this to False so transfers do not destabilize active jobs.",
    "sweep_live_sync_max_items_per_poll": "Maximum completed sweep items to live-sync per poll. This prevents local transfer work from starving the remote heartbeat.",
    "sweep_sync_soma_vs": "When True, remote sweep sync includes raw soma voltage traces. Defaults to False so sweeps sync compact spike/LFP artifacts only.",
    "sweep_sync_voltage_summary": "When True, remote sweep sync includes voltage-summary arrays. Defaults to False unless a configured sweep analysis signal requires mean voltage.",
    "remote_poll_interval_s": "Polling interval in seconds for remote Slurm jobs.",
    "remote_log_poll_interval_s": "How often to do heavier remote log-tail and sacct reconciliation polls while still updating progress every remote_poll_interval_s seconds.",
    "remote_live_status": "When True, print live remote Slurm state updates in the notebook while polling.",
    "remote_live_logs": "When True, stream remote bootstrap/stdout/stderr/slurm log updates into the notebook while polling.",
    "remote_heartbeat_timeout_s": "Remote Slurm watchdog timeout in seconds. Notebook-managed jobs and reusable allocations self-terminate if the notebook stops refreshing their heartbeat for longer than this.",
    "remote_ssh_command_timeout_s": "Timeout in seconds for one notebook-managed Paramiko shell command. None or <=0 disables the timeout.",
    "remote_ssh_exec_timeout_s": "Timeout in seconds for Paramiko exec_command acknowledgement before the command output phase starts.",
    "remote_ssh_upload_timeout_s": "Timeout in seconds for Paramiko shell-backed file upload send/write operations.",
    "remote_poll_command_timeout_s": "Timeout in seconds for lightweight remote status-poll shell commands. Keeps a stale SSH channel from freezing an active notebook worker.",
    "remote_poll_json_retries": "Number of times to retry a successful remote status poll that returns empty or malformed JSON before failing the run.",
    "remote_cleanup_stale_allocations": "When True, cancel stale or pre-heartbeat notebook-managed reusable allocations on the remote before submitting a new run.",
    "remote_sync_compress": "When True, compress the remote result directory before downloading it back to the notebook.",
    "remote_defer_soma_vs_sync": "Deprecated. Raw soma traces are synced and loaded with the main result payload; stale True values are ignored.",
    "remote_preserve_paramiko_session": "When True, never silently open a fresh Paramiko login after one notebook session has already authenticated; fail closed instead of re-prompting mid-run.",
    "remote_allow_paramiko_reauth": "When True, allow a fresh Paramiko login after a prior session authenticated. Defaults to False so dropped notebook SSH sessions fail closed instead of re-prompting mid-run.",
    "slurm_partition": "Optional Slurm partition for remote submission. Set it explicitly when needed; None omits --partition entirely.",
    "slurm_account": "Optional Slurm account for remote submission.",
    "slurm_time": "Optional Slurm walltime, e.g. '02:00:00'.",
    "slurm_gpus": "Optional GPU count requested from Slurm.",
    "slurm_cpus_per_task": "Optional CPU count requested per Slurm task.",
    "slurm_step_ntasks": "When reusing a remote allocation, number of tasks to request per benchmark step. Defaults to nranks for remote MPI runs.",
    "slurm_mem": "Optional Slurm memory request, e.g. '32G'.",
    "slurm_extra_args": "Optional extra sbatch arguments passed as raw strings.",
    "ssh_options": "Extra SSH options, e.g. ['-J', 'jumphost'].",
    "ssh_transport": "Deprecated compatibility option. Remote notebook runs now always use Paramiko; 'auto' and 'paramiko' are accepted.",
    "ssh_keepalive_s": "Paramiko keepalive interval in seconds for notebook-managed SSH sessions. Higher values reduce background traffic; lower values make idle sessions less likely to die between runs.",
    "ssh_connect_retries": "How many times notebook-managed Paramiko should retry a fresh SSH connect/handshake before failing. Helps absorb transient banner/read errors on reused notebook tunnels.",
    "ssh_connect_retry_backoff_s": "Base backoff in seconds between Paramiko reconnect attempts. Later attempts use small multiples of this delay.",
    "add_connections": "Add new connections between existing neurons.",
    "modify_connections": "Modify the synaptic weight between two specific neurons.",
    "swap_cell_types": "A list of cells to swap to another cell type."
}
CONTROL_HELP.update(hfo_control_help())


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


@dataclass
class FrequencyPlotConfig:
    """Shared rendering controls for spike/event frequency distribution plots."""

    modulus: float | None = 1e8
    max_freq_hz: float = 200.0
    kde_bw_method: str | float = "scott"
    kde1d_engine: str = "histogram"
    kde_bw_x: float = 0.15
    kde_bw_y: float = 0.2
    kde2d_engine: str = "histogram"
    kde_resolution_t: int = 100
    kde_resolution_f: int = 100
    kde_f_resolution: int = 1600
    num_time_bins: int = 32
    bin_alpha: float = 0.5
    kde_cmap: str = "inferno"
    dot_size: float = 5.0
    dot_alpha: float = 0.2
    strip_plot: bool = True
    guide_line_spacing_ms: float = 0.0


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
    """Render a compact fallback progress bar for non-tqdm environments."""
    if total <= 0:
        return "[" + ("?" * width) + "]"
    progress = max(0.0, min(float(current) / float(total), 1.0))
    filled = int(round(progress * width))
    return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"


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
_NOTEBOOK_RUNTIME = _neuroinfra_ensure_notebook_remote_runtime(builtins._OBGPU_NOTEBOOK_RUNTIME)
_LIVE_PARAMIKO_CONNECTIONS: dict[str, Any] = _NOTEBOOK_RUNTIME["paramiko_connections"]
_LIVE_PARAMIKO_AUTHENTICATED_KEYS: set[str] = _NOTEBOOK_RUNTIME["paramiko_authenticated_keys"]
_LIVE_PARAMIKO_PROMPT_CACHE: dict[str, dict[str, str]] = _NOTEBOOK_RUNTIME["paramiko_prompt_cache"]
_LIVE_SLURM_ALLOCATIONS: dict[str, Any] = _NOTEBOOK_RUNTIME["slurm_allocations"]
_LIVE_REMOTE_GIT_REFS: dict[str, set[str]] = _NOTEBOOK_RUNTIME["remote_git_refs"]
_LIVE_REMOTE_HELPER_CACHES: dict[str, Any] = _NOTEBOOK_RUNTIME["remote_helper_caches"]
_LIVE_REMOTE_PREFLIGHTS: dict[str, Any] = _NOTEBOOK_RUNTIME["remote_preflight"]
_LIVE_REMOTE_STALE_CLEANUPS: dict[str, Any] = _NOTEBOOK_RUNTIME["remote_stale_cleanup"]


class _SSHCommandTimeoutError(_NeuroinfraSSHCommandTimeoutError):
    """Backward-compatible local alias for the extracted timeout surface."""


def _slurm_allocation_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the SSH/runtime subset needed to rediscover or cancel one allocation."""
    return _neuroinfra_allocation_runtime_config(config)


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
    """Return a generic remote Slurm config for notebook-driven runs.

    Slurm arguments are only emitted when explicitly provided.
    """
    _warn_remote_execution_mode_reset()
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
            None
            if config.get("soma_spike_threshold_mv") is None
            else float(config["soma_spike_threshold_mv"])
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


def available_controls() -> dict[str, str]:
    """Return the notebook control catalog."""
    return dict(CONTROL_HELP)


def print_available_controls() -> None:
    """Pretty-print the notebook control catalog."""
    print(json.dumps(available_controls(), indent=2, sort_keys=True))


def _benchmark_param_overrides_payload(config: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return benchmark overrides plus any sidecar input-spec path."""
    param_overrides = build_param_overrides(config)
    input_spec_file = param_overrides.pop("_input_spec_file", None)
    return param_overrides, input_spec_file


def _write_benchmark_overrides_file(path: str | Path, param_overrides: dict[str, Any]) -> Path:
    """Write benchmark overrides to a sidecar file so argv stays compact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(param_overrides), indent=2, sort_keys=True))
    return path


def _remote_benchmark_overrides_path(config: dict[str, Any], label: str) -> PurePosixPath:
    """Return the remote sidecar path used for one run's benchmark overrides."""
    return _remote_results_root(config) / ".obgpu-wrapper" / str(label) / "overrides.json"

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
    overrides_file: str | Path | None = None,
    param_overrides: dict[str, Any] | None = None,
    input_spec_file: str | Path | None = None,
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

    if param_overrides is None:
        param_overrides, discovered_input_spec_file = _benchmark_param_overrides_payload(config)
        if input_spec_file is None:
            input_spec_file = discovered_input_spec_file
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
        ]
    )
    if overrides_file is None:
        command.extend(["--overrides-json", json.dumps(param_overrides, sort_keys=True)])
    else:
        command.extend(["--overrides-file", str(overrides_file)])
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


def _safe_sweep_path_label(path_value: Any) -> str:
    """Return a compact label component for one sweep path or path mapping."""
    import hashlib

    if isinstance(path_value, dict):
        raw = "_".join(str(key) for key in path_value.keys())
    elif isinstance(path_value, (list, tuple)):
        raw = "_".join(str(part) for part in path_value)
    else:
        raw = str(path_value)
    cleaned = _safe_name(raw.replace(".", "_"))
    if len(cleaned) > 64:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        prefix = cleaned[:24] if cleaned else "sweep"
        cleaned = f"{prefix}_{digest}"
    return cleaned or "sweep"


def _make_sweep_label(base_config: dict[str, Any], *, sweep_path: Any, timestamp: str) -> str:
    """Build the top-level label for one completed sweep."""
    sweep_config = deepcopy(base_config)
    prefix = str(sweep_config.get("label_prefix", "sweep"))
    sweep_config["label_prefix"] = f"{prefix}_{_safe_sweep_path_label(sweep_path)}_sweep"
    return make_label(sweep_config, timestamp=timestamp)


def _make_sweep_item_label(
    base_config: dict[str, Any],
    *,
    sweep_path: Any,
    timestamp: str,
    index: int,
) -> str:
    """Build one stable per-item label nested under a sweep."""
    base_label = _make_sweep_label(base_config, sweep_path=sweep_path, timestamp=timestamp)
    return f"{base_label}_{index:03d}"


def _requested_task_count_from_slurm_args(values: list[str] | tuple[str, ...] | None) -> int | None:
    """Best-effort parse of one total Slurm task count from extra args."""
    args = [str(value) for value in (values or [])]

    nodes: int | None = None
    ntasks_per_node: int | None = None

    for index, part in enumerate(args):
        if part in {"-n", "--ntasks"} and index + 1 < len(args):
            try:
                return max(int(args[index + 1]), 1)
            except ValueError:
                continue
        if part in {"-N", "--nodes", "--ntasks-per-node"} and index + 1 < len(args):
            try:
                value = max(int(args[index + 1]), 1)
            except ValueError:
                continue
            if part == "--ntasks-per-node":
                ntasks_per_node = value
                continue
            if part in {"-N", "--nodes"}:
                nodes = value
                continue
        if part.startswith("--nodes="):
            try:
                nodes = max(int(part.split("=", 1)[1]), 1)
            except ValueError:
                nodes = None
            continue
        if part.startswith("--ntasks-per-node="):
            try:
                ntasks_per_node = max(int(part.split("=", 1)[1]), 1)
            except ValueError:
                ntasks_per_node = None
            continue
        if part.startswith("--ntasks="):
            try:
                return max(int(part.split("=", 1)[1]), 1)
            except ValueError:
                continue
        if part.startswith("-n") and part != "-n":
            try:
                return max(int(part[2:]), 1)
            except ValueError:
                continue
        if part.startswith("-N") and part != "-N":
            try:
                nodes = max(int(part[2:]), 1)
            except ValueError:
                nodes = None

    if nodes is not None and ntasks_per_node is not None:
        return max(nodes * ntasks_per_node, 1)

    return None


def _requested_gpus_from_slurm_args(values: list[str] | tuple[str, ...] | None) -> int | None:
    """Best-effort parse of one total Slurm GPU count from extra args."""
    args = [str(value) for value in (values or [])]
    for index, part in enumerate(args):
        if part == "--gpus" and index + 1 < len(args):
            try:
                return max(int(args[index + 1]), 0)
            except ValueError:
                continue
        if part.startswith("--gpus="):
            raw = part.split("=", 1)[1]
            if ":" in raw:
                raw = raw.rsplit(":", 1)[-1]
            try:
                return max(int(raw), 0)
            except ValueError:
                continue
    return None


def _remote_sweep_parallelism(config: dict[str, Any], *, tasks_per_item: int) -> int:
    """Choose one best-effort concurrent item count for a remote sweep batch job."""
    explicit = config.get("sweep_parallelism")
    if explicit not in (None, "", 0):
        return max(int(explicit), 1)

    remote_mpi_exec = str(config.get("remote_mpi_exec") or default_remote_mpi_exec()).strip()
    launcher_head = shlex.split(remote_mpi_exec)[0] if remote_mpi_exec else ""

    # Prefer optimization profile / allocation metadata over launcher inference.
    total_tasks = config.get("optimizer_total_tasks")
    if total_tasks in (None, ""):
        total_tasks = config.get("slurm_total_tasks")

    if total_tasks in (None, ""):
        total_tasks = _requested_task_count_from_slurm_args(config.get("slurm_extra_args", []))

    # GPU sweeps are intentionally conservative unless the caller specifies otherwise.
    slurm_gpus = config.get("slurm_gpus")
    if slurm_gpus in (None, ""):
        slurm_gpus = _requested_gpus_from_slurm_args(config.get("slurm_extra_args", []))
    try:
        gpu_count = int(slurm_gpus) if slurm_gpus not in (None, "") else 0
    except (TypeError, ValueError):
        gpu_count = 0
    if gpu_count > 0:
        return 1

    if total_tasks is None and os.path.basename(launcher_head) != "srun":
        return 1

    if total_tasks is None:
        allocation_job_id = config.get("slurm_allocation_job_id")
        if allocation_job_id in (None, ""):
            return 1
        # Last-ditch fallback: use optimization profile size if we missed it above.
        fallback_total = config.get("optimizer_total_tasks")
        if fallback_total not in (None, ""):
            try:
                total_tasks = max(int(fallback_total), 1)
            except (TypeError, ValueError):
                return 1
        else:
            return 1

    try:
        total_tasks = int(total_tasks)
    except (TypeError, ValueError):
        return 1

    if total_tasks <= 0:
        return 1

    if tasks_per_item <= 0:
        return 1

    if total_tasks is None or tasks_per_item <= 0:
        return 1
    return max(total_tasks // max(tasks_per_item, 1), 1)


def _sweep_uses_remote_batch_engine(config: dict[str, Any]) -> bool:
    """Return whether sweeps should run through one remote batch job."""
    engine = str(config.get("sweep_engine", "auto")).strip().lower()
    if engine == "legacy":
        return False
    if engine == "remote_batch":
        return True
    return str(config.get("runner_backend", "local")) in {"sol_slurm", "slurm_remote"}


def _sweep_base_dir(config: dict[str, Any]) -> Path:
    """Return the local parent directory that stores grouped sweep outputs."""
    return Path(config.get("results_base", DEFAULT_RESULTS_BASE)) / "sweeps"


def _sweep_dir(config: dict[str, Any], sweep_label: str) -> Path:
    """Return the local directory for one grouped sweep."""
    return _sweep_base_dir(config) / str(sweep_label)


def _sweep_item_runs_dir(config: dict[str, Any], sweep_label: str) -> Path:
    """Return the local directory that stores actual per-item result folders."""
    return _sweep_dir(config, sweep_label) / "item_runs"


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


def _record_timing(timings: dict[str, float], key: str, started_at: float) -> None:
    """Accumulate one elapsed stage timing in seconds."""
    timings[key] = round(timings.get(key, 0.0) + (time.perf_counter() - started_at), 3)


def _timing_summary_text(timings: dict[str, float], *, limit: int = 6) -> str:
    """Format the slowest recorded stages into one compact log string."""
    items = [(key, float(value)) for key, value in timings.items() if float(value) > 0.0]
    if not items:
        return ""
    items.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(f"{key}={value:.2f}s" for key, value in items[: max(int(limit), 1)])


def _standard_result_artifact_sizes(result_dir: str | Path) -> dict[str, int]:
    """Return sizes for the standard notebook result artifacts that exist locally."""
    result_dir = Path(result_dir)
    filenames = (
        "input_times.pkl",
        "gc_output_events.pkl",
        "lfp.pkl",
        SOMA_SPIKES_FILENAME_NPZ,
        VOLTAGE_SUMMARY_FILENAME_NPZ,
        "summary.json",
        "run_info.json",
    )
    sizes: dict[str, int] = {}
    soma_path = find_soma_trace_artifact(result_dir)
    if soma_path is not None and soma_path.exists():
        sizes[soma_path.name] = int(soma_path.stat().st_size)
    for filename in filenames:
        path = result_dir / filename
        if path.exists():
            sizes[filename] = int(path.stat().st_size)
    return sizes


def _remote_sweep_metadata_files() -> tuple[str, ...]:
    """Return the small top-level sweep metadata files needed after remote completion."""
    return (
        "summary.json",
        "sim_progress.json",
        "sweep_manifest.json",
        "sweep_manifest.submit.json",
        "mpi_preflight.log",
        "bootstrap.log",
        "stdout.txt",
        "stderr.txt",
    )


_NONEMPTY_LOCAL_SYNC_ARTIFACTS = {
    "summary.json",
    "run_info.json",
    "input_times.pkl",
    SOMA_TRACE_FILENAME_PKL,
    SOMA_TRACE_FILENAME_NPZ,
    SOMA_SPIKES_FILENAME_NPZ,
    VOLTAGE_SUMMARY_FILENAME_NPZ,
    "gc_output_events.pkl",
    "lfp.pkl",
    "sim_progress.json",
}


def _local_sync_artifact_is_usable(path: str | Path) -> bool:
    """Return True when one synced local artifact exists and is not a known empty placeholder."""
    path = Path(path)
    if not path.exists():
        return False
    if not path.is_file():
        return True
    if path.name in _NONEMPTY_LOCAL_SYNC_ARTIFACTS:
        return path.stat().st_size > 0
    return True


def _replace_file_via_temp_copy(copy_fn: Any, local_path: Path) -> None:
    """Write one synced file through a temporary sibling, then atomically replace the target."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f".{local_path.name}.obgpu-partial-{os.getpid()}")
    try:
        temp_path.unlink(missing_ok=True)
        copy_fn(temp_path)
        os.replace(temp_path, local_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _remote_fast_sync_files(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return the top-level result artifacts needed for a fast successful remote run_and_load."""
    files = ["summary.json", "input_times.pkl"]
    if config is None or bool(config.get("enable_lfp", True)):
        files.append("lfp.pkl")
    if config is None or bool(config.get("record_gc_output_events", False)):
        files.append("gc_output_events.pkl")
    if config is None or bool(config.get("record_from_somas", [])):
        files.extend([SOMA_SPIKES_FILENAME_NPZ, VOLTAGE_SUMMARY_FILENAME_NPZ])
    return tuple(files)


def _sweep_signal_requires_voltage_summary(signal: Any) -> bool:
    """Return True when a configured sweep analysis signal needs saved voltage moments."""
    if not isinstance(signal, str):
        return False
    normalized = signal.strip()
    return re.fullmatch(r"mean_[A-Z]+_voltage", normalized) is not None


def _should_sync_sweep_voltage_summary(config: dict[str, Any] | None = None) -> bool:
    """Return True when remote sweep item sync should include voltage-summary arrays."""
    if bool((config or {}).get("sweep_sync_voltage_summary", False)):
        return True
    if config is None:
        return False
    return any(
        _sweep_signal_requires_voltage_summary(config.get(key))
        for key in ("spectrogram_signal", "wavelet_signal")
    )


def _should_sync_remote_sweep_finished_items(
    config: dict[str, Any] | None = None,
    *,
    pending_count: int,
    running_count: int,
) -> bool:
    """Return whether compact finished-item sync should run during sweep polling."""
    if bool((config or {}).get("sweep_sync_live", False)):
        return True
    # Once the final wave is in flight, overlap compact item transfer with the
    # remaining remote work so the next batch pays less end-of-sweep latency.
    return int(pending_count) <= 0 and int(running_count) > 0


def _remote_sweep_item_sync_files(config: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return compact per-item artifacts synced by remote sweeps by default."""
    files = [
        "summary.json",
        "input_times.pkl",
        *(
            ("lfp.pkl",)
            if config is None or bool(config.get("enable_lfp", True))
            else ()
        ),
        *(
            ("gc_output_events.pkl",)
            if config is None or bool(config.get("record_gc_output_events", False))
            else ()
        ),
        *(
            (SOMA_SPIKES_FILENAME_NPZ,)
            if config is None or bool(config.get("record_from_somas", []))
            else ()
        ),
        *(
            (VOLTAGE_SUMMARY_FILENAME_NPZ,)
            if _should_sync_sweep_voltage_summary(config)
            else ()
        ),
        "run_info.json",
        "command.txt",
        "stdout.txt",
        "stderr.txt",
    ]
    if bool((config or {}).get("sweep_sync_soma_vs", False)):
        files.extend([SOMA_TRACE_FILENAME_NPZ, SOMA_TRACE_FILENAME_PKL])
    return tuple(dict.fromkeys(files))


def _remote_sweep_item_diagnostic_files() -> tuple[str, ...]:
    """Return small per-item diagnostics for failed remote sweep items."""
    return ("summary.json", "run_info.json", "command.txt", "stdout.txt", "stderr.txt", "bootstrap.log")


def _merge_run_info_payload(result_dir: str | Path, extra_payload: dict[str, Any]) -> None:
    """Merge extra metadata into an existing run_info.json payload."""
    result_dir = Path(result_dir)
    run_info_path = result_dir / "run_info.json"
    payload = _read_json_if_present(run_info_path) or {}
    payload.update(_json_ready(extra_payload))
    run_info_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _remote_helper_bundle_entries() -> tuple[HelperBundleEntry, ...]:
    """Return the helper-bundle entries that should be cached on the remote host."""
    helper_dir = REPO_ROOT / "tools" / "remote"
    neuroinfra_dir = REPO_ROOT / "neuroinfra"
    return (
        HelperBundleEntry("slurm_common.py", helper_dir / "slurm_common.py"),
        HelperBundleEntry("submit_sol_run.py", helper_dir / "submit_sol_run.py"),
        HelperBundleEntry("submit_slurm_allocation.py", helper_dir / "submit_slurm_allocation.py"),
        HelperBundleEntry("poll_sol_run.py", helper_dir / "poll_sol_run.py"),
        HelperBundleEntry("cleanup_stale_allocations.py", helper_dir / "cleanup_stale_allocations.py"),
        HelperBundleEntry("neuroinfra/__init__.py", neuroinfra_dir / "__init__.py"),
        HelperBundleEntry("neuroinfra/inventory.py", neuroinfra_dir / "inventory.py"),
        HelperBundleEntry("neuroinfra/remote_script_common.py", neuroinfra_dir / "remote_script_common.py"),
        HelperBundleEntry("neuroinfra/remote_script_submit.py", neuroinfra_dir / "remote_script_submit.py"),
        HelperBundleEntry("neuroinfra/remote_script_polling.py", neuroinfra_dir / "remote_script_polling.py"),
        HelperBundleEntry("neuroinfra/remote_script_allocations.py", neuroinfra_dir / "remote_script_allocations.py"),
    )


def _remote_helper_sources() -> dict[str, Path]:
    """Return the helper scripts that should be cached on the remote host."""
    return bundle_entries_by_path(_remote_helper_bundle_entries())


def _remote_helper_signature() -> str:
    """Return a content signature for the current remote-helper set."""
    return helper_bundle_signature(_remote_helper_bundle_entries())[:20]


def _remote_helper_cache_runtime_key(config: dict[str, Any]) -> str:
    """Return the runtime cache key for one uploaded remote helper directory."""
    return _neuroinfra_helper_cache_runtime_key(
        connection_key=_paramiko_connection_key(config),
        results_root=_remote_results_root(config),
        signature=_remote_helper_signature(),
    )


def _remote_helper_cache_dir(config: dict[str, Any]) -> PurePosixPath:
    """Return the remote directory that stores cached notebook helper scripts."""
    return _neuroinfra_helper_cache_dir(
        results_root=_remote_results_root(config),
        signature=_remote_helper_signature(),
    )


_REMOTE_SLURM_TERMINAL_OK = _NEUROINFRA_REMOTE_SLURM_TERMINAL_OK
_REMOTE_SLURM_TERMINAL_FAIL = _NEUROINFRA_REMOTE_SLURM_TERMINAL_FAIL


def _resolve_local_git_head() -> str | None:
    """Return the current local git HEAD commit or ``None`` when unavailable."""
    return _neuroinfra_resolve_local_git_head(REPO_ROOT)


def _resolve_local_git_branch() -> str | None:
    """Return the current local branch name or ``None`` when detached."""
    return _neuroinfra_resolve_local_git_branch(REPO_ROOT)


def _resolve_local_git_upstream_ref() -> str | None:
    """Return the current branch upstream ref, or ``None`` when unavailable."""
    return _neuroinfra_resolve_local_git_upstream_ref(REPO_ROOT)


def _git_rev_parse(ref_name: str) -> str | None:
    """Resolve one local git ref to a commit SHA."""
    return _neuroinfra_git_rev_parse(REPO_ROOT, ref_name)


def _git_ref_points_to_commit(ref_name: str, commit_sha: str) -> bool:
    """Return whether one local git ref currently resolves to the requested commit."""
    return _neuroinfra_git_ref_points_to_commit(REPO_ROOT, ref_name, commit_sha)


def _git_ref_is_ancestor(ancestor_ref: str, descendant_ref: str) -> bool:
    """Return whether one git ref is an ancestor of another."""
    return _neuroinfra_git_ref_is_ancestor(REPO_ROOT, ancestor_ref, descendant_ref)


def _git_merged_ref_shas(commit_sha: str, *, max_count: int = 128) -> list[str]:
    """Return ancestor ref tips already merged into one commit."""
    return _neuroinfra_git_merged_ref_shas(REPO_ROOT, commit_sha, max_count=max_count)


def _local_git_sync_base_candidates(commit_sha: str, *, max_count: int = 500) -> list[str]:
    """Return local ancestor SHAs to test as possible remote bundle bases."""
    return _neuroinfra_local_git_sync_base_candidates(REPO_ROOT, commit_sha, max_count=max_count)


def _resolve_remote_git_ref(config: dict[str, Any]) -> str | None:
    """Return the requested Sol git ref, defaulting to the current local HEAD commit."""
    configured = config.get("remote_git_ref")
    if configured not in (None, ""):
        return str(configured)
    return _resolve_local_git_head()


def _require_remote_host(config: dict[str, Any]) -> str:
    """Return the configured remote SSH target."""
    return _neuroinfra_require_remote_host(config)


def _remote_transport(config: dict[str, Any]) -> str:
    """Return the active remote transport.

    The notebook remote backend is Paramiko-only. The old OpenSSH control-master
    and subprocess transfer branch repeatedly caused duplicate authentication
    and stale socket failures in notebook sweeps, so ``ssh_transport`` remains
    only as a compatibility guard for older configs.
    """
    configured = str(config.get("ssh_transport", "auto")).strip().lower()
    if configured in {"", "auto"}:
        configured = "paramiko"
    if configured != "paramiko":
        raise ValueError(f"Unsupported ssh_transport={configured!r}")
    if paramiko is None:
        raise RuntimeError(
            "Remote notebook runs require the optional 'paramiko' dependency. "
            "Install/update the maintained OBGPU environment before using a remote Slurm backend."
        )
    return "paramiko"


def _remote_endpoint(config: dict[str, Any]) -> tuple[str, int, str]:
    """Resolve hostname, port, and username from the remote config."""
    return _neuroinfra_resolve_remote_endpoint(config)


def _paramiko_connection_key(config: dict[str, Any]) -> str:
    """Build the cache key for one persistent Paramiko connection."""
    return _neuroinfra_remote_connection_key(config)


def _paramiko_transport_is_usable(transport: Any) -> bool:
    """Return whether one cached Paramiko transport still looks authenticated and alive."""
    return _neuroinfra_transport_is_usable(transport)


def _paramiko_connect_retry_count(config: dict[str, Any]) -> int:
    """Return how many times one fresh Paramiko connect may be retried."""
    return _neuroinfra_connect_retry_count(config)


def _paramiko_connect_retry_backoff_s(config: dict[str, Any]) -> float:
    """Return the base sleep between fresh Paramiko connect retries."""
    return _neuroinfra_connect_retry_backoff_s(config)


def _paramiko_transport_context(config: dict[str, Any]) -> _NeuroinfraParamikoTransportContext:
    """Build the reusable Paramiko transport context for one notebook config."""
    def _dynamic_ipython_getter() -> Any:
        try:
            from IPython import get_ipython as _ipython_getter
        except Exception:  # pragma: no cover - optional notebook integration
            return None
        return _ipython_getter()

    return _NeuroinfraParamikoTransportContext(
        config=config,
        paramiko_module=paramiko,
        live_connections=_LIVE_PARAMIKO_CONNECTIONS,
        authenticated_keys=_LIVE_PARAMIKO_AUTHENTICATED_KEYS,
        progress_write=_progress_write,
        connection_key_fn=_paramiko_connection_key,
        can_reconnect_fn=_paramiko_can_reconnect,
        midrun_reauth_error_fn=_paramiko_midrun_reauth_error,
        remote_endpoint_fn=_remote_endpoint,
        connect_retry_count_fn=_paramiko_connect_retry_count,
        connect_retry_backoff_s_fn=_paramiko_connect_retry_backoff_s,
        transport_is_usable_fn=_paramiko_transport_is_usable,
        get_cached_prompt_response_fn=_get_cached_paramiko_prompt_response,
        cache_prompt_response_fn=_cache_paramiko_prompt_response,
        ssh_command_timeout_s_fn=_remote_ssh_command_timeout_s,
        ssh_exec_timeout_s_fn=_remote_ssh_exec_timeout_s,
        socket_create_connection_fn=socket.create_connection,
        sleep_fn=time.sleep,
        getpass_fn=getpass,
        ipython_getter=_dynamic_ipython_getter,
    )


def _paramiko_connect_error_is_retryable(exc: BaseException) -> bool:
    """Return whether one fresh Paramiko connect failure is transient enough to retry."""
    return _neuroinfra_connect_error_is_retryable(exc, paramiko_module=paramiko)


def _paramiko_prompt_key(prompt_text: str) -> str:
    """Normalize one interactive-auth prompt into a stable cache key."""
    return _neuroinfra_prompt_key(prompt_text)


def _paramiko_cached_prompt_responses(config: dict[str, Any]) -> dict[str, str]:
    """Return the cached auth-prompt responses for one endpoint."""
    return _neuroinfra_cached_prompt_responses(_LIVE_PARAMIKO_PROMPT_CACHE, _paramiko_connection_key(config))


def _get_cached_paramiko_prompt_response(config: dict[str, Any], prompt_text: str) -> str | None:
    """Return one remembered auth response for this endpoint, if available."""
    return _neuroinfra_get_cached_prompt_response(
        _LIVE_PARAMIKO_PROMPT_CACHE,
        _paramiko_connection_key(config),
        prompt_text,
    )


def _cache_paramiko_prompt_response(config: dict[str, Any], prompt_text: str, response: str) -> None:
    """Remember one auth response for later noninteractive reconnects."""
    _neuroinfra_cache_prompt_response(
        _LIVE_PARAMIKO_PROMPT_CACHE,
        _paramiko_connection_key(config),
        prompt_text,
        response,
    )


def _paramiko_has_cached_auth(config: dict[str, Any]) -> bool:
    """Return whether one endpoint has cached auth responses for silent reconnect."""
    return _neuroinfra_has_cached_auth(_LIVE_PARAMIKO_PROMPT_CACHE, _paramiko_connection_key(config))


def _paramiko_can_reconnect(config: dict[str, Any]) -> bool:
    """Return whether one dead transport may be recovered automatically."""
    return _neuroinfra_can_reconnect(
        connection_key=_paramiko_connection_key(config),
        preserve_session=bool(config.get("remote_preserve_paramiko_session", True)),
        allow_reauth=bool(config.get("remote_allow_paramiko_reauth", False)),
        authenticated_keys=_LIVE_PARAMIKO_AUTHENTICATED_KEYS,
        prompt_cache=_LIVE_PARAMIKO_PROMPT_CACHE,
    )


def _paramiko_midrun_reauth_error(config: dict[str, Any]) -> str:
    """Explain why a fresh Paramiko login is being refused mid-run."""
    return _neuroinfra_midrun_reauth_error(_paramiko_connection_key(config))


def _remote_git_ref_cache_key(config: dict[str, Any], remote_repo_root: PurePosixPath) -> str:
    """Build the runtime cache key for remote git-object presence checks."""
    return f"{_paramiko_connection_key(config)}::{remote_repo_root.as_posix()}"


def _build_remote_git_repo_probe_command(remote_repo_root: PurePosixPath) -> str:
    """Build a remote shell command that verifies the configured repo exists."""
    return _neuroinfra_build_remote_git_repo_probe_command(remote_repo_root)


def _normalize_slurm_state(raw_state: str) -> str:
    """Normalize Slurm state tokens by removing suffixes such as '+'."""
    return _neuroinfra_normalize_slurm_state(raw_state)


def _remote_heartbeat_timeout_s(config: dict[str, Any]) -> int:
    """Return the notebook heartbeat timeout used by remote Slurm watchdogs."""
    return _neuroinfra_heartbeat_timeout_s(config)


def _remote_ssh_command_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the per-command Paramiko shell timeout, or None when disabled."""
    return _neuroinfra_ssh_command_timeout_s(config)


def _remote_ssh_exec_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the Paramiko exec request acknowledgement timeout."""
    return _neuroinfra_ssh_exec_timeout_s(config)


def _remote_ssh_upload_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the Paramiko shell upload timeout, or None when disabled."""
    return _neuroinfra_ssh_upload_timeout_s(config)


def _remote_poll_command_timeout_s(config: dict[str, Any]) -> float | None:
    """Return the tighter timeout for lightweight remote polling commands."""
    return _neuroinfra_poll_command_timeout_s(config)


def _slurm_allocation_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Return the cache signature for one reusable remote Slurm allocation."""
    return _neuroinfra_allocation_signature(
        connection_key=_paramiko_connection_key(config),
        results_root=_remote_results_root(config),
        partition=config.get("slurm_partition"),
        account=config.get("slurm_account"),
        time_limit=str(config.get("slurm_allocation_time") or config.get("slurm_time") or ""),
        gpus=config.get("slurm_gpus"),
        cpus_per_task=config.get("slurm_cpus_per_task"),
        mem=config.get("slurm_mem"),
        extra_args=[str(arg) for arg in config.get("slurm_extra_args", [])],
        remote_conda_activate_cmd=str(config.get("remote_conda_activate_cmd") or ""),
        remote_runtime_profiles=_json_ready(config.get("remote_runtime_profiles") or []),
        remote_fallback_conda_activate_cmd=str(config.get("remote_fallback_conda_activate_cmd") or ""),
        remote_fast_node_feature=str(config.get("remote_fast_node_feature") or ""),
        remote_mechanism_profile=str(config.get("remote_mechanism_profile") or "default"),
        remote_fallback_mechanism_profile=str(config.get("remote_fallback_mechanism_profile") or "portable"),
        name=str(config.get("slurm_allocation_name") or "obgpu_notebook_alloc"),
    )


def _slurm_allocation_cache_key(config: dict[str, Any]) -> str:
    """Return the runtime cache key for one reusable remote Slurm allocation."""
    return _neuroinfra_allocation_cache_key(_slurm_allocation_signature(config))


def _remote_allocation_runtime_context(config: dict[str, Any]) -> _NeuroinfraRemoteAllocationRuntimeContext:
    """Build the reusable remote allocation runtime context for one config."""
    return _NeuroinfraRemoteAllocationRuntimeContext(
        config=config,
        live_slurm_allocations=_LIVE_SLURM_ALLOCATIONS,
        live_remote_stale_cleanups=_LIVE_REMOTE_STALE_CLEANUPS,
        progress_write=_progress_write,
        connection_key_fn=_paramiko_connection_key,
        remote_results_root_fn=_remote_results_root,
        poll_command_timeout_s_fn=_remote_poll_command_timeout_s,
        heartbeat_timeout_s_fn=_remote_heartbeat_timeout_s,
        allocation_cache_key_fn=_slurm_allocation_cache_key,
        allocation_runtime_config_fn=_slurm_allocation_runtime_config,
        build_remote_touch_command_fn=_build_remote_touch_command,
        build_remote_cleanup_allocations_command_fn=lambda cfg, remote_helper_dir: _build_remote_cleanup_allocations_command(
            cfg,
            remote_helper_dir=remote_helper_dir,
        ),
        build_remote_allocation_discovery_command_fn=lambda cfg, remote_helper_dir: _build_remote_allocation_discovery_command(
            cfg,
            remote_helper_dir=remote_helper_dir,
        ),
        build_remote_allocation_submit_command_fn=lambda cfg, remote_helper_dir: _build_remote_allocation_submit_command(
            cfg,
            remote_helper_dir=remote_helper_dir,
        ),
        build_remote_cancel_command_fn=lambda job_id: _build_remote_cancel_command(job_id=job_id),
        run_ssh_shell_fn=lambda cfg, command, check=False, timeout_s=None: _run_ssh_shell(
            cfg,
            command,
            check=check,
            timeout_s=timeout_s,
        ),
        query_remote_slurm_job_state_fn=_query_remote_slurm_job_state,
        time_fn=time.time,
        sleep_fn=time.sleep,
    )


def _paramiko_stream_sync_hooks() -> _NeuroinfraParamikoStreamSyncHooks:
    """Build reusable hooks for Paramiko-driven archive and direct-file streaming."""
    return _NeuroinfraParamikoStreamSyncHooks(
        transport_for_config_fn=lambda cfg: _connect_paramiko(cfg)["transport"],
        run_paramiko_shell_fn=_run_paramiko_shell,
        build_remote_stream_archive_command_fn=lambda remote_result_dir, compressor: _build_remote_stream_archive_command(
            remote_result_dir,
            compressor=compressor,
        ),
        build_remote_selected_archive_probe_command_fn=lambda remote_result_dir, include_files: _build_remote_selected_archive_probe_command(
            remote_result_dir,
            include_files=include_files,
        ),
        local_archive_decompress_command_fn=_local_archive_decompress_command,
        channel_stream_finished_fn=_paramiko_channel_stream_finished,
        progress_factory_fn=lambda total, desc: _ProgressBar(
            total=total,
            desc=desc,
            unit="B",
            unit_scale=True,
            display_step=10 * 1024 * 1024,
        ),
        sleep_fn=time.sleep,
    )


def _remote_result_sync_hooks() -> _NeuroinfraRemoteResultSyncHooks:
    """Build reusable hooks for higher-level Paramiko result-sync policy."""
    return _NeuroinfraRemoteResultSyncHooks(
        remote_transport_fn=_remote_transport,
        run_paramiko_shell_fn=_run_paramiko_shell,
        build_remote_archive_probe_command_fn=_build_remote_archive_probe_command,
        probe_selected_sync_files_fn=lambda cfg, remote_result_dir, include_files: _probe_remote_selected_sync_files(
            cfg,
            remote_result_dir=remote_result_dir,
            include_files=include_files,
        ),
        build_remote_selected_stream_archive_command_fn=lambda remote_result_dir, include_files, compressor: _build_remote_selected_stream_archive_command(
            remote_result_dir,
            include_files=include_files,
            compressor=compressor,
        ),
        stream_archive_to_local_dir_fn=_stream_paramiko_archive_to_local_dir,
        get_paramiko_sftp_fn=_get_paramiko_sftp,
        close_paramiko_sftp_fn=_close_paramiko_sftp,
        sftp_copy_files_fn=lambda sftp, remote_dir, local_dir, file_names: _sftp_copy_files(
            sftp,
            remote_dir,
            local_dir,
            file_names,
        ),
        sftp_copy_tree_fn=lambda sftp, remote_dir, local_dir: _sftp_copy_tree(
            sftp,
            remote_dir,
            local_dir,
        ),
        cached_transport_fn=lambda cfg: (
            (_LIVE_PARAMIKO_CONNECTIONS.get(_paramiko_connection_key(cfg)) or {}).get("transport")
            if isinstance(_LIVE_PARAMIKO_CONNECTIONS.get(_paramiko_connection_key(cfg)), dict)
            else None
        ),
        transport_is_usable_fn=_paramiko_transport_is_usable,
        preserve_reauth_blocked_fn=lambda cfg: (
            bool(cfg.get("remote_preserve_paramiko_session", True))
            and _paramiko_connection_key(cfg) in _LIVE_PARAMIKO_AUTHENTICATED_KEYS
        ),
        drop_paramiko_connection_fn=_drop_paramiko_connection,
        midrun_reauth_error_fn=_paramiko_midrun_reauth_error,
        progress_write=_progress_write,
        missing_local_sync_artifacts_fn=lambda result_dir, expected_files: _missing_local_sync_artifacts(
            result_dir,
            expected_files=expected_files,
        ),
        local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
        sleep_fn=time.sleep,
    )


def _deferred_remote_artifact_sync_hooks() -> _NeuroinfraDeferredArtifactSyncHooks:
    """Build reusable hooks for deferred remote artifact sync policy."""
    return _NeuroinfraDeferredArtifactSyncHooks(
        local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
        sync_remote_result_dir_fn=_sync_remote_result_dir,
        progress_write=_progress_write,
        format_bytes_fn=_format_bytes,
        direct_stream_supported_fn=lambda filename: filename in soma_trace_artifact_candidates(),
        run_paramiko_shell_fn=_run_paramiko_shell,
        stream_file_to_local_path_fn=_stream_paramiko_file_to_local_path,
        perf_counter_fn=time.perf_counter,
    )


def _artifact_loading_hooks() -> _NeuroinfraArtifactLoadingHooks:
    """Build reusable hooks for local result artifact loading."""
    return _NeuroinfraArtifactLoadingHooks(
        load_pickle_fn=load_pickle,
        apply_loaded_fn=_apply_loaded_result_artifact,
        progress_factory_fn=lambda total_bytes, desc: _ProgressBar(
            total=total_bytes,
            desc=desc,
            unit="B",
            unit_scale=True,
        ),
        progress_write=_progress_write,
        format_bytes_fn=_format_bytes,
        render_progress_bar_fn=_render_progress_bar,
        perf_counter_fn=time.perf_counter,
    )


def _result_view_hooks() -> _NeuroinfraResultViewHooks:
    """Build reusable hooks for result-view planning and lazy artifact wiring."""
    return _NeuroinfraResultViewHooks(
        read_json_if_present_fn=_read_json_if_present,
        standard_result_artifact_sizes_fn=_standard_result_artifact_sizes,
        local_sync_artifact_is_usable_fn=_local_sync_artifact_is_usable,
        sync_deferred_artifact_fn=_sync_deferred_remote_artifact,
        load_pickle_fn=load_pickle,
        set_lazy_artifact_path_fn=_OBGPU_RESULT_VIEW_SCHEMA.set_lazy_artifact_path,
        local_lazy_notice_fn=lambda key, path: (
            f"[OBGPU load] Deferred {key} ({_format_bytes(path.stat().st_size)}) until result['{key}'] is accessed."
        ),
        remote_lazy_notice_fn=lambda key, _path: (
            f"[OBGPU load] Deferred {key} stays remote until result['{key}'] is accessed."
        ),
        progress_write=_progress_write,
    )


def _remote_json_poll_hooks(
    config: dict[str, Any],
    notebook_timings: dict[str, float],
) -> _NeuroinfraRemoteJSONPollHooks:
    """Build reusable hooks for remote JSON status polling."""
    return _NeuroinfraRemoteJSONPollHooks(
        run_command_fn=lambda command, timeout_s=None: _run_ssh_shell(
            config,
            command,
            timeout_s=timeout_s,
        ),
        record_timing_fn=lambda key, started: _record_timing(notebook_timings, key, started),
        sleep_fn=time.sleep,
        perf_counter_fn=time.perf_counter,
    )


def _remote_run_artifact_hooks(
    notebook_timings: dict[str, float],
) -> _NeuroinfraRemoteRunArtifactHooks:
    """Build reusable hooks for remote single-run final sync and artifact collection."""
    return _NeuroinfraRemoteRunArtifactHooks(
        sync_remote_result_dir_resilient_fn=_sync_remote_result_dir_resilient,
        sync_remote_result_dir_fn=_sync_remote_result_dir,
        run_paramiko_shell_fn=_run_ssh_shell,
        build_remote_result_listing_command_fn=_build_remote_result_listing_command,
        local_result_dir_has_loadable_payload_fn=_local_result_dir_has_loadable_payload,
        local_result_dir_has_diagnostics_fn=_local_result_dir_has_diagnostics,
        standard_result_artifact_sizes_fn=_standard_result_artifact_sizes,
        synthesize_partial_sync_summary_fn=_synthesize_partial_sync_summary,
        compact_remote_poll_events_fn=_compact_remote_poll_events,
        read_json_if_present_fn=_read_json_if_present,
        progress_write=_progress_write,
        record_timing_fn=lambda key, started: _record_timing(notebook_timings, key, started),
        sleep_fn=time.sleep,
        perf_counter_fn=time.perf_counter,
    )


def _remote_run_monitor_hooks(
    *,
    effective_config: dict[str, Any],
    remote_job_heartbeat_path: str | None,
    allocation_heartbeat_path: str | None,
    remote_repo_root: PurePosixPath,
    remote_result_dir: PurePosixPath,
    remote_helper_dir: PurePosixPath | None,
    notebook_timings: dict[str, float],
    submission: dict[str, Any],
    local_result_dir: Path,
) -> _NeuroinfraRemoteRunMonitorHooks:
    """Build reusable hooks for live remote single-run monitoring."""

    def refresh_remote_leases(*, warn: bool = False) -> None:
        _refresh_remote_heartbeat(effective_config, remote_job_heartbeat_path, warn=warn)
        _refresh_remote_heartbeat(effective_config, allocation_heartbeat_path, warn=warn)

    def poll_status_once(
        *,
        refresh_heartbeat: bool = True,
        include_logs: bool = True,
        include_sacct: bool = True,
    ) -> dict[str, Any]:
        if refresh_heartbeat:
            refresh_remote_leases()
        poll_shell = _build_remote_poll_command(
            effective_config,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_result_dir,
            job_id=str(submission["job_id"]),
            wrapper_dir=str(submission.get("wrapper_dir") or ""),
            worktree_path=str(submission.get("worktree_path") or ""),
            remote_helper_dir=remote_helper_dir,
            include_sacct=include_sacct,
            include_tails=include_logs,
        )
        return _neuroinfra_poll_remote_json_status(
            poll_shell,
            poll_json_retries=max(int(effective_config.get("remote_poll_json_retries", 3) or 1), 1),
            error_prefix="Remote Sol status poll",
            hooks=_remote_json_poll_hooks(effective_config, notebook_timings),
        )

    def cancel_job() -> subprocess.CompletedProcess[str]:
        return _run_ssh_shell(
            effective_config,
            _build_remote_cancel_command(job_id=str(submission["job_id"])),
        )

    def sync_partial_artifacts() -> subprocess.CompletedProcess[str]:
        sync_started = time.perf_counter()
        sync_completed = _sync_remote_result_dir(
            effective_config,
            remote_result_dir=remote_result_dir,
            local_result_dir=local_result_dir,
        )
        _record_timing(notebook_timings, "partial_sync_s", sync_started)
        (local_result_dir / "sync_stdout.txt").write_text(sync_completed.stdout or "")
        (local_result_dir / "sync_stderr.txt").write_text(sync_completed.stderr or "")
        return sync_completed

    return _NeuroinfraRemoteRunMonitorHooks(
        refresh_remote_leases_fn=refresh_remote_leases,
        poll_status_fn=poll_status_once,
        cancel_job_fn=cancel_job,
        sync_partial_artifacts_fn=sync_partial_artifacts,
        remote_status_has_artifacts_fn=_remote_status_has_artifacts,
        progress_bar_factory_fn=lambda total_ms, desc: _ProgressBar(
            total=total_ms,
            desc=desc,
            unit="ms",
            unit_scale=False,
        ),
        filter_live_log_line_fn=_filter_live_remote_log_line,
        progress_write=_progress_write,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
        time_fn=time.time,
    )


def _remote_sweep_monitor_hooks(
    *,
    effective_config: dict[str, Any],
    remote_job_heartbeat_path: str | None,
    allocation_heartbeat_path: str | None,
    remote_repo_root: PurePosixPath,
    remote_sweep_root: PurePosixPath,
    remote_helper_dir: PurePosixPath | None,
    notebook_timings: dict[str, float],
    submission: dict[str, Any],
    synced_labels: set[str],
    sync_finished_items_fn: Callable[[dict[str, Any]], None],
) -> _NeuroinfraRemoteSweepMonitorHooks:
    """Build reusable hooks for live remote sweep monitoring."""

    def refresh_remote_leases(*, warn: bool = False) -> None:
        _refresh_remote_heartbeat(effective_config, remote_job_heartbeat_path, warn=warn)
        _refresh_remote_heartbeat(effective_config, allocation_heartbeat_path, warn=warn)

    def poll_status_once(*, refresh_heartbeat: bool = True, include_sacct: bool = True) -> dict[str, Any]:
        if refresh_heartbeat:
            refresh_remote_leases()
        poll_shell = _build_remote_poll_command(
            effective_config,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_sweep_root,
            job_id=str(submission["job_id"]),
            wrapper_dir=str(submission.get("wrapper_dir") or ""),
            worktree_path=str(submission.get("worktree_path") or ""),
            remote_helper_dir=remote_helper_dir,
            include_sacct=include_sacct,
            include_tails=False,
        )
        return _neuroinfra_poll_remote_json_status(
            poll_shell,
            poll_json_retries=max(int(effective_config.get("remote_poll_json_retries", 3) or 1), 1),
            error_prefix="Remote sweep status poll",
            hooks=_remote_json_poll_hooks(effective_config, notebook_timings),
            timeout_s=_remote_poll_command_timeout_s(effective_config),
        )

    def cancel_job() -> subprocess.CompletedProcess[str]:
        return _run_ssh_shell(
            effective_config,
            _build_remote_cancel_command(job_id=str(submission["job_id"])),
        )

    return _NeuroinfraRemoteSweepMonitorHooks(
        refresh_remote_leases_fn=refresh_remote_leases,
        poll_status_fn=poll_status_once,
        sync_finished_items_fn=sync_finished_items_fn,
        cancel_job_fn=cancel_job,
        synced_count_fn=lambda: len(synced_labels),
        progress_write=_progress_write,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    )


def _remote_sweep_artifact_hooks(
    *,
    refresh_remote_leases_fn: Callable[..., None],
    notebook_timings: dict[str, float],
) -> _NeuroinfraRemoteSweepArtifactHooks:
    """Build reusable hooks for remote sweep final sync and artifact collection."""
    return _NeuroinfraRemoteSweepArtifactHooks(
        sync_remote_result_dir_fn=_sync_remote_result_dir,
        sync_remote_sweep_compact_items_fn=_sync_remote_sweep_compact_items,
        read_json_if_present_fn=_read_json_if_present,
        recover_local_sweep_summary_fn=_recover_local_sweep_summary,
        remote_sweep_metadata_files_fn=_remote_sweep_metadata_files,
        remote_sweep_item_sync_files_fn=_remote_sweep_item_sync_files,
        remote_sweep_item_diagnostic_files_fn=_remote_sweep_item_diagnostic_files,
        local_sweep_item_sync_complete_fn=_local_sweep_item_sync_complete,
        local_result_dir_has_diagnostics_fn=_local_result_dir_has_diagnostics,
        progress_write=_progress_write,
        refresh_remote_leases_fn=refresh_remote_leases_fn,
        record_timing_fn=lambda key, started: _record_timing(notebook_timings, key, started),
        perf_counter_fn=time.perf_counter,
    )


def _apply_loaded_result_artifact(result: MutableMapping[str, Any], key: str, loaded: Any) -> None:
    """Apply one loaded artifact payload to the standard notebook result dict."""
    _OBGPU_RESULT_VIEW_SCHEMA.apply_loaded_artifact(result, key, loaded)


def _paramiko_prompt_response(prompt_text: str, *, config: dict[str, Any] | None = None) -> str:
    """Prompt the notebook user for one interactive SSH auth field."""
    cfg = {} if config is None else config
    return _paramiko_transport_context(cfg).prompt_response(prompt_text, config=config)


def _drop_paramiko_connection(config: dict[str, Any]) -> None:
    """Close and forget one cached Paramiko connection."""
    _paramiko_transport_context(config).drop_connection()


def _get_paramiko_sftp(config: dict[str, Any]) -> Any:
    """Return the cached Paramiko SFTP client, opening it only when needed."""
    connection = _connect_paramiko(config)
    sftp = connection.get("sftp")
    if sftp is not None:
        return sftp
    _progress_write("[Sol remote] Opening SFTP channel...")
    try:
        sftp = paramiko.SFTPClient.from_transport(connection["transport"])
    except Exception:
        connection["sftp"] = None
        if not _paramiko_transport_is_usable(connection.get("transport")):
            _drop_paramiko_connection(config)
        raise
    connection["sftp"] = sftp
    return sftp


def _close_paramiko_sftp(config: dict[str, Any]) -> None:
    """Close the cached Paramiko SFTP channel while keeping the SSH transport alive."""
    cached = _LIVE_PARAMIKO_CONNECTIONS.get(_paramiko_connection_key(config))
    if cached is None:
        return
    sftp = cached.get("sftp")
    if sftp is None:
        return
    cached["sftp"] = None
    try:
        sftp.close()
    except Exception:
        pass


def _connect_paramiko(config: dict[str, Any]) -> Any:
    """Open or reuse one persistent Paramiko transport for the Sol backend."""
    return _paramiko_transport_context(config).connect()


def _run_paramiko_shell(
    config: dict[str, Any],
    remote_shell_command: str,
) -> subprocess.CompletedProcess[str]:
    """Run one shell command over a persistent Paramiko transport."""
    return _paramiko_transport_context(config).run_shell(remote_shell_command)


def _sftp_copy_tree(sftp: Any, remote_dir: str, local_dir: Path) -> None:
    """Recursively copy one remote directory tree through SFTP with progress output."""
    hooks = _NeuroinfraSFTPSyncHooks(
        progress_factory=lambda total: _ProgressBar(total=total, desc="[OBGPU load] Sync from Sol", unit="B", unit_scale=True),
        progress_write=_progress_write,
        format_bytes=_format_bytes,
        render_progress_bar=_render_progress_bar,
        replace_file_via_temp_copy=_replace_file_via_temp_copy,
    )
    _neuroinfra_sftp_copy_tree(sftp, remote_dir, local_dir, hooks=hooks)


def _sftp_copy_files(sftp: Any, remote_dir: str, local_dir: Path, file_names: list[str] | tuple[str, ...]) -> None:
    """Copy a selected set of remote files through SFTP with progress output."""
    hooks = _NeuroinfraSFTPSyncHooks(
        progress_factory=lambda total: _ProgressBar(total=total, desc="[OBGPU load] Sync from Sol", unit="B", unit_scale=True),
        progress_write=_progress_write,
        format_bytes=_format_bytes,
        render_progress_bar=_render_progress_bar,
        replace_file_via_temp_copy=_replace_file_via_temp_copy,
    )
    _neuroinfra_sftp_copy_files(sftp, remote_dir, local_dir, file_names, hooks=hooks)


def _run_ssh_shell(
    config: dict[str, Any],
    remote_shell_command: str,
    *,
    check: bool = False,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one shell command on the remote Slurm host over the cached Paramiko session."""
    _remote_transport(config)
    shell_config = config
    if timeout_s is not None:
        shell_config = {**config, "remote_ssh_command_timeout_s": timeout_s}
    completed = _run_paramiko_shell(shell_config, remote_shell_command)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _build_remote_archive_command(remote_result_dir: PurePosixPath) -> str:
    """Build one remote shell command that packs the result dir into a compressed tar archive."""
    return _neuroinfra_build_remote_archive_command(remote_result_dir)


def _build_remote_archive_probe_command(remote_result_dir: PurePosixPath) -> str:
    """Build one remote shell command that selects a compressor and reports stream metadata."""
    return _neuroinfra_build_remote_archive_probe_command(remote_result_dir)


def _build_remote_selected_archive_probe_command(
    remote_result_dir: PurePosixPath,
    *,
    include_files: tuple[str, ...],
) -> str:
    """Build one remote shell command that reports stream metadata for selected files."""
    return _neuroinfra_build_remote_selected_archive_probe_command(
        remote_result_dir,
        include_files=include_files,
    )


def _build_remote_stream_archive_command(
    remote_result_dir: PurePosixPath,
    *,
    compressor: str,
) -> str:
    """Build one remote shell command that streams a compressed tar archive to stdout."""
    return _neuroinfra_build_remote_stream_archive_command(
        remote_result_dir,
        compressor=compressor,
    )


def _build_remote_selected_stream_archive_command(
    remote_result_dir: PurePosixPath,
    *,
    include_files: tuple[str, ...],
    compressor: str,
) -> str:
    """Build one remote shell command that streams selected files as a compressed tar archive."""
    return _neuroinfra_build_remote_selected_stream_archive_command(
        remote_result_dir,
        include_files=include_files,
        compressor=compressor,
    )


def _build_remote_sweep_compact_stream_archive_command(
    *,
    entries: list[dict[str, Any]],
    compressor: str,
) -> str:
    """Build one remote command that streams compact artifacts for many sweep items."""
    return _neuroinfra_build_remote_sweep_compact_stream_archive_command(
        entries=entries,
        compressor=compressor,
        json_ready=_json_ready,
    )


def _sync_remote_sweep_compact_items(
    config: dict[str, Any],
    *,
    local_sweep_dir: Path,
    entries: list[dict[str, Any]],
) -> subprocess.CompletedProcess[str]:
    """Sync compact artifacts for many remote sweep items in one Paramiko stream."""
    if not entries:
        return subprocess.CompletedProcess(
            args=["remote-sweep-compact-bulk", str(local_sweep_dir)],
            returncode=0,
            stdout="",
            stderr="",
        )
    if _remote_transport(config) != "paramiko":
        return subprocess.CompletedProcess(
            args=["remote-sweep-compact-bulk", str(local_sweep_dir)],
            returncode=1,
            stdout="",
            stderr="Bulk compact sweep sync currently requires ssh_transport='paramiko'.",
        )

    compressor = "zstd" if shutil.which("zstd") else "gzip"
    stream_command = _build_remote_sweep_compact_stream_archive_command(
        entries=entries,
        compressor=compressor,
    )
    try:
        completed = _stream_paramiko_archive_to_local_dir(
            config,
            remote_result_dir=PurePosixPath(str(entries[0].get("result_dir") or ".")),
            local_result_dir=local_sweep_dir,
            compressor=compressor,
            raw_bytes=0,
            stream_command=stream_command,
        )
    except Exception as exc:
        return subprocess.CompletedProcess(
            args=["remote-sweep-compact-bulk", str(local_sweep_dir)],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )
    return subprocess.CompletedProcess(
        args=["remote-sweep-compact-bulk", str(local_sweep_dir)],
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _probe_remote_selected_sync_files(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    include_files: tuple[str, ...],
) -> tuple[str, int, tuple[str, ...]] | subprocess.CompletedProcess[str]:
    """Return one selected-file sync plan using only remote files that actually exist."""
    return _neuroinfra_probe_selected_sync_files(
        config,
        remote_result_dir=remote_result_dir,
        include_files=tuple(include_files),
        hooks=_paramiko_stream_sync_hooks(),
    )


def _remove_remote_file(config: dict[str, Any], remote_path: str) -> None:
    """Best-effort remote file removal used for temporary sync archives."""
    remote_shell = "rm -f {}".format(shlex.quote(remote_path))
    _remote_transport(config)
    try:
        _run_paramiko_shell(config, remote_shell)
    except Exception:
        pass


def _upload_remote_bytes_file(
    config: dict[str, Any],
    *,
    remote_path: PurePosixPath,
    data: bytes,
    close_sftp: bool = True,
) -> None:
    """Upload one file to the remote host over the active notebook SSH transport."""
    _remote_transport(config)
    mkdir_completed = _run_paramiko_shell(
        config,
        "mkdir -p {}".format(shlex.quote(remote_path.parent.as_posix())),
    )
    if mkdir_completed.returncode != 0:
        raise RuntimeError(
            "Could not create the remote directory for an uploaded sweep payload.\n"
            f"Remote path: {remote_path.as_posix()}\n"
            f"Stdout:\n{mkdir_completed.stdout}\n\n"
            f"Stderr:\n{mkdir_completed.stderr}"
        )
    sftp = _get_paramiko_sftp(config)
    try:
        with sftp.open(remote_path.as_posix(), "wb") as handle:
            handle.write(data)
    finally:
        if close_sftp:
            _close_paramiko_sftp(config)


def _upload_remote_text_file(
    config: dict[str, Any],
    *,
    remote_path: PurePosixPath,
    text: str,
    close_sftp: bool = True,
) -> None:
    """Upload one UTF-8 text file to the remote host over the active notebook SSH transport."""
    _upload_remote_bytes_file(
        config,
        remote_path=remote_path,
        data=text.encode("utf-8"),
        close_sftp=close_sftp,
    )


def _ensure_remote_helper_cache(config: dict[str, Any]) -> PurePosixPath | None:
    """Upload notebook helper scripts once per session or reuse a remote cache hit."""
    _remote_transport(config)

    cache_key = _remote_helper_cache_runtime_key(config)
    cached = _LIVE_REMOTE_HELPER_CACHES.get(cache_key)
    if isinstance(cached, dict) and cached.get("remote_dir"):
        return PurePosixPath(str(cached["remote_dir"]))

    remote_dir = _remote_helper_cache_dir(config)
    manifest_path = _neuroinfra_helper_cache_manifest_path(remote_dir)
    expected_signature = _remote_helper_signature()
    probe_started = time.perf_counter()
    probe_completed = _run_paramiko_shell(
        config,
        _neuroinfra_helper_cache_probe_command(manifest_path),
    )
    if probe_completed.returncode == 0 and _neuroinfra_helper_cache_probe_matches(
        probe_completed.stdout or "",
        expected_signature=expected_signature,
    ):
        _LIVE_REMOTE_HELPER_CACHES[cache_key] = {
            "remote_dir": remote_dir.as_posix(),
            "signature": expected_signature,
            "cache_hit": True,
            "probe_s": round(time.perf_counter() - probe_started, 3),
        }
        return remote_dir

    _progress_write("[Sol remote] Uploading remote helper cache...")
    helper_entries = _remote_helper_bundle_entries()
    helper_sources, manifest_payload, manifest_path = _neuroinfra_helper_cache_upload_payload(
        remote_dir=remote_dir,
        entries=helper_entries,
        signature=expected_signature,
    )
    upload_started = time.perf_counter()
    sftp = _get_paramiko_sftp(config)
    try:
        mkdir_targets = _neuroinfra_helper_cache_mkdir_targets(
            remote_dir=remote_dir,
            entries=helper_entries,
        )
        mkdir_completed = _run_paramiko_shell(
            config,
            "mkdir -p {}".format(" ".join(shlex.quote(target) for target in mkdir_targets)),
        )
        if mkdir_completed.returncode != 0:
            raise RuntimeError(
                "Could not create the remote helper-cache directory.\n"
                f"Remote dir: {remote_dir.as_posix()}\n"
                f"Stdout:\n{mkdir_completed.stdout}\n\n"
                f"Stderr:\n{mkdir_completed.stderr}"
            )
        for name, path in helper_sources.items():
            with sftp.open((remote_dir / PurePosixPath(name)).as_posix(), "wb") as handle:
                handle.write(path.read_bytes())
        with sftp.open(manifest_path.as_posix(), "wb") as handle:
            handle.write(json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8"))
    finally:
        _close_paramiko_sftp(config)

    _LIVE_REMOTE_HELPER_CACHES[cache_key] = {
        "remote_dir": remote_dir.as_posix(),
        "signature": expected_signature,
        "cache_hit": False,
        "upload_s": round(time.perf_counter() - upload_started, 3),
    }
    return remote_dir


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
    return _neuroinfra_local_archive_decompress_command(compressor)


def _paramiko_channel_stream_finished(channel: Any) -> bool:
    """Return whether one Paramiko exec channel has reached a fully drained EOF."""
    return _neuroinfra_paramiko_channel_stream_finished(channel)


def _stream_paramiko_archive_to_local(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_archive_path: Path,
    compressor: str,
    raw_bytes: int,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote compressed tar archive over Paramiko into a local file."""
    return _neuroinfra_stream_archive_to_local(
        config,
        remote_result_dir,
        local_archive_path=local_archive_path,
        compressor=compressor,
        raw_bytes=raw_bytes,
        hooks=_paramiko_stream_sync_hooks(),
    )


def _stream_paramiko_archive_to_local_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    compressor: str,
    raw_bytes: int,
    stream_command: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote compressed tar archive over Paramiko directly into local extraction."""
    return _neuroinfra_stream_archive_to_local_dir(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
        compressor=compressor,
        raw_bytes=raw_bytes,
        hooks=_paramiko_stream_sync_hooks(),
        stream_command=stream_command,
    )


def _stream_paramiko_file_to_local_path(
    config: dict[str, Any],
    *,
    remote_file_path: PurePosixPath,
    local_path: Path,
    expected_bytes: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Stream one remote file over the existing Paramiko session without using SFTP."""
    return _neuroinfra_stream_file_to_local_path(
        config,
        remote_file_path=remote_file_path,
        local_path=local_path,
        expected_bytes=expected_bytes,
        hooks=_paramiko_stream_sync_hooks(),
    )


def _sync_remote_result_dir(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    expected_files: tuple[str, ...] | None = None,
    include_files: tuple[str, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Sync one remote result directory back into the local notebook results tree."""
    return _neuroinfra_sync_remote_result_dir(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
        expected_files=expected_files,
        include_files=include_files,
        hooks=_remote_result_sync_hooks(),
    )


def _combine_sync_attempt_stderr(attempts: list[tuple[str, subprocess.CompletedProcess[str]]]) -> str:
    """Render sync-attempt stderr with stage labels for actionable diagnostics."""
    return _neuroinfra_combine_sync_attempt_stderr(attempts)


def _sync_remote_result_dir_resilient(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    expected_files: tuple[str, ...] | None = None,
    include_files: tuple[str, ...] | None = None,
    wrapper_dir: str | PurePosixPath | None = None,
    retry_delay_s: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    """Sync remote results while treating selected-file sync as an optimization."""
    return _neuroinfra_sync_remote_result_dir_resilient(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
        expected_files=expected_files,
        include_files=include_files,
        wrapper_dir=wrapper_dir,
        retry_delay_s=retry_delay_s,
        hooks=_remote_result_sync_hooks(),
    )


def _local_result_dir_has_loadable_payload(result_dir: str | Path) -> bool:
    """Return True when the local result directory already has standard notebook payloads."""
    result_dir = Path(result_dir)
    for filename in (
        "input_times.pkl",
        "gc_output_events.pkl",
        "lfp.pkl",
        SOMA_SPIKES_FILENAME_NPZ,
        VOLTAGE_SUMMARY_FILENAME_NPZ,
    ):
        if _local_sync_artifact_is_usable(result_dir / filename):
            return True
    if find_soma_trace_artifact(result_dir) is not None:
        return True
    return False


def _local_result_dir_has_compact_payload(result_dir: str | Path) -> bool:
    """Return True when a result dir has compact artifacts expected from sweep sync."""
    result_dir = Path(result_dir)
    for filename in (
        "input_times.pkl",
        "gc_output_events.pkl",
        "lfp.pkl",
        SOMA_SPIKES_FILENAME_NPZ,
        VOLTAGE_SUMMARY_FILENAME_NPZ,
    ):
        if _local_sync_artifact_is_usable(result_dir / filename):
            return True
    return False


def _local_result_dir_has_diagnostics(result_dir: str | Path) -> bool:
    """Return True when a failed remote run has useful local logs to report."""
    result_dir = Path(result_dir)
    for filename in ("stdout.txt", "stderr.txt", "bootstrap.log", "command.txt", "submit_stdout.txt", "submit_stderr.txt"):
        if _local_sync_artifact_is_usable(result_dir / filename):
            return True
    return any(path.is_file() and path.stat().st_size > 0 for path in result_dir.glob("slurm-*.out"))


def _local_sweep_item_sync_complete(result_dir: str | Path) -> bool:
    """Return True when one local sweep item already has a loadable payload and summary."""
    result_dir = Path(result_dir)
    return _local_sync_artifact_is_usable(result_dir / "summary.json") and _local_result_dir_has_compact_payload(result_dir)


def _local_sweep_item_dirs(local_runs_dir: str | Path, label: str) -> list[Path]:
    """Return plausible local directories for one sweep item, newest payload dirs first."""
    local_runs_dir = Path(local_runs_dir)
    candidates: list[Path] = []
    exact = local_runs_dir / str(label)
    if exact.exists():
        candidates.append(exact)
    candidates.extend(path for path in local_runs_dir.glob(f"{label}_*") if path.is_dir())
    unique = {path.resolve(): path for path in candidates}
    return sorted(
        unique.values(),
        key=lambda path: (
            _local_sweep_item_sync_complete(path),
            _local_result_dir_has_loadable_payload(path),
            path.stat().st_mtime if path.exists() else 0.0,
            path.name,
        ),
        reverse=True,
    )


def _resolve_local_sweep_item_dir(
    local_runs_dir: str | Path,
    label: str,
    *,
    require_payload: bool = True,
) -> Path | None:
    """Return the best available local directory for one sweep item."""
    for candidate in _local_sweep_item_dirs(local_runs_dir, label):
        if not require_payload or _local_result_dir_has_loadable_payload(candidate):
            return candidate
    return None


def _local_result_dir_has_remote_sync_artifacts(result_dir: str | Path) -> bool:
    """Return True when one local result directory has any remote-generated sync artifacts."""
    result_dir = Path(result_dir)
    if _local_result_dir_has_loadable_payload(result_dir):
        return True
    for filename in (
        "summary.json",
        "stdout.txt",
        "stderr.txt",
        "bootstrap.log",
        "command.txt",
        "git_commit.txt",
        "git_ref.txt",
        "remote_submit.json",
        "sweep_manifest.json",
        "sweep_info.json",
        "sweep_status.json",
    ):
        if _local_sync_artifact_is_usable(result_dir / filename):
            return True
    if any(result_dir.glob("slurm-*.out")):
        return True
    for dirname in ("item_runs", "runs", "figures", "animations"):
        if (result_dir / dirname).exists():
            return True
    return False


def _missing_local_sync_artifacts(
    result_dir: str | Path,
    *,
    expected_files: tuple[str, ...] | None = None,
) -> list[str]:
    """Return missing required local sync artifacts, or an empty list when the sync looks usable."""
    result_dir = Path(result_dir)
    if expected_files:
        missing = [
            name
            for name in expected_files
            if not _local_sync_artifact_is_usable(result_dir / name)
        ]
        if missing:
            return missing
        return []
    if _local_result_dir_has_remote_sync_artifacts(result_dir):
        return []
    return ["remote result artifacts"]


def _should_use_incremental_sweep_final_sync(
    manifest_items: list[dict[str, Any]],
    *,
    local_runs_dir: str | Path,
) -> bool:
    """Return True when most sweep payloads already exist locally and a bulk root sync is wasteful."""
    total = len(manifest_items)
    if total <= 0:
        return True
    local_runs_dir = Path(local_runs_dir)
    ready = sum(
        1
        for item in manifest_items
        if (
            (resolved := _resolve_local_sweep_item_dir(local_runs_dir, str(item["label"])))
            is not None
            and _local_sweep_item_sync_complete(resolved)
        )
    )
    return ready > 0 and ready * 2 >= total


def _recover_local_sweep_summary(
    sweep_dir: str | Path,
    *,
    sweep_label: str,
    total_items: int,
) -> dict[str, Any]:
    """Recover one top-level sweep summary from local progress metadata when possible."""
    sweep_dir = Path(sweep_dir)
    progress = _read_json_if_present(sweep_dir / "sim_progress.json") or {}
    finished_items = progress.get("finished_items") or []
    if not isinstance(finished_items, list):
        finished_items = []

    if not finished_items:
        manifest_payload = (
            _read_json_if_present(sweep_dir / "sweep_manifest.json")
            or _read_json_if_present(sweep_dir / "sweep_manifest.submit.json")
            or {}
        )
        if isinstance(manifest_payload, dict):
            manifest_items = manifest_payload.get("items") or []
        elif isinstance(manifest_payload, list):
            manifest_items = manifest_payload
        else:
            manifest_items = []
        item_runs_dir = sweep_dir / "item_runs"
        for item in manifest_items:
            if not isinstance(item, dict) or not item.get("label"):
                continue
            label = str(item["label"])
            result_dir = _resolve_local_sweep_item_dir(item_runs_dir, label)
            if result_dir is None:
                continue
            finished_items.append(
                {
                    "index": int(item.get("index", len(finished_items))),
                    "label": label,
                    "ok": _local_result_dir_has_loadable_payload(result_dir),
                    "result_dir": str(result_dir),
                    "value": item.get("value"),
                    "recovered_from": "local_item_payload_scan",
                }
            )

    if not finished_items:
        return {}
    completed_items = [item for item in finished_items if bool(item.get("ok", False))]
    failed_items = [item for item in finished_items if not bool(item.get("ok", False))]
    summary = {
        "kind": "remote_sweep",
        "sweep_label": sweep_label,
        "total_items": int(total_items),
        "completed_items": completed_items,
        "failed_items": failed_items,
        "items": finished_items,
        "partial": len(finished_items) < int(total_items),
        "recovered_from": "sim_progress.json",
    }
    for key in ("pending_labels", "running_items", "completed_labels", "failed_labels"):
        if key in progress:
            summary[key] = progress[key]
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


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
    soma_path = find_soma_trace_artifact(result_dir)
    if soma_path is not None and soma_path.exists():
        files[soma_path.name] = {"size_bytes": int(soma_path.stat().st_size)}
    for filename in ("input_times.pkl", "gc_output_events.pkl", "lfp.pkl"):
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


def _remote_helper_script_path(remote_helper_dir: PurePosixPath | None, script_name: str) -> PurePosixPath | None:
    """Return one uploaded remote-helper path when a cache directory is available."""
    return _neuroinfra_remote_helper_script_path(remote_helper_dir, script_name)


def _remote_python_exec_prefix() -> str:
    """Return the remote shell prefix that resolves python3/python and execs it."""
    return _neuroinfra_remote_python_exec_prefix()


def _build_remote_python_file_command(script_path: PurePosixPath, argv: list[str]) -> str:
    """Build a remote shell command that executes one uploaded helper script."""
    return _neuroinfra_build_remote_python_file_command(script_path, argv)


def _build_remote_python_inline_command(script_path: Path, argv: list[str]) -> str:
    """Build a remote shell command that executes one helper script inline."""
    return _neuroinfra_build_remote_python_inline_command(script_path, argv)


def _build_remote_allocation_submit_command(
    config: dict[str, Any],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> tuple[str, PurePosixPath, str]:
    """Build the remote helper invocation that submits one reusable allocation."""
    remote_helper = REPO_ROOT / "tools" / "remote" / "submit_slurm_allocation.py"
    allocation_key = _slurm_allocation_cache_key(config)
    allocation_root = _remote_results_root(config) / ".obgpu-allocations" / allocation_key
    allocation_name_base = str(config.get("slurm_allocation_name") or "obgpu_notebook_alloc")
    allocation_name = f"{allocation_name_base[:100]}_{allocation_key[:8]}"
    allocation_time = config.get("slurm_allocation_time") or config.get("slurm_time")
    argv = _neuroinfra_build_submit_slurm_allocation_argv(
        allocation_root=allocation_root,
        allocation_name=allocation_name,
        heartbeat_timeout_s=_remote_heartbeat_timeout_s(config),
        partition=config.get("slurm_partition"),
        account=config.get("slurm_account"),
        time_limit=allocation_time,
        mem=config.get("slurm_mem"),
        gpus=config.get("slurm_gpus"),
        cpus_per_task=config.get("slurm_cpus_per_task"),
        sbatch_args=config.get("slurm_extra_args", []),
    )
    command = _neuroinfra_build_remote_helper_launch_command(
        remote_helper,
        argv,
        remote_helper_dir=remote_helper_dir,
    )
    return command, allocation_root, allocation_name


def _build_remote_touch_command(path_value: str | PurePosixPath) -> str:
    """Build a remote command that refreshes one heartbeat path."""
    return _neuroinfra_build_remote_touch_command(path_value)


def _refresh_remote_heartbeat(
    config: dict[str, Any],
    heartbeat_path: str | PurePosixPath | None,
    *,
    warn: bool = False,
) -> bool:
    """Best-effort refresh of a remote notebook heartbeat file."""
    return _remote_allocation_runtime_context(config).refresh_heartbeat(
        heartbeat_path,
        warn=warn,
    )


def _build_remote_cleanup_allocations_command(
    config: dict[str, Any],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> str:
    """Build a remote command that cancels stale notebook-managed allocations."""
    cleanup_root = _remote_results_root(config) / ".obgpu-allocations"
    remote_helper = REPO_ROOT / "tools" / "remote" / "cleanup_stale_allocations.py"
    argv = _neuroinfra_build_cleanup_stale_allocations_argv(
        cleanup_root=cleanup_root,
        default_timeout_s=_remote_heartbeat_timeout_s(config),
    )
    return _neuroinfra_build_remote_helper_launch_command(
        remote_helper,
        argv,
        remote_helper_dir=remote_helper_dir,
    )


def _cleanup_stale_remote_slurm_allocations(
    config: dict[str, Any],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> list[dict[str, Any]]:
    """Cancel stale remote notebook-managed reusable allocations before a new run."""
    return _remote_allocation_runtime_context(config).cleanup_stale_allocations(
        remote_helper_dir=remote_helper_dir,
    )


def _maybe_cleanup_stale_remote_slurm_allocations(
    config: dict[str, Any],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> list[dict[str, Any]]:
    """Throttle stale-allocation cleanup so warm sessions do not repeat the same scan."""
    return _remote_allocation_runtime_context(config).maybe_cleanup_stale_allocations(
        remote_helper_dir=remote_helper_dir,
    )


def _build_remote_allocation_discovery_command(
    config: dict[str, Any],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> tuple[str, PurePosixPath, str]:
    """Build the remote command that returns allocation metadata for one config, if present."""
    _submit_command, allocation_root, allocation_name = _build_remote_allocation_submit_command(
        config,
        remote_helper_dir=remote_helper_dir,
    )
    allocation_json = allocation_root / "allocation.json"
    command = _neuroinfra_build_allocation_discovery_command(allocation_json)
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
    step_ntasks: int | None = None,
    remote_helper_dir: PurePosixPath | None = None,
) -> str:
    """Build the remote `submit_sol_run.py` invocation shell line."""
    resolved_step_ntasks = 1
    if config.get("slurm_allocation_job_id") not in (None, ""):
        if step_ntasks is None:
            resolved_step_ntasks = max(
                int(config.get("slurm_step_ntasks", 1) or 1),
                int(config.get("nranks", 1) or 1),
            )
        else:
            resolved_step_ntasks = max(int(step_ntasks or 1), 1)
    remote_helper = REPO_ROOT / "tools" / "remote" / "submit_sol_run.py"
    argv = _neuroinfra_build_submit_sol_run_argv(
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        label=label,
        benchmark_command=benchmark_command,
        repo_mode=str(config.get("remote_repo_mode", "shared")),
        remote_mpi_exec=str(remote_mpi_exec),
        conda_activate_cmd=str(config.get("remote_conda_activate_cmd")),
        heartbeat_timeout_s=_remote_heartbeat_timeout_s(config),
        runtime_profiles=config.get("remote_runtime_profiles") or [],
        fallback_conda_activate_cmd=config.get("remote_fallback_conda_activate_cmd"),
        fast_node_feature=config.get("remote_fast_node_feature"),
        mechanism_profile=config.get("remote_mechanism_profile"),
        fallback_mechanism_profile=config.get("remote_fallback_mechanism_profile"),
        remote_git_ref=remote_git_ref,
        remote_git_fetch=bool(config.get("remote_git_fetch", False)),
        remote_git_remote=str(config.get("remote_git_remote", "origin")),
        allocation_job_id=config.get("slurm_allocation_job_id"),
        step_ntasks=resolved_step_ntasks if config.get("slurm_allocation_job_id") not in (None, "") else None,
        partition=config.get("slurm_partition"),
        account=config.get("slurm_account"),
        time_limit=config.get("slurm_time"),
        mem=config.get("slurm_mem"),
        gpus=config.get("slurm_gpus"),
        cpus_per_task=config.get("slurm_cpus_per_task"),
        sbatch_args=config.get("slurm_extra_args", []),
    )
    return _neuroinfra_build_remote_helper_launch_command(
        remote_helper,
        argv,
        remote_helper_dir=remote_helper_dir,
    )


def _build_remote_poll_command(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    remote_result_dir: PurePosixPath,
    job_id: str,
    wrapper_dir: str | None = None,
    worktree_path: str | None = None,
    remote_helper_dir: PurePosixPath | None = None,
    include_sacct: bool = True,
    include_tails: bool = True,
) -> str:
    """Build the remote `poll_sol_run.py` invocation shell line."""
    remote_helper = REPO_ROOT / "tools" / "remote" / "poll_sol_run.py"
    argv = _neuroinfra_build_poll_sol_run_argv(
        job_id=str(job_id),
        remote_result_dir=remote_result_dir,
        wrapper_dir=wrapper_dir,
        remote_repo_root=remote_repo_root if worktree_path not in (None, "") else None,
        worktree_path=worktree_path,
        include_sacct=include_sacct,
        include_tails=include_tails,
    )
    return _neuroinfra_build_remote_helper_launch_command(
        remote_helper,
        argv,
        remote_helper_dir=remote_helper_dir,
    )


def _build_remote_preflight_command(
    *,
    remote_repo_root: PurePosixPath,
) -> str:
    """Build one remote shell command that validates Sol-side prerequisites."""
    return _neuroinfra_build_remote_preflight_command(remote_repo_root=remote_repo_root)


def _remote_preflight_cache_key(config: dict[str, Any], remote_repo_root: PurePosixPath) -> str:
    """Return the runtime cache key for one successful remote preflight."""
    return _neuroinfra_remote_preflight_cache_key(
        connection_key=_paramiko_connection_key(config),
        remote_repo_root=remote_repo_root,
        remote_conda_activate_cmd=str(config.get("remote_conda_activate_cmd") or ""),
        helper_signature=_remote_helper_signature(),
    )


def _run_remote_preflight_cached(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run one remote preflight only once per notebook session."""
    cache_key = _remote_preflight_cache_key(config, remote_repo_root)
    return _neuroinfra_run_remote_preflight_cached(
        cache=_LIVE_REMOTE_PREFLIGHTS,
        cache_key=cache_key,
        remote_repo_root=remote_repo_root,
        command=_build_remote_preflight_command(remote_repo_root=remote_repo_root),
        run_command=lambda command: _run_ssh_shell(
            config,
            command,
        ),
    )


def _build_remote_result_listing_command(
    *,
    remote_result_dir: PurePosixPath,
) -> str:
    """Build one remote shell command that lists the synced result directory contents."""
    return _neuroinfra_build_remote_result_listing_command(remote_result_dir=remote_result_dir)


def _build_remote_cancel_command(*, job_id: str) -> str:
    """Build one remote shell command that cancels a submitted Slurm job."""
    return _neuroinfra_build_remote_cancel_command(job_id=job_id)


def _query_remote_slurm_job_state(config: dict[str, Any], job_id: str) -> dict[str, str]:
    """Query one remote Slurm job state without requiring a result directory."""
    return _neuroinfra_query_remote_slurm_job_state(
        job_id=str(job_id),
        run_command=lambda command: _run_ssh_shell(config, command),
    )


def _ensure_cached_remote_slurm_allocation(
    config: dict[str, Any],
    *,
    remote_helper_dir: PurePosixPath | None = None,
) -> dict[str, Any]:
    """Acquire or reuse one notebook-cached remote Slurm allocation."""
    return _remote_allocation_runtime_context(config).ensure_cached_allocation(
        remote_helper_dir=remote_helper_dir,
    )


def release_remote_slurm_allocation(config: dict[str, Any]) -> bool:
    """Cancel and forget the cached or remotely-discovered reusable Slurm allocation."""
    return _remote_allocation_runtime_context(config).release_allocation()


def _remote_submission_payload(
    config: dict[str, Any],
    *,
    label: str,
    remote_helper_dir: PurePosixPath | None = None,
    overrides_file: str | PurePosixPath | None = None,
    param_overrides: dict[str, Any] | None = None,
    input_spec_file: str | Path | None = None,
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
        overrides_file=overrides_file,
        param_overrides=param_overrides,
        input_spec_file=input_spec_file,
    )
    submit_command = _build_remote_submit_command(
        config,
        label=label,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_results_root,
        benchmark_command=remote_command,
        remote_mpi_exec=str(remote_mpi_exec),
        remote_git_ref=remote_git_ref,
        step_ntasks=max(
            int(config.get("slurm_step_ntasks", 1) or 1),
            int(config.get("nranks", 1) or 1),
        ),
        remote_helper_dir=remote_helper_dir,
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
    return _neuroinfra_remote_status_has_artifacts(status)


def _create_git_bundle_for_commit(commit_sha: str, *, exclude_ref: str | None = None) -> tuple[Path, str]:
    """Create a temporary git bundle for the requested commit."""
    return _neuroinfra_create_git_bundle_for_commit(REPO_ROOT, commit_sha, exclude_ref=exclude_ref)


def _remote_notebook_tracking_ref_for_source(source_ref: str) -> str | None:
    """Return the stable remote notebook ref for one published local branch tip."""
    return _neuroinfra_remote_notebook_tracking_ref_for_source(source_ref)


def _resolve_remote_tracking_bundle_base(
    config: dict[str, Any],
    *,
    remote_repo_root: PurePosixPath,
    commit_sha: str,
    source_ref: str | None,
) -> str | None:
    """Return the last notebook-published branch tip on the remote when it is a valid local ancestor."""
    if not source_ref:
        return None
    tracking_ref = _remote_notebook_tracking_ref_for_source(source_ref)
    if not tracking_ref:
        return None
    command = (
        f"git -C {shlex.quote(remote_repo_root.as_posix())} "
        f"rev-parse --verify {shlex.quote(tracking_ref + '^{commit}')}"
    )
    completed = _run_ssh_shell(config, command)
    if completed.returncode != 0:
        return None
    base_sha = (completed.stdout or "").strip().splitlines()
    if not base_sha:
        return None
    resolved_base = base_sha[-1].strip()
    if not resolved_base or resolved_base == commit_sha:
        return None
    if not _git_ref_is_ancestor(resolved_base, commit_sha):
        return None
    return resolved_base


def _build_remote_git_bundle_fetch_command(
    *,
    remote_repo_root: PurePosixPath,
    remote_bundle_path: str,
    source_ref: str,
    remote_git_ref: str,
) -> str:
    """Build the remote git fetch command used to publish one local bundle."""
    return _neuroinfra_build_remote_git_bundle_fetch_command(
        remote_repo_root=remote_repo_root,
        remote_bundle_path=remote_bundle_path,
        source_ref=source_ref,
        remote_git_ref=remote_git_ref,
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


def _upload_paramiko_file_via_shell(
    config: dict[str, Any],
    *,
    local_path: Path,
    remote_path: str,
    progress_desc: str,
) -> subprocess.CompletedProcess[str]:
    """Upload one local file over the active Paramiko shell transport without opening SFTP."""
    connection = _connect_paramiko(config)
    transport = connection["transport"]
    channel = None
    stderr_chunks: list[bytes] = []
    bytes_written = 0
    upload_timeout_s = _remote_ssh_upload_timeout_s(config)
    progress = _ProgressBar(
        total=max(int(local_path.stat().st_size), 0),
        desc=progress_desc,
        unit="B",
        unit_scale=True,
        display_step=10 * 1024 * 1024,
    )
    try:
        channel = transport.open_session()
        if upload_timeout_s is not None:
            channel.settimeout(upload_timeout_s)
        remote_shell_command = f"cat > {shlex.quote(remote_path)}"
        channel.exec_command(f"bash -lc {shlex.quote(remote_shell_command)}")
        with local_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while len(view):
                    sent = channel.send(view)
                    if sent <= 0:
                        raise TimeoutError(
                            "Paramiko upload channel accepted no bytes while writing "
                            f"{local_path} to {remote_path}."
                        )
                    view = view[sent:]
                    bytes_written += sent
                    progress.update_to(bytes_written)
                while channel.recv_stderr_ready():
                    stderr_chunks.append(channel.recv_stderr(65536))
        channel.shutdown_write()
        while not channel.exit_status_ready():
            while channel.recv_stderr_ready():
                stderr_chunks.append(channel.recv_stderr(65536))
            time.sleep(0.05)
        while channel.recv_stderr_ready():
            stderr_chunks.append(channel.recv_stderr(65536))
        progress.update_to(bytes_written)
        progress.close()
        return subprocess.CompletedProcess(
            args=["paramiko-upload", str(local_path), remote_path],
            returncode=channel.recv_exit_status(),
            stdout="",
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        )
    except socket.timeout as exc:
        progress.close()
        raise TimeoutError(
            "Timed out while uploading a file over the notebook Paramiko transport.\n"
            f"Local file: {local_path}\n"
            f"Remote file: {remote_path}\n"
            f"Bytes written before timeout: {bytes_written}\n"
            f"Timeout: {upload_timeout_s}s"
        ) from exc
    except Exception as exc:
        progress.close()
        if not _paramiko_transport_is_usable(transport):
            if (
                bool(config.get("remote_preserve_paramiko_session", True))
                and _paramiko_connection_key(config) in _LIVE_PARAMIKO_AUTHENTICATED_KEYS
            ):
                raise RuntimeError(
                    _paramiko_midrun_reauth_error(config) + f"\nOriginal error: {exc}"
                ) from exc
            _drop_paramiko_connection(config)
        raise
    finally:
        progress.close()
        if channel is not None:
            channel.close()


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

    _remote_transport(config)

    branch_name = _resolve_local_git_branch()
    source_branch_ref = (
        f"refs/heads/{branch_name}"
        if branch_name and _git_ref_points_to_commit(branch_name, remote_git_ref)
        else None
    )
    bundle_base = _resolve_remote_tracking_bundle_base(
        config,
        remote_repo_root=remote_repo_root,
        commit_sha=remote_git_ref,
        source_ref=source_branch_ref,
    )
    bundle_base = _find_remote_git_bundle_base(
        config,
        remote_repo_root=remote_repo_root,
        candidate_shas=_local_git_sync_base_candidates(remote_git_ref),
    ) if bundle_base is None else bundle_base
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
    fetch_command = _build_remote_git_bundle_fetch_command(
        remote_repo_root=remote_repo_root,
        remote_bundle_path=remote_bundle_path,
        source_ref=source_ref,
        remote_git_ref=remote_git_ref,
    )

    try:
        _progress_write(f"[Sol remote] Uploading git bundle for commit {remote_git_ref[:12]}...")
        upload_completed = _upload_paramiko_file_via_shell(
            config,
            local_path=bundle_path,
            remote_path=remote_bundle_path,
            progress_desc="[Sol remote] Upload git bundle",
        )
        if upload_completed.returncode != 0:
            raise RuntimeError(
                "Could not upload the local git bundle to the Sol backend over the notebook SSH transport.\n"
                f"Remote bundle: {remote_bundle_path}\n"
                f"Commit: {remote_git_ref}\n"
                f"Stderr:\n{upload_completed.stderr}"
            )
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
            _run_paramiko_shell(config, f"rm -f {shlex.quote(remote_bundle_path)}")
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
    if bool(effective_config.get("remote_defer_soma_vs_sync", False)):
        _progress_write(
            "[OBGPU load] remote_defer_soma_vs_sync=True is deprecated and ignored; "
            "raw soma traces will be synced with the main result payload."
        )
        effective_config["remote_defer_soma_vs_sync"] = False
    remote_repo_root = _remote_repo_root(effective_config)
    remote_git_ref = _resolve_remote_git_ref(effective_config)
    param_overrides, input_spec_file = _benchmark_param_overrides_payload(effective_config)
    remote_overrides_path = _remote_benchmark_overrides_path(effective_config, label)
    notebook_timings: dict[str, float] = {}
    remote_helper_dir: PurePosixPath | None = None
    (
        _remote_repo_root_value,
        _remote_results_root_value,
        remote_benchmark_command,
        remote_metadata,
        submit_shell,
    ) = _remote_submission_payload(
        effective_config,
        label=label,
        overrides_file=remote_overrides_path,
        param_overrides=param_overrides,
        input_spec_file=input_spec_file,
    )
    started = time.perf_counter()
    _ensure_remote_git_ref_available(
        effective_config,
        remote_repo_root=remote_repo_root,
        remote_git_ref=remote_git_ref,
    )
    _record_timing(notebook_timings, "git_publish_s", started)
    _progress_write("[Sol remote] Running remote preflight checks...")
    started = time.perf_counter()
    preflight_completed, preflight_cached = _run_remote_preflight_cached(
        effective_config,
        remote_repo_root=remote_repo_root,
    )
    _record_timing(notebook_timings, "preflight_s", started)
    remote_metadata["preflight_cached"] = bool(preflight_cached)
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

    started = time.perf_counter()
    remote_helper_dir = _ensure_remote_helper_cache(effective_config)
    _record_timing(notebook_timings, "helper_cache_s", started)
    if remote_helper_dir is not None:
        helper_cache_meta = _LIVE_REMOTE_HELPER_CACHES.get(_remote_helper_cache_runtime_key(effective_config)) or {}
        (
            _remote_repo_root_value,
            _remote_results_root_value,
            remote_benchmark_command,
            remote_metadata,
            submit_shell,
        ) = _remote_submission_payload(
            effective_config,
            label=label,
            remote_helper_dir=remote_helper_dir,
            overrides_file=remote_overrides_path,
            param_overrides=param_overrides,
            input_spec_file=input_spec_file,
        )
        remote_metadata["remote_helper_dir"] = remote_helper_dir.as_posix()
        remote_metadata["remote_helper_cache_hit"] = bool(helper_cache_meta.get("cache_hit", False))

    started = time.perf_counter()
    cleanup_actions = _maybe_cleanup_stale_remote_slurm_allocations(
        effective_config,
        remote_helper_dir=remote_helper_dir,
    )
    _record_timing(notebook_timings, "allocation_cleanup_s", started)
    remote_metadata["stale_allocation_cleanup_count"] = len(cleanup_actions)

    started = time.perf_counter()
    allocation_info = _ensure_cached_remote_slurm_allocation(
        effective_config,
        remote_helper_dir=remote_helper_dir,
    )
    _record_timing(notebook_timings, "allocation_wait_s", started)
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
        ) = _remote_submission_payload(
            effective_config,
            label=label,
            remote_helper_dir=remote_helper_dir,
            overrides_file=remote_overrides_path,
            param_overrides=param_overrides,
            input_spec_file=input_spec_file,
        )
        if remote_helper_dir is not None:
            remote_metadata["remote_helper_dir"] = remote_helper_dir.as_posix()
        remote_metadata["auto_reused_allocation"] = bool(
            effective_config.get("slurm_reuse_allocation", False)
            and not allocation_info.get("manual", False)
        )
        remote_metadata["allocation_state"] = allocation_info.get("state", "")
        remote_metadata["allocation_reason"] = allocation_info.get("reason", "")
        remote_metadata["allocation_location"] = allocation_info.get("location", "")
        remote_metadata["allocation_heartbeat_path"] = allocation_heartbeat_path

    _progress_write("[Sol remote] Uploading benchmark overrides file...")
    started = time.perf_counter()
    _upload_remote_text_file(
        effective_config,
        remote_path=remote_overrides_path,
        text=json.dumps(_json_ready(param_overrides), indent=2, sort_keys=True),
    )
    _record_timing(notebook_timings, "overrides_upload_s", started)
    remote_metadata["benchmark_overrides_file"] = remote_overrides_path.as_posix()

    _progress_write("[Sol remote] Submitting Slurm job...")
    started = time.perf_counter()
    submit_completed = _run_ssh_shell(effective_config, submit_shell)
    _record_timing(notebook_timings, "submit_s", started)
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
    log_poll_interval_s = max(
        float(effective_config.get("remote_log_poll_interval_s", max(poll_interval_s, 5.0))),
        poll_interval_s,
    )
    live_status = bool(effective_config.get("remote_live_status", True))
    live_logs = bool(effective_config.get("remote_live_logs", True))
    monitor_result = _neuroinfra_monitor_remote_run(
        job_id=str(submission["job_id"]),
        poll_interval_s=poll_interval_s,
        log_poll_interval_s=log_poll_interval_s,
        live_status=live_status,
        live_logs=live_logs,
        missing_artifact_retry_limit=3,
        hooks=_remote_run_monitor_hooks(
            effective_config=effective_config,
            remote_job_heartbeat_path=remote_job_heartbeat_path,
            allocation_heartbeat_path=allocation_heartbeat_path,
            remote_repo_root=remote_repo_root,
            remote_result_dir=remote_result_dir,
            remote_helper_dir=remote_helper_dir,
            notebook_timings=notebook_timings,
            submission=submission,
            local_result_dir=local_result_dir,
        ),
    )
    final_status = monitor_result.final_status
    poll_transcript = monitor_result.poll_transcript

    deferred_remote_artifacts: list[str] = []
    final_sync_include_files: tuple[str, ...] | None = None
    if final_status and final_status.get("ok") and bool(effective_config.get("remote_defer_soma_vs_sync", False)):
        final_sync_include_files = _remote_fast_sync_files(effective_config)
        deferred_remote_artifacts.append(preferred_soma_trace_artifact_name())
    artifact_result = _neuroinfra_finalize_remote_run_artifacts(
        effective_config,
        final_status=final_status,
        local_result_dir=local_result_dir,
        remote_result_dir=remote_result_dir,
        wrapper_dir=submission.get("wrapper_dir"),
        label=label,
        timestamp=timestamp,
        notebook_timings=notebook_timings,
        poll_transcript=poll_transcript,
        include_files=final_sync_include_files,
        deferred_remote_artifacts=deferred_remote_artifacts,
        hooks=_remote_run_artifact_hooks(notebook_timings),
    )
    final_status = artifact_result.final_status
    sync_warning = artifact_result.sync_warning
    stdout_text = artifact_result.stdout_text
    stderr_text = artifact_result.stderr_text
    bootstrap_text = artifact_result.bootstrap_text
    slurm_text = artifact_result.slurm_text
    remote_listing_text = artifact_result.remote_listing_text
    remote_git_commit = artifact_result.remote_git_commit
    remote_git_ref = artifact_result.remote_git_ref or remote_metadata.get("remote_git_ref")
    returncode = artifact_result.returncode
    completed = SimpleNamespace(returncode=returncode, stdout=stdout_text, stderr=stderr_text)
    summary = artifact_result.summary
    compact_poll_events = artifact_result.compact_poll_events
    poll_events_path = artifact_result.poll_events_path
    artifact_sizes = artifact_result.artifact_sizes
    remote_metadata["deferred_remote_artifacts"] = list(artifact_result.deferred_remote_artifacts)
    remote_metadata["artifact_sizes"] = artifact_sizes
    remote_metadata["notebook_timing_seconds"] = notebook_timings

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
                "artifact_sizes": artifact_sizes,
                "notebook_timing_seconds": notebook_timings,
            }
        },
    )
    timing_summary = _timing_summary_text(notebook_timings)
    if timing_summary:
        _progress_write(f"[OBGPU load] Notebook pipeline timings: {timing_summary}")

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


def run_simulation(
    config: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> RunRecord:
    """Run one timestamped notebook simulation and return its recorded metadata."""
    config = build_run_config(**(config or {}))
    timestamp = make_timestamp()
    label = str(label or make_label(config, timestamp=timestamp))
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

    result_dir.mkdir(parents=True, exist_ok=True)
    param_overrides, input_spec_file = _benchmark_param_overrides_payload(config)
    overrides_path = result_dir.parent / ".obgpu-wrapper" / label / "overrides.json"
    _write_benchmark_overrides_file(overrides_path, param_overrides)
    command = build_run_command(
        config,
        label,
        overrides_file=overrides_path,
        param_overrides=param_overrides,
        input_spec_file=input_spec_file,
    )
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


def _prepare_sweep_plan(
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
    *,
    grid: bool = False,
) -> dict[str, Any]:
    """Normalize one sweep request into explicit per-item configs and labels."""
    from itertools import product as _product

    base_config = build_run_config(**deepcopy(base_config))
    timestamp = make_timestamp()
    sweep_label = _make_sweep_label(base_config, sweep_path=sweep_path, timestamp=timestamp)

    items: list[dict[str, Any]] = []
    normalized_values: list[Any] = []

    if grid:
        if not isinstance(sweep_path, dict):
            raise TypeError("Grid sweeps require a dict of {path: values}")
        paths = list(sweep_path.keys())
        value_lists = list(sweep_path.values())
        iterable = [dict(zip(paths, combo)) for combo in _product(*value_lists)]
    elif isinstance(sweep_path, dict):
        paths = list(sweep_path.keys())
        value_lists = list(sweep_path.values())
        lengths = [len(v) for v in value_lists]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All parameter lists must have the same length for a joint sweep; "
                f"got lengths {dict(zip(paths, lengths))}"
            )
        iterable = [dict(zip(paths, combo)) for combo in zip(*value_lists)]
    else:
        if values is None:
            raise ValueError("values must be provided for single-axis sweeps")
        iterable = list(values)

    for index, value in enumerate(iterable):
        sweep_config = deepcopy(base_config)
        if isinstance(value, dict):
            for path, path_value in value.items():
                set_path_value(sweep_config, path, path_value)
            item_value = dict(value)
        else:
            set_path_value(sweep_config, sweep_path, value)
            item_value = value
        item_label = _make_sweep_item_label(
            base_config,
            sweep_path=sweep_path,
            timestamp=timestamp,
            index=index,
        )
        items.append(
            {
                "index": index,
                "value": item_value,
                "config": sweep_config,
                "label": item_label,
            }
        )
        normalized_values.append(item_value)

    return {
        "path": sweep_path,
        "values": normalized_values,
        "items": items,
        "paramset": base_config.get("paramset"),
        "timestamp": timestamp,
        "sweep_label": sweep_label,
        "base_config": base_config,
        "grid": {p: list(v) for p, v in sweep_path.items()} if grid and isinstance(sweep_path, dict) else None,
    }


def _write_sweep_info(
    sweep: dict[str, Any],
    *,
    sweep_dir: str | Path,
    timestamp: str,
) -> Path:
    """Persist sweep metadata for both local and remote-batch sweep runs."""
    return _neuroinfra_write_sweep_info(
        sweep,
        sweep_dir=sweep_dir,
        timestamp=timestamp,
        json_ready=_json_ready,
        resolve_git_head=_resolve_local_git_head,
    )


def _build_remote_sweep_driver_command(
    config: dict[str, Any],
    *,
    sweep_plan: dict[str, Any],
    remote_repo_root: PurePosixPath,
    remote_sweep_root: PurePosixPath,
) -> tuple[list[str], list[dict[str, Any]], str, PurePosixPath, int]:
    """Build the one driver command that runs an entire sweep inside one Slurm job."""
    remote_driver = Path(remote_repo_root) / "tools" / "remote" / "remote_sweep_driver.py"
    remote_runs_root = remote_sweep_root / "item_runs"
    remote_manifest_path = remote_sweep_root / "sweep_manifest.submit.json"
    remote_mpi_exec = str(config.get("remote_mpi_exec") or default_remote_mpi_exec())
    tasks_per_item = max(int(config.get("nranks", 1) or 1), 1)
    max_concurrent = _remote_sweep_parallelism(config, tasks_per_item=tasks_per_item)

    manifest_items: list[dict[str, Any]] = []
    for item in sweep_plan["items"]:
        remote_result_dir = remote_runs_root / item["label"]
        item_param_overrides, item_input_spec_file = _benchmark_param_overrides_payload(item["config"])
        item_overrides_file = remote_sweep_root / "overrides" / f"{item['label']}.json"
        benchmark_command = build_run_command(
            item["config"],
            item["label"],
            repo_root=remote_repo_root,
            results_base=remote_runs_root,
            mpi_exec=remote_mpi_exec,
            include_mpi_launcher=True,
            overrides_file=item_overrides_file,
            param_overrides=item_param_overrides,
            input_spec_file=item_input_spec_file,
        )
        manifest_items.append(
            {
                "index": int(item["index"]),
                "label": str(item["label"]),
                "value": _json_ready(item["value"]),
                "result_dir": remote_result_dir.as_posix(),
                "command": benchmark_command,
                "overrides_file": item_overrides_file.as_posix(),
                "overrides": _json_ready(item_param_overrides),
            }
        )

    manifest_json = json.dumps(manifest_items, indent=2, sort_keys=True)
    driver_command = [
        "python3",
        str(remote_driver),
        "--repo-root",
        remote_repo_root.as_posix(),
        "--sweep-root",
        remote_sweep_root.as_posix(),
        "--items-json",
        remote_manifest_path.as_posix(),
        "--max-concurrent",
        str(max_concurrent),
    ]
    return driver_command, manifest_items, manifest_json, remote_manifest_path, max_concurrent


def _finalize_synced_sweep_item(
    *,
    item: dict[str, Any],
    local_result_dir: Path,
    timestamp: str,
    remote_payload: dict[str, Any],
    returncode: int,
) -> tuple[RunRecord, dict[str, Any]]:
    """Write run_info for one synced sweep item, then return the standard run/result pair."""
    stdout = (local_result_dir / "stdout.txt").read_text() if (local_result_dir / "stdout.txt").exists() else ""
    stderr = (local_result_dir / "stderr.txt").read_text() if (local_result_dir / "stderr.txt").exists() else ""
    summary = _read_json_if_present(local_result_dir / "summary.json")
    completed = SimpleNamespace(returncode=int(returncode), stdout=stdout, stderr=stderr)
    _write_notebook_run_info(
        local_result_dir,
        config=item["config"],
        label=item["label"],
        timestamp=timestamp,
        command=item["command"],
        env={},
        completed=completed,
        summary=summary,
        extra_payload={
            "remote": remote_payload,
            "sweep_item": {
                "index": int(item["index"]),
                "value": _json_ready(item["value"]),
            },
        },
    )
    run = load_run_record(local_result_dir)
    result = load_result(run)
    return run, result


def _run_remote_sweep(
    sweep_plan: dict[str, Any],
) -> dict[str, Any]:
    """Run a remote sweep as one Slurm job, with optional concurrent in-job steps."""
    base_config = dict(sweep_plan["base_config"])
    effective_config = dict(base_config)
    notebook_timings: dict[str, float] = {}
    remote_helper_dir: PurePosixPath | None = None
    sweep_label = str(sweep_plan["sweep_label"])
    timestamp = str(sweep_plan["timestamp"])
    local_sweep_dir = _sweep_dir(effective_config, sweep_label)
    local_runs_dir = _sweep_item_runs_dir(effective_config, sweep_label)
    local_sweep_dir.mkdir(parents=True, exist_ok=True)
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    remote_repo_root = _remote_repo_root(effective_config)
    remote_git_ref = _resolve_remote_git_ref(effective_config)
    remote_sweeps_root = _remote_results_root(effective_config) / "sweeps"
    remote_sweep_root = remote_sweeps_root / sweep_label
    (
        remote_driver_command,
        manifest_items,
        manifest_json,
        remote_manifest_path,
        max_concurrent,
    ) = _build_remote_sweep_driver_command(
        effective_config,
        sweep_plan=sweep_plan,
        remote_repo_root=remote_repo_root,
        remote_sweep_root=remote_sweep_root,
    )
    remote_metadata = {
        "runner_backend": str(effective_config.get("runner_backend", "slurm_remote")),
        "remote_host": _require_remote_host(effective_config),
        "remote_repo_root": remote_repo_root.as_posix(),
        "remote_results_root": remote_sweeps_root.as_posix(),
        "remote_mpi_exec": str(effective_config.get("remote_mpi_exec") or default_remote_mpi_exec()),
        "remote_repo_mode": str(effective_config.get("remote_repo_mode", "shared")),
        "remote_git_ref": remote_git_ref,
        "remote_git_fetch": bool(effective_config.get("remote_git_fetch", False)),
        "remote_git_remote": str(effective_config.get("remote_git_remote", "origin")),
        "sweep_label": sweep_label,
        "sweep_parallelism": int(max_concurrent),
        "sweep_items": len(manifest_items),
    }
    (local_sweep_dir / "sweep_manifest.submit.json").write_text(manifest_json)

    started = time.perf_counter()
    _ensure_remote_git_ref_available(
        effective_config,
        remote_repo_root=remote_repo_root,
        remote_git_ref=remote_git_ref,
    )
    _record_timing(notebook_timings, "git_publish_s", started)
    _progress_write("[Sol remote] Running remote preflight checks for sweep...")
    started = time.perf_counter()
    preflight_completed, preflight_cached = _run_remote_preflight_cached(
        effective_config,
        remote_repo_root=remote_repo_root,
    )
    _record_timing(notebook_timings, "preflight_s", started)
    remote_metadata["preflight_cached"] = bool(preflight_cached)
    if preflight_completed.returncode != 0:
        raise RuntimeError(
            "Remote sweep preflight failed.\n"
            f"Stdout:\n{preflight_completed.stdout}\n\n"
            f"Stderr:\n{preflight_completed.stderr}"
        )

    started = time.perf_counter()
    remote_helper_dir = _ensure_remote_helper_cache(effective_config)
    _record_timing(notebook_timings, "helper_cache_s", started)
    if remote_helper_dir is not None:
        remote_metadata["remote_helper_dir"] = remote_helper_dir.as_posix()
        helper_cache_meta = _LIVE_REMOTE_HELPER_CACHES.get(_remote_helper_cache_runtime_key(effective_config)) or {}
        remote_metadata["remote_helper_cache_hit"] = bool(helper_cache_meta.get("cache_hit", False))

    started = time.perf_counter()
    cleanup_actions = _maybe_cleanup_stale_remote_slurm_allocations(
        effective_config,
        remote_helper_dir=remote_helper_dir,
    )
    _record_timing(notebook_timings, "allocation_cleanup_s", started)
    remote_metadata["stale_allocation_cleanup_count"] = len(cleanup_actions)

    started = time.perf_counter()
    allocation_info = _ensure_cached_remote_slurm_allocation(
        effective_config,
        remote_helper_dir=remote_helper_dir,
    )
    _record_timing(notebook_timings, "allocation_wait_s", started)
    allocation_heartbeat_path = None
    if allocation_info.get("job_id") not in (None, ""):
        effective_config["slurm_allocation_job_id"] = str(allocation_info["job_id"])
        allocation_heartbeat_path = allocation_info.get("heartbeat_path")
        remote_metadata["auto_reused_allocation"] = bool(
            effective_config.get("slurm_reuse_allocation", False)
            and not allocation_info.get("manual", False)
        )
        remote_metadata["allocation_state"] = allocation_info.get("state", "")
        remote_metadata["allocation_reason"] = allocation_info.get("reason", "")
        remote_metadata["allocation_location"] = allocation_info.get("location", "")
        remote_metadata["allocation_heartbeat_path"] = allocation_heartbeat_path

    _progress_write("[Sol remote] Uploading remote sweep manifest...")
    started = time.perf_counter()
    _upload_remote_text_file(
        effective_config,
        remote_path=remote_manifest_path,
        text=manifest_json,
    )
    _record_timing(notebook_timings, "manifest_upload_s", started)
    remote_metadata["sweep_manifest_path"] = remote_manifest_path.as_posix()

    submit_shell = _build_remote_submit_command(
        effective_config,
        label=sweep_label,
        remote_repo_root=remote_repo_root,
        remote_results_root=remote_sweeps_root,
        benchmark_command=remote_driver_command,
        remote_mpi_exec=str(effective_config.get("remote_mpi_exec") or default_remote_mpi_exec()),
        remote_git_ref=remote_git_ref,
        step_ntasks=1,
        remote_helper_dir=remote_helper_dir,
    )

    _progress_write("[Sol remote] Submitting remote sweep batch job...")
    started = time.perf_counter()
    submit_completed = _run_ssh_shell(effective_config, submit_shell)
    _record_timing(notebook_timings, "submit_s", started)
    (local_sweep_dir / "submit_stdout.txt").write_text(submit_completed.stdout or "")
    (local_sweep_dir / "submit_stderr.txt").write_text(submit_completed.stderr or "")
    if submit_completed.returncode != 0:
        raise RuntimeError(
            "Remote sweep submission failed.\n"
            f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
        )

    try:
        submission = json.loads((submit_completed.stdout or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Remote sweep submission did not return valid JSON.\n"
            f"Stdout:\n{submit_completed.stdout}\n\nStderr:\n{submit_completed.stderr}"
        ) from exc

    remote_job_heartbeat_path = submission.get("heartbeat_path")
    remote_metadata["job_heartbeat_path"] = remote_job_heartbeat_path
    remote_metadata["heartbeat_timeout_s"] = submission.get(
        "heartbeat_timeout_s",
        _remote_heartbeat_timeout_s(effective_config),
    )

    manifest_by_label = {item["label"]: item for item in manifest_items}
    synced_labels: set[str] = set()
    item_status_by_label: dict[str, dict[str, Any]] = {}
    final_status: dict[str, Any] | None = None
    live_status = bool(effective_config.get("remote_live_status", True))
    poll_interval_s = max(float(effective_config.get("remote_poll_interval_s", 1.0)), 1.0)
    log_poll_interval_s = max(
        float(effective_config.get("remote_log_poll_interval_s", max(poll_interval_s, 5.0))),
        poll_interval_s,
    )
    live_sync_max_items_per_poll = max(
        int(effective_config.get("sweep_live_sync_max_items_per_poll", 8) or 0),
        0,
    )

    def refresh_remote_leases(*, warn: bool = False) -> None:
        _refresh_remote_heartbeat(effective_config, remote_job_heartbeat_path, warn=warn)
        _refresh_remote_heartbeat(effective_config, allocation_heartbeat_path, warn=warn)

    def sync_finished_items(status: dict[str, Any]) -> None:
        progress_payload = status.get("progress_payload") or {}
        pending_labels = progress_payload.get("pending_labels") or []
        running_items = progress_payload.get("running_items") or []
        if not _should_sync_remote_sweep_finished_items(
            effective_config,
            pending_count=len(pending_labels),
            running_count=len(running_items),
        ):
            return
        finished_items = progress_payload.get("finished_items") or []
        synced_this_poll = 0
        for finished in finished_items:
            if not isinstance(finished, dict):
                continue
            label = str(finished.get("label") or "").strip()
            if not label or label in synced_labels or label not in manifest_by_label:
                continue
            if live_sync_max_items_per_poll and synced_this_poll >= live_sync_max_items_per_poll:
                break
            manifest_item = manifest_by_label[label]
            remote_result_dir = PurePosixPath(str(finished.get("result_dir") or manifest_item["result_dir"]))
            local_result_dir = local_runs_dir / label
            refresh_remote_leases()
            sync_completed = _sync_remote_result_dir(
                effective_config,
                remote_result_dir=remote_result_dir,
                local_result_dir=local_result_dir,
                expected_files=("summary.json",),
                include_files=_remote_sweep_item_sync_files(effective_config),
            )
            refresh_remote_leases()
            if sync_completed.returncode != 0:
                continue
            item_status_by_label[label] = dict(finished)
            synced_labels.add(label)
            synced_this_poll += 1

    _progress_write(
        f"[Sol remote] Submitted sweep job {submission['job_id']} "
        f"for {len(manifest_items)} items (parallelism={max_concurrent})."
    )
    monitor_result = _neuroinfra_monitor_remote_sweep(
        job_id=str(submission["job_id"]),
        poll_interval_s=poll_interval_s,
        log_poll_interval_s=log_poll_interval_s,
        live_status=live_status,
        hooks=_remote_sweep_monitor_hooks(
            effective_config=effective_config,
            remote_job_heartbeat_path=remote_job_heartbeat_path,
            allocation_heartbeat_path=allocation_heartbeat_path,
            remote_repo_root=remote_repo_root,
            remote_sweep_root=remote_sweep_root,
            remote_helper_dir=remote_helper_dir,
            notebook_timings=notebook_timings,
            submission=submission,
            synced_labels=synced_labels,
            sync_finished_items_fn=sync_finished_items,
        ),
    )
    final_status = monitor_result.final_status

    sweep_artifacts = _neuroinfra_finalize_remote_sweep_artifacts(
        effective_config,
        final_status=final_status,
        local_sweep_dir=local_sweep_dir,
        local_runs_dir=local_runs_dir,
        remote_sweep_root=remote_sweep_root,
        sweep_label=sweep_label,
        manifest_items=manifest_items,
        item_status_by_label=item_status_by_label,
        hooks=_remote_sweep_artifact_hooks(
            refresh_remote_leases_fn=refresh_remote_leases,
            notebook_timings=notebook_timings,
        ),
    )
    final_sync = sweep_artifacts.final_sync
    sweep_summary = sweep_artifacts.sweep_summary
    item_status_by_label = sweep_artifacts.item_status_by_label
    if final_sync.returncode != 0:
        raise RuntimeError(
            "Remote sweep result sync failed.\n"
            f"Sweep dir: {local_sweep_dir}\n"
            f"Stderr:\n{final_sync.stderr}"
        )

    sweep_items = []
    load_errors: dict[str, str] = {}
    for item in manifest_items:
        plan_item = sweep_plan["items"][int(item["index"])]
        finalize_item = {**item, "config": plan_item["config"], "value": plan_item["value"]}
        local_result_dir = _resolve_local_sweep_item_dir(local_runs_dir, str(item["label"]))
        status_payload = item_status_by_label.get(item["label"], {})
        item_entry = {
            "index": int(item["index"]),
            "label": str(item["label"]),
            "value": plan_item["value"],
            "config": plan_item["config"],
            "run": None,
            "result": None,
            "status": status_payload,
        }
        if local_result_dir is None:
            sweep_items.append(item_entry)
            continue
        if not _local_sync_artifact_is_usable(local_result_dir / "summary.json"):
            summary = _synthesize_partial_sync_summary(
                local_result_dir,
                label=str(item["label"]),
                timestamp=timestamp,
                config=plan_item["config"],
            )
            (local_result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        remote_metadata["notebook_timing_seconds"] = notebook_timings
        try:
            run, result = _finalize_synced_sweep_item(
                item=finalize_item,
                local_result_dir=local_result_dir,
                timestamp=timestamp,
                remote_payload=remote_metadata,
                returncode=int(status_payload.get("returncode", 0) or 0),
            )
        except Exception as exc:
            load_errors[str(item["label"])] = str(exc)
            item_entry["status"] = {**status_payload, "load_error": str(exc)}
        else:
            item_entry["run"] = run
            item_entry["result"] = result
        sweep_items.append(item_entry)

    missing_labels = [
        item["label"]
        for item in manifest_items
        if _resolve_local_sweep_item_dir(local_runs_dir, str(item["label"])) is None
    ]

    sweep = {
        "path": sweep_plan["path"],
        "values": list(sweep_plan["values"]),
        "items": sweep_items,
        "paramset": sweep_plan["paramset"],
    }
    if sweep_plan.get("grid") is not None:
        sweep["grid"] = sweep_plan["grid"]
    _write_sweep_info(sweep, sweep_dir=local_sweep_dir, timestamp=timestamp)
    _merge_run_info_payload(
        local_sweep_dir,
        {
            "remote": {
                **remote_metadata,
                "job_id": submission.get("job_id"),
                "final_status": _summarize_remote_status(final_status),
                "notebook_timing_seconds": notebook_timings,
            }
        },
    )
    timing_summary = _timing_summary_text(notebook_timings)
    if timing_summary:
        _progress_write(f"[OBGPU load] Sweep notebook pipeline timings: {timing_summary}")

    failed_labels = []
    for failed in sweep_summary.get("failed_items", []):
        if isinstance(failed, dict) and failed.get("label"):
            failed_labels.append(str(failed["label"]))
    result_labels = {str(item.get("label")) for item in sweep_items if item.get("result") is not None}
    failed_without_result = [label for label in failed_labels if label not in result_labels]
    recovered_failed_labels = [label for label in failed_labels if label in result_labels]
    loaded_count = sum(1 for item in sweep_items if item.get("result") is not None)
    partial_reasons = []
    if failed_without_result:
        partial_reasons.append(f"{len(failed_without_result)} failed")
    if missing_labels:
        partial_reasons.append(f"{len(missing_labels)} missing")
    if load_errors:
        partial_reasons.append(f"{len(load_errors)} load errors")
    sweep["partial"] = bool(partial_reasons)
    sweep["failed_labels"] = failed_labels
    sweep["failed_without_result"] = failed_without_result
    sweep["recovered_failed_labels"] = recovered_failed_labels
    sweep["missing_labels"] = missing_labels
    sweep["load_errors"] = load_errors
    if partial_reasons:
        _write_sweep_info(sweep, sweep_dir=local_sweep_dir, timestamp=timestamp)
        _progress_write(
            "[OBGPU load] Remote sweep returned partial results: "
            f"{loaded_count}/{len(manifest_items)} usable items "
            f"({', '.join(partial_reasons)})."
        )
    if final_status is not None and not final_status.get("ok", True) and not sweep_summary and loaded_count == 0:
        raise RuntimeError(
            "Remote sweep failed before writing a summary.\n"
            f"Sweep dir: {local_sweep_dir}\n"
            f"State: {final_status.get('state')}"
        )
    return sweep


def run_parameter_sweep(
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Run a parameter sweep locally or via the configured remote sweep engine.

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
    sub-dict of ``{path: value}`` pairs. Remote Slurm backends default to a
    single submitted sweep job unless ``sweep_engine='legacy'`` is requested.
    """
    sweep_plan = _prepare_sweep_plan(base_config, sweep_path, values, grid=False)
    if _sweep_uses_remote_batch_engine(sweep_plan["base_config"]):
        return _run_remote_sweep(sweep_plan)

    local_sweep_dir = _sweep_dir(sweep_plan["base_config"], str(sweep_plan["sweep_label"]))
    local_item_runs_dir = _sweep_item_runs_dir(sweep_plan["base_config"], str(sweep_plan["sweep_label"]))
    local_item_runs_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for item in sweep_plan["items"]:
        sweep_config = deepcopy(item["config"])
        sweep_config["results_base"] = str(local_item_runs_dir)
        run, result = run_and_load(sweep_config, label=str(item["label"]))
        items.append({"value": item["value"], "config": sweep_config, "run": run, "result": result})
    sweep = {
        "path": sweep_plan["path"],
        "values": list(sweep_plan["values"]),
        "items": items,
        "paramset": sweep_plan["paramset"],
    }
    save_sweep(sweep, name=str(sweep_plan["sweep_label"]), base_dir=local_sweep_dir.parent)
    return sweep


def run_grid_sweep(
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
) -> dict[str, Any]:
    """Run every combination of the provided parameter grid.

    Example::

        sweep = run_grid_sweep(config, {'gaba_gmax': [0, 1, 2], 'gap_mc': [16, 32]})
        # 6 runs: (0,16), (0,32), (1,16), (1,32), (2,16), (2,32)

    Items are ordered row-major (first parameter varies slowest). Each item's
    ``value`` is a ``{path: value}`` dict, matching the joint-sweep convention.
    """
    sweep_plan = _prepare_sweep_plan(base_config, param_grid, grid=True)
    if _sweep_uses_remote_batch_engine(sweep_plan["base_config"]):
        return _run_remote_sweep(sweep_plan)

    local_sweep_dir = _sweep_dir(sweep_plan["base_config"], str(sweep_plan["sweep_label"]))
    local_item_runs_dir = _sweep_item_runs_dir(sweep_plan["base_config"], str(sweep_plan["sweep_label"]))
    local_item_runs_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for item in sweep_plan["items"]:
        sweep_config = deepcopy(item["config"])
        sweep_config["results_base"] = str(local_item_runs_dir)
        run, result = run_and_load(sweep_config, label=str(item["label"]))
        items.append({"value": item["value"], "config": sweep_config, "run": run, "result": result})

    sweep = {
        "path": param_grid,
        "values": list(sweep_plan["values"]),
        "items": items,
        "paramset": sweep_plan["paramset"],
        "grid": sweep_plan["grid"],
    }
    save_sweep(sweep, name=str(sweep_plan["sweep_label"]), base_dir=local_sweep_dir.parent)
    return sweep


def load_pickle(path: str | Path) -> Any:
    """Load one saved result artifact from disk."""
    return load_saved_result_artifact(path)


def _sync_deferred_remote_artifact(
    result_dir: str | Path,
    *,
    run_info: dict[str, Any] | None,
    filename: str,
) -> Path:
    """Fetch one deferred remote artifact into the local result directory and return its path."""
    return _neuroinfra_sync_deferred_remote_artifact(
        result_dir,
        run_info=run_info,
        filename=filename,
        hooks=_deferred_remote_artifact_sync_hooks(),
    )


def _sync_deferred_remote_artifact_direct(
    config: dict[str, Any],
    *,
    remote_result_dir: PurePosixPath,
    local_result_dir: Path,
    filename: str,
) -> subprocess.CompletedProcess[str]:
    """Fetch one deferred artifact via a direct SSH-channel byte stream."""
    return _neuroinfra_sync_deferred_remote_artifact_direct(
        config,
        remote_result_dir=remote_result_dir,
        local_result_dir=local_result_dir,
        filename=filename,
        hooks=_deferred_remote_artifact_sync_hooks(),
    )


class LazyResult(_NeuroinfraLazyResult):
    """Notebook result dict that adds progress and artifact-size bookkeeping."""

    def _ensure_loaded(self, key: str) -> None:
        if key not in self._lazy_loaders:
            return
        _progress_write(f"[OBGPU load] Lazy-loading {key}...")
        started = time.perf_counter()
        try:
            super()._ensure_loaded(key)
        finally:
            elapsed_s = time.perf_counter() - started
        if key == "soma_vs":
            soma_path = dict.get(self, "soma_vs_file")
            artifact_sizes = dict.get(self, "artifact_sizes")
            if isinstance(soma_path, Path) and soma_path.exists() and isinstance(artifact_sizes, dict):
                artifact_sizes[soma_path.name] = int(soma_path.stat().st_size)
        _progress_write(f"[OBGPU load] Loaded {key} in {elapsed_s:.1f}s")


def _apply_loaded_lfp_payload(result: MutableMapping[str, Any], loaded: Any) -> None:
    """Apply one loaded LFP payload into the standard result mapping."""
    lfp_t, lfp = loaded
    result["lfp_t"] = np.asarray(lfp_t, dtype=float)
    result["lfp"] = np.asarray(lfp, dtype=float)


_OBGPU_RESULT_VIEW_SCHEMA = _NeuroinfraResultViewSchema(
    fields=(
        _NeuroinfraResultFieldSpec("input_times", default_factory=list),
        _NeuroinfraResultFieldSpec("soma_vs", default_factory=list, lazy_path_key="soma_vs_file"),
        _NeuroinfraResultFieldSpec("soma_spikes", default_factory=dict),
        _NeuroinfraResultFieldSpec("voltage_summary", default_factory=dict),
        _NeuroinfraResultFieldSpec("gc_output_events", default_factory=list),
        _NeuroinfraResultFieldSpec("lfp_t", default_factory=lambda: np.array([])),
        _NeuroinfraResultFieldSpec(
            "lfp",
            default_factory=lambda: np.array([]),
            apply_loaded_fn=_apply_loaded_lfp_payload,
        ),
    ),
    result_type=LazyResult,
)


def _make_result_view(
    *,
    result_dir: Path,
    summary: dict[str, Any] | None,
    run_info: dict[str, Any] | None,
    artifact_sizes: dict[str, int],
) -> LazyResult:
    """Build the standard notebook result mapping before artifact payload loads."""
    return _OBGPU_RESULT_VIEW_SCHEMA.create_result(
        result_dir=result_dir,
        summary=summary,
        run_info=run_info,
        artifact_sizes=artifact_sizes,
    )


def load_result(
    run_or_dir: RunRecord | str | Path,
    *,
    lazy_soma_vs: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Load the standard saved outputs for a notebook run directory."""
    result_dir = Path(run_or_dir.result_dir if isinstance(run_or_dir, RunRecord) else run_or_dir)
    soma_path = find_soma_trace_artifact(result_dir)
    view_plan = _neuroinfra_plan_result_view(
        result_dir,
        result_factory_fn=_make_result_view,
        artifact_bindings=[
            _NeuroinfraResultArtifactBinding("input_times", result_dir / "input_times.pkl"),
            _NeuroinfraResultArtifactBinding(
                "soma_vs",
                soma_path,
                deferred_remote_name=preferred_soma_trace_artifact_name(),
                deferred_remote_names=soma_trace_artifact_candidates(),
            ),
            _NeuroinfraResultArtifactBinding("gc_output_events", result_dir / "gc_output_events.pkl"),
            _NeuroinfraResultArtifactBinding("lfp", result_dir / "lfp.pkl"),
            _NeuroinfraResultArtifactBinding("soma_spikes", result_dir / SOMA_SPIKES_FILENAME_NPZ),
            _NeuroinfraResultArtifactBinding("voltage_summary", result_dir / VOLTAGE_SUMMARY_FILENAME_NPZ),
        ],
        lazy_keys={"soma_vs"} if lazy_soma_vs else set(),
        hooks=_result_view_hooks(),
    )
    result = view_plan.result

    load_timings, load_total_seconds = _neuroinfra_load_local_artifact_plan(
        result,
        view_plan.load_plan,
        hooks=_artifact_loading_hooks(),
        progress=progress,
    )

    if lazy_soma_vs:
        _neuroinfra_attach_lazy_artifact_loaders(
            view_plan,
            hooks=_result_view_hooks(),
            progress=progress,
        )

    result["load_timing_seconds"] = load_timings
    result["load_total_seconds"] = load_total_seconds
    if load_timings and progress:
        timing_summary = ", ".join(
            f"{name}={seconds:.2f}s"
            for name, seconds in sorted(load_timings.items(), key=lambda item: item[1], reverse=True)
        )
        _progress_write(f"[OBGPU load] Local file timings: {timing_summary}")

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


def run_and_load(
    config: dict[str, Any] | None = None,
    *,
    label: str | None = None,
) -> tuple[RunRecord, dict[str, Any]]:
    """Run a simulation and immediately load its outputs from disk."""
    print("[OBGPU load] Starting simulation run...", flush=True)
    run = run_simulation(config, label=label)
    print(f"[OBGPU load] Simulation complete. Loading results from {run.result_dir}...", flush=True)
    result = load_result(run)
    _merge_run_info_payload(
        run.result_dir,
        {
            "artifact_sizes": result.get("artifact_sizes", {}),
            "load_timing_seconds": result.get("load_timing_seconds", {}),
            "load_total_seconds": result.get("load_total_seconds"),
        },
    )
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
    cell_type = match.group(1)
    return CELL_TYPE_ALIASES.get(cell_type, cell_type)


def _ordered_cell_types(cell_types: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    """Return cell types in stable notebook display order."""
    seen = {str(cell_type) for cell_type in cell_types}
    ordered = [cell_type for cell_type in PRIMARY_CELL_TYPE_ORDER if cell_type in seen]
    ordered.extend(sorted(cell_type for cell_type in seen if cell_type not in set(PRIMARY_CELL_TYPE_ORDER) and cell_type != "other"))
    if "other" in seen:
        ordered.append("other")
    return ordered


_OBGPU_CATEGORY_CATALOG_HOOKS = _NeuroinfraCategoryCatalogHooks(
    categorize_label_fn=lambda label: cell_type_of(label),
    order_categories_fn=lambda cell_types: _ordered_cell_types(set(cell_types)),
    unknown_category="other",
)


def _display_group_for_cell_type(cell_type: str, *, combine_mt: bool = True) -> str:
    """Map notebook cell-family labels to a small display bucket."""
    cell_type = str(cell_type)
    if combine_mt and cell_type in {"MC", "TC"}:
        return "MT"
    return cell_type


def _ordered_display_groups(groups: list[str] | tuple[str, ...] | set[str], *, combine_mt: bool = True) -> list[str]:
    """Return display buckets in a stable order for plots."""
    raw_seen = {str(group) for group in groups}
    if combine_mt:
        seen = {_display_group_for_cell_type(group, combine_mt=True) for group in raw_seen}
    else:
        return _ordered_cell_types(raw_seen)
    if not seen:
        return []
    ordered = [group for group in PLOT_DISPLAY_CELL_GROUPS if group in seen]
    ordered.extend(sorted(g for g in seen if g not in set(PLOT_DISPLAY_CELL_GROUPS) and g != "other"))
    if "other" in seen:
        ordered.append("other")
    return ordered


def _truncate_display_rows_for_group(
    rows: list[tuple[str, Any]],
    max_rows: int,
    *,
    combine_mt: bool,
    display_group: str,
) -> list[tuple[str, Any]]:
    """Limit rows per display bucket with fair MT sampling when MC/TC are merged."""
    if not rows or max_rows <= 0:
        return []
    if not combine_mt or display_group != "MT":
        return rows[:max_rows]

    subgroups: dict[str, list[tuple[str, Any]]] = {}
    subgroup_order: list[str] = []
    for row in rows:
        label = str(row[0])
        try:
            subgroup = cell_type_of(label)
        except ValueError:
            subgroup = "other"
        if subgroup not in subgroups:
            subgroups[subgroup] = []
            subgroup_order.append(subgroup)
        subgroups[subgroup].append(row)

    if len(subgroup_order) <= 1:
        return rows[:max_rows]

    selected: list[tuple[str, Any]] = []
    indices = {subgroup: 0 for subgroup in subgroup_order}
    while len(selected) < max_rows:
        added = False
        for subgroup in subgroup_order:
            idx = indices[subgroup]
            bucket = subgroups[subgroup]
            if idx < len(bucket):
                selected.append(bucket[idx])
                indices[subgroup] = idx + 1
                added = True
                if len(selected) >= max_rows:
                    break
        if not added:
            break
    return selected


def _infer_grouped_cell_types_from_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    """Infer ordered cell-family buckets from one list of saved labels."""
    inferred = []
    for label in labels:
        try:
            inferred.append(cell_type_of(label))
        except ValueError:
            inferred.append("other")
    return _ordered_cell_types(inferred)


def _cell_color(cell_type: str) -> str:
    """Return a stable plotting color for one cell family."""
    return CELL_TYPE_COLORS.get(str(cell_type), "tab:purple")


def get_slice_dir(slice_name: str = "DorsalColumnSlice") -> Path:
    """Return the on-disk directory for a named slice export."""
    return REPO_ROOT / "olfactorybulb" / "slices" / str(slice_name)


def load_slice_connectivity(slice_name: str = "DorsalColumnSlice") -> dict[str, Any]:
    """Load static group, glomerular, and synapse-set JSON for a slice."""
    slice_dir = get_slice_dir(slice_name)
    with open(slice_dir / "glom_cells.json") as f:
        glom_cells = json.load(f)

    cell_groups = {}
    for path in sorted(slice_dir.glob("*.json")):
        if path.name in {"glom_cells.json"} or "__" in path.stem:
            continue
        with open(path) as f:
            payload = json.load(f)
        cell_groups[path.stem] = payload

    synapse_sets = {}
    for path in sorted(slice_dir.glob("*__*.json")):
        with open(path) as f:
            synapse_sets[path.stem] = json.load(f)["entries"]

    return {
        "slice_name": slice_name,
        "slice_dir": slice_dir,
        "glom_cells": glom_cells,
        "cell_groups": cell_groups,
        "synapse_sets": synapse_sets,
    }


def summarize_slice_connectivity(slice_name: str = "DorsalColumnSlice") -> dict[str, Any]:
    """Return compact root-count and synapse-count summaries for one slice export."""
    connectivity = load_slice_connectivity(slice_name=slice_name)
    cell_groups = connectivity["cell_groups"]
    synapse_sets = connectivity["synapse_sets"]
    return {
        "slice_name": slice_name,
        "slice_dir": connectivity["slice_dir"],
        "cell_group_counts": {
            group_name: len(group_payload.get("roots", []))
            for group_name, group_payload in sorted(cell_groups.items())
        },
        "synapse_set_counts": {
            set_name: len(entries)
            for set_name, entries in sorted(synapse_sets.items())
        },
    }


def find_cell_drivers(cell_name: str, slice_name: str = "DorsalColumnSlice") -> dict[str, Any]:
    """Summarize glomerular peers and static synaptic connectivity for one cell."""
    target = normalize_cell_name(cell_name)
    target_type = cell_type_of(target)
    connectivity = load_slice_connectivity(slice_name=slice_name)
    glom_cells = connectivity["glom_cells"]
    synapse_sets = connectivity["synapse_sets"]

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
        entries = synapse_sets.get(reciprocal_set, [])
        reciprocal_inputs = [
            row for row in entries if normalize_cell_name(row["dest_section"]) == target
        ]
        source_counts = Counter(normalize_cell_name(row["source_section"]) for row in reciprocal_inputs)
        dest_section_counts = Counter(row["dest_section"].split(".", 1)[1] for row in reciprocal_inputs)

    incoming_by_set = {}
    outgoing_by_set = {}
    for set_name, entries in synapse_sets.items():
        incoming = [row for row in entries if normalize_cell_name(row["dest_section"]) == target]
        outgoing = [row for row in entries if normalize_cell_name(row["source_section"]) == target]
        if incoming:
            incoming_by_set[set_name] = incoming
        if outgoing:
            outgoing_by_set[set_name] = outgoing

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
        "incoming_synapses_by_set": incoming_by_set,
        "outgoing_synapses_by_set": outgoing_by_set,
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
    context = _neuroinfra_build_result_overview_context(result)
    soma_meta = _neuroinfra_first_result_file_metadata(context, soma_trace_artifact_candidates())
    input_meta = context.files.get("input_times.pkl") if isinstance(context.files.get("input_times.pkl"), dict) else {}
    lfp_meta = context.files.get("lfp.pkl") if isinstance(context.files.get("lfp.pkl"), dict) else {}
    return _neuroinfra_build_result_overview(
        context,
        extra_fields={
            "n_inputs": _neuroinfra_metadata_value_or_result_length(
                context,
                metadata=input_meta,
                metadata_key="items",
                result_key="input_times",
            ),
            "n_soma_traces": _neuroinfra_metadata_value_or_result_length(
                context,
                metadata=soma_meta,
                metadata_key="items",
                result_key="soma_vs",
            ),
            "n_gc_output_connections": len(result.get("gc_output_events", [])),
            "n_lfp_samples": _neuroinfra_metadata_value_or_result_length(
                context,
                metadata=lfp_meta,
                metadata_key="len_1",
                result_key="lfp",
            ),
        },
    )

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
    saved_rows = _saved_soma_spike_rows(
        result,
        cell_types=list(allowed_types),
        threshold=threshold,
    )
    if saved_rows is None:
        saved_rows = []
        for label, t, v in result["soma_vs"]:
            if not label.startswith(allowed_types):
                continue
            saved_rows.append((str(label), detect_spikes(t, v, threshold=threshold)))

    for label, spikes in saved_rows:
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


def _adaptive_spike_peak_floor(v: np.ndarray) -> float:
    """Estimate a conservative spike-height floor from one voltage trace."""
    return adaptive_soma_spike_peak_floor(v)


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
    return detect_soma_spikes(
        t,
        v,
        threshold=threshold,
        min_prominence_mv=min_prominence_mv,
        refractory_ms=refractory_ms,
    )


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
    return _neuroinfra_calculate_event_frequency(times)


def _saved_soma_spikes_match_threshold(result: dict[str, Any], threshold: float | None) -> bool:
    """Return whether saved soma spikes can satisfy one requested threshold."""
    if threshold is None:
        return True
    metadata = (dict.get(result, "soma_spikes") or {}).get("metadata", {})
    saved_threshold = metadata.get("threshold_mv")
    return saved_threshold is not None and np.isclose(float(saved_threshold), float(threshold))


def _saved_soma_spike_rows(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = None,
    threshold: float | None = None,
) -> list[tuple[str, np.ndarray]] | None:
    """Return saved ``(label, spike_times)`` rows, or None when raw traces are required."""
    soma_spikes = dict.get(result, "soma_spikes") or {}
    labels = soma_spikes.get("labels")
    spike_times = soma_spikes.get("spike_times")
    if not labels or spike_times is None:
        return None
    if not _saved_soma_spikes_match_threshold(result, threshold):
        return None

    prefixes = tuple(str(name) for name in cell_types) if cell_types else None
    if indices is None:
        indices = range(len(labels))

    rows = []
    for index in indices:
        if index >= len(labels):
            break
        label = str(labels[index])
        if prefixes is not None and not any(label.startswith(prefix) for prefix in prefixes):
            continue
        rows.append((label, np.asarray(spike_times[index], dtype=float)))
    return rows


def _saved_soma_spike_rows_by_type(
    result: dict[str, Any],
    *,
    max_cells_per_type: int,
    threshold: float | None = None,
    combine_mt: bool = True,
) -> list[tuple[str, np.ndarray]] | None:
    """Return saved spike rows grouped in stable family display order for rasters."""
    rows = _saved_soma_spike_rows(result, threshold=threshold)
    if rows is None:
        return None

    grouped: dict[str, list[tuple[str, np.ndarray]]] = {}
    for label, spikes in rows:
        try:
            bucket = cell_type_of(label)
        except ValueError:
            bucket = "other"
        bucket = _display_group_for_cell_type(bucket, combine_mt=combine_mt)
        grouped.setdefault(bucket, []).append((label, spikes))

    ordered = []
    for cell_type in _ordered_display_groups(grouped.keys(), combine_mt=combine_mt):
        ordered.extend(
            _truncate_display_rows_for_group(
                grouped.get(cell_type, []),
                max_cells_per_type,
                combine_mt=combine_mt,
                display_group=cell_type,
            )
        )
    return ordered


def _saved_voltage_summary_signal(
    result: dict[str, Any],
    *,
    cell_type: str,
    moment: str = "mean",
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one saved voltage-summary moment, or None when raw traces are required."""
    voltage_summary = dict.get(result, "voltage_summary") or {}
    t_by_type = voltage_summary.get("t_by_type") or {}
    values_by_type = voltage_summary.get(f"{moment}_by_type") or {}
    if cell_type not in t_by_type or cell_type not in values_by_type:
        return None
    return uniform_trace(t_by_type[cell_type], values_by_type[cell_type], dt_ms=dt_ms)


def _coerce_frequency_plot_config(
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    **overrides: Any,
) -> FrequencyPlotConfig:
    """Normalize a frequency-plot config input into a dataclass instance."""
    if config is None:
        base = FrequencyPlotConfig()
    elif isinstance(config, FrequencyPlotConfig):
        base = FrequencyPlotConfig(**vars(config))
    elif isinstance(config, dict):
        base = FrequencyPlotConfig(**config)
    else:
        raise TypeError(f"Unsupported frequency-plot config type {type(config)!r}")

    for key, value in overrides.items():
        if value is not None:
            setattr(base, key, value)
    return base


def frequency_plot_config_with_modulus(
    config: FrequencyPlotConfig | dict[str, Any] | None,
    modulus: float | int | None,
) -> FrequencyPlotConfig:
    """Copy one frequency plot config while replacing its time modulus."""
    copied = replace(_coerce_frequency_plot_config(config))
    copied.modulus = _normalize_time_modulus(modulus)
    return copied


def _apply_frequency_kde_y_scale(kde: Any, scale_y: float) -> None:
    """Rescale a 1D KDE in-place along its frequency axis."""
    if float(scale_y) == 1.0:
        return
    kde.covariance *= float(scale_y) ** 2
    kde.cho_cov = np.linalg.cholesky(kde.covariance)
    kde.log_det = 2 * np.log(np.diag(kde.cho_cov * np.sqrt(2 * np.pi))).sum()


def _apply_frequency_kde_xy_scale(kernel: Any, scale_x: float, scale_y: float) -> None:
    """Rescale a 2D time/frequency KDE in-place."""
    if float(scale_x) == 1.0 and float(scale_y) == 1.0:
        return
    kernel.covariance[0, 0] *= float(scale_x) ** 2
    kernel.covariance[1, 1] *= float(scale_y) ** 2
    kernel.covariance[0, 1] *= float(scale_x) * float(scale_y)
    kernel.covariance[1, 0] *= float(scale_x) * float(scale_y)
    kernel.cho_cov = np.linalg.cholesky(kernel.covariance)
    kernel.log_det = 2 * np.log(np.diag(kernel.cho_cov * np.sqrt(2 * np.pi))).sum()


def collect_spike_frequency_samples(
    result: dict[str, Any],
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = ("TC", "MC"),
    modulus: float | None = 1e8,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Collect midpoint/frequency samples from detected soma spikes."""
    prefixes = tuple(str(name) for name in cell_types) if cell_types else None
    all_freq_t = []
    all_freq = []
    labels = []

    saved_rows = _saved_soma_spike_rows(
        result,
        indices=indices,
        cell_types=cell_types,
        threshold=threshold,
    )
    if saved_rows is not None:
        trace_rows = [(label, spike_times) for label, spike_times in saved_rows]
    else:
        soma_vs = list(result.get("soma_vs", []))
        if indices is None:
            indices = range(len(soma_vs))
        trace_rows = []
        for i in indices:
            if i >= len(soma_vs):
                break
            label, t, mp = soma_vs[i]
            if prefixes is not None and not any(label.startswith(prefix) for prefix in prefixes):
                continue
            trace_rows.append((str(label), detect_spikes(t, mp, threshold=threshold)))

    for label, spike_times in trace_rows:
        t_freq, spiking_hz = calculate_event_frequency(spike_times)
        if len(t_freq) == 0:
            continue
        t_freq = np.asarray(t_freq, dtype=float)
        if modulus is not None:
            t_freq = np.mod(t_freq, float(modulus))
        all_freq_t.append(t_freq)
        all_freq.append(np.asarray(spiking_hz, dtype=float))
        labels.append(str(label))

    if all_freq_t:
        times = np.concatenate(all_freq_t)
        freqs = np.concatenate(all_freq)
    else:
        times = np.array([], dtype=float)
        freqs = np.array([], dtype=float)

    return {
        "times": times,
        "freqs": freqs,
        "labels": labels,
        "n_traces": len(labels),
        "cell_types": list(prefixes) if prefixes is not None else None,
    }


def _plot_frequency_kde_1d_from_samples(
    freqs: np.ndarray,
    *,
    config: FrequencyPlotConfig,
    title: str,
    ax: Any = None,
) -> Any:
    """Plot a 1D KDE from frequency samples."""
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[np.isfinite(freqs)]
    if len(freqs) == 0:
        ax.text(0.5, 0.5, "No frequency samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, float(config.max_freq_hz))
        return ax

    f_upper = max(float(config.max_freq_hz), float(np.max(freqs)) * 1.1)
    engine = str(getattr(config, "kde1d_engine", "histogram")).strip().lower()
    if engine in {"exact", "gaussian", "gaussian_kde", "scipy"}:
        kde = gaussian_kde(freqs, bw_method=config.kde_bw_method)
        _apply_frequency_kde_y_scale(kde, config.kde_bw_y)
        f_range = np.linspace(0.0, f_upper, int(config.kde_f_resolution))
        density = kde(f_range)
    else:
        bins = max(16, int(config.kde_f_resolution))
        clipped = freqs[(freqs >= 0.0) & (freqs <= f_upper)]
        if len(clipped) == 0:
            clipped = freqs
        density, edges = np.histogram(clipped, bins=bins, range=(0.0, f_upper), density=True)
        sigma = max(0.0, float(config.kde_bw_y) * 8.0)
        density = gaussian_filter1d(density, sigma=sigma, mode="nearest")
        f_range = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(f_range, density)
    ax.fill_between(f_range, density, alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim(0, float(config.max_freq_hz))
    return ax


def _plot_frequency_kde_2d_from_samples(
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    config: FrequencyPlotConfig,
    title: str,
    ax: Any = None,
) -> Any:
    """Plot a 2D time/frequency KDE from samples."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    finite = np.isfinite(times) & np.isfinite(freqs)
    times = times[finite]
    freqs = freqs[finite]
    if len(times) < 2 or len(freqs) < 2:
        ax.text(0.5, 0.5, "Not enough frequency samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, float(config.max_freq_hz))
        return ax

    tstop = float(np.max(times))
    max_freq_hz = float(config.max_freq_hz)
    engine = str(getattr(config, "kde2d_engine", "histogram")).strip().lower()
    if engine in {"exact", "gaussian", "gaussian_kde", "scipy"}:
        kernel = gaussian_kde(np.vstack([times, freqs]), bw_method=config.kde_bw_method)
        _apply_frequency_kde_xy_scale(kernel, config.kde_bw_x, config.kde_bw_y)
        t_grid = np.linspace(0.0, tstop, int(config.kde_resolution_t))
        f_grid = np.linspace(0.0, max_freq_hz, int(config.kde_resolution_f))
        t_mesh, f_mesh = np.meshgrid(t_grid, f_grid)
        positions = np.vstack([t_mesh.ravel(), f_mesh.ravel()])
        density = np.reshape(kernel(positions).T, t_mesh.shape)
    else:
        # Exact gaussian_kde is O(samples * grid) per frame and is too slow for
        # large sweeps. A smoothed 2D histogram preserves the sweep-level visual
        # signal while scaling linearly in sample count plus grid size.
        mask = (times >= 0.0) & (freqs >= 0.0) & (freqs <= max_freq_hz)
        times = times[mask]
        freqs = freqs[mask]
        density, _t_edges, _f_edges = np.histogram2d(
            times,
            freqs,
            bins=(int(config.kde_resolution_t), int(config.kde_resolution_f)),
            range=((0.0, tstop), (0.0, max_freq_hz)),
        )
        sigma_t = max(0.0, float(config.kde_bw_x) * 6.0)
        sigma_f = max(0.0, float(config.kde_bw_y) * 6.0)
        density = gaussian_filter(density.T, sigma=(sigma_f, sigma_t), mode="nearest")

    im = ax.imshow(
        density,
        origin="lower",
        extent=[0, tstop, 0, max_freq_hz],
        aspect="auto",
        cmap=config.kde_cmap,
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="Density (KDE)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return ax


def _plot_frequency_time_binned_from_samples(
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    config: FrequencyPlotConfig,
    title: str,
    ax: Any = None,
    show_dots: bool = True,
    show_ridgeline_kde: bool = False,
) -> Any:
    """Plot time-binned frequency distributions from midpoint/frequency samples."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if len(times) == 0 or len(freqs) == 0:
        ax.text(0.5, 0.5, "No frequency samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, float(config.max_freq_hz))
        return ax

    tstop = float(np.max(times))
    t_bins = np.linspace(0.0, tstop, int(config.num_time_bins) + 1)
    if len(t_bins) < 2:
        t_bins = np.array([0.0, max(tstop, 1.0)], dtype=float)
    bin_width = float(t_bins[1] - t_bins[0])

    for i in range(len(t_bins) - 1):
        t_start, t_end = float(t_bins[i]), float(t_bins[i + 1])
        mask = (times >= t_start) & (times < t_end)
        if not np.any(mask):
            continue
        bin_f = freqs[mask]

        if show_dots:
            if bool(config.strip_plot):
                x_pos = np.full_like(bin_f, t_start + bin_width / 2.0)
            else:
                jitter = np.random.uniform(0.0, bin_width * 0.8, size=len(bin_f))
                x_pos = t_start + jitter
            ax.scatter(
                x_pos,
                bin_f,
                s=float(config.dot_size),
                alpha=float(config.dot_alpha),
                color="black",
                edgecolors="none",
            )

        if show_ridgeline_kde and len(bin_f) > 2:
            kde = gaussian_kde(bin_f, bw_method=config.kde_bw_method)
            _apply_frequency_kde_y_scale(kde, config.kde_bw_y)
            f_range = np.linspace(0.0, max(float(np.max(freqs)) * 1.1, float(config.max_freq_hz)), int(config.kde_f_resolution))
            density = kde(f_range)
            if float(np.max(density)) > 0:
                density = density / float(np.max(density))
            ax.fill_betweenx(
                f_range,
                t_start,
                t_start + density * bin_width * 0.8,
                alpha=float(config.bin_alpha),
            )
            ax.plot(
                t_start + density * bin_width * 0.8,
                f_range,
                linewidth=1.0,
                color="black",
                alpha=0.3,
            )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim(0, float(config.max_freq_hz))
    ax.grid(True, alpha=0.3)
    return ax


def split_traces_by_type(result: dict[str, Any]) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Group saved soma traces by cell family prefix."""
    return _neuroinfra_group_rows_by_category(
        list(result["soma_vs"]),
        label_fn=lambda row: row[0],
        transform_row_fn=lambda row: (
            row[0],
            np.asarray(row[1], dtype=float),
            np.asarray(row[2], dtype=float),
        ),
        hooks=_OBGPU_CATEGORY_CATALOG_HOOKS,
    )


def list_available_cell_types(result: dict[str, Any]) -> list[str]:
    """List saved cell families available for analysis in stable display order."""
    voltage_summary = dict.get(result, "voltage_summary") or {}
    soma_spikes = dict.get(result, "soma_spikes") or {}
    return _neuroinfra_list_available_categories(
        label_sources=(
            (label for label, _t, _v in result.get("soma_vs", [])),
            (str(cell_type) for cell_type in voltage_summary.get("cell_types", []) or []),
            (label for label in soma_spikes.get("labels", []) or []),
        ),
        hooks=_OBGPU_CATEGORY_CATALOG_HOOKS,
    )


def list_available_soma_labels(result: dict[str, Any]) -> list[str]:
    """List saved soma labels from raw traces or compact spike artifacts."""
    soma_spikes = dict.get(result, "soma_spikes") or {}
    return _neuroinfra_list_unique_labels(
        (label for label, _t, _v in result.get("soma_vs", [])),
        (label for label in soma_spikes.get("labels", []) or []),
    )


def _lfp_signal_provider() -> _NeuroinfraResultSignalProvider:
    """Provide the standard LFP named signal when it is present."""

    def _list_names(result: dict[str, Any], _context: dict[str, Any]) -> list[str]:
        lfp_t = result.get("lfp_t")
        lfp = result.get("lfp")
        if lfp_t is None or lfp is None:
            return []
        try:
            if len(lfp_t) > 0 and len(lfp) > 0:
                return ["lfp"]
        except TypeError:
            return []
        return []

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if signal != "lfp":
            raise KeyError(signal)
        return uniform_trace(result["lfp_t"], result["lfp"], dt_ms=context.get("dt_ms"))

    return _NeuroinfraResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda signal: signal == "lfp",
        resolve_fn=_resolve,
    )


def _gc_output_rate_signal_provider() -> _NeuroinfraResultSignalProvider:
    """Provide named GC inhibitory-output rate signals."""

    def _list_names(result: dict[str, Any], _context: dict[str, Any]) -> list[str]:
        events = result.get("gc_output_events") or []
        return ["gc_output_rate", "gc_output_rate_MC", "gc_output_rate_TC"] if len(events) > 0 else []

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if signal not in {"gc_output_rate", "gc_output_rate_MC", "gc_output_rate_TC"}:
            raise KeyError(signal)
        target_types = None
        if signal.endswith("_MC"):
            target_types = ["MC"]
        elif signal.endswith("_TC"):
            target_types = ["TC"]
        bin_ms = 5.0 if context.get("dt_ms") is None else float(context["dt_ms"])
        return compute_gc_output_rate(
            result,
            bin_ms=bin_ms,
            smooth_sigma_ms=max(2.0 * bin_ms, 5.0),
            target_types=target_types,
            normalization="per_target_cell",
        )

    return _NeuroinfraResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda signal: signal in {"gc_output_rate", "gc_output_rate_MC", "gc_output_rate_TC"},
        resolve_fn=_resolve,
    )


def _input_rate_signal_provider() -> _NeuroinfraResultSignalProvider:
    """Provide named odor-input rate signals."""

    def _list_names(result: dict[str, Any], _context: dict[str, Any]) -> list[str]:
        input_times = result.get("input_times") or []
        return ["input_rate", "input_rate_MC", "input_rate_TC"] if len(input_times) > 0 else []

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if signal not in {"input_rate", "input_rate_MC", "input_rate_TC"}:
            raise KeyError(signal)
        target_types = None
        if signal.endswith("_MC"):
            target_types = ["MC"]
        elif signal.endswith("_TC"):
            target_types = ["TC"]
        bin_ms = 5.0 if context.get("dt_ms") is None else float(context["dt_ms"])
        return compute_input_rate(
            result,
            bin_ms=bin_ms,
            smooth_sigma_ms=max(2.0 * bin_ms, 5.0),
            target_types=target_types,
            normalization="per_target_cell",
        )

    return _NeuroinfraResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda signal: signal in {"input_rate", "input_rate_MC", "input_rate_TC"},
        resolve_fn=_resolve,
    )


def _mean_voltage_signal_provider() -> _NeuroinfraResultSignalProvider:
    """Provide dynamic per-cell-type mean-voltage signals."""

    def _list_names(result: dict[str, Any], _context: dict[str, Any]) -> list[str]:
        return [f"mean_{cell_type}_voltage" for cell_type in list_available_cell_types(result)]

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        mean_voltage_match = re.fullmatch(r"mean_([A-Z]+)_voltage", signal)
        if mean_voltage_match is None:
            raise KeyError(signal)
        cell_type = mean_voltage_match.group(1)
        saved_signal = _saved_voltage_summary_signal(
            result,
            cell_type=cell_type,
            moment="mean",
            dt_ms=context.get("dt_ms"),
        )
        if saved_signal is not None:
            return saved_signal
        grouped = split_traces_by_type(result)
        traces = grouped.get(cell_type, [])
        if not traces:
            raise KeyError(f"No soma traces found for {cell_type}")
        first_t, _first_v = uniform_trace(traces[0][1], traces[0][2], dt_ms=context.get("dt_ms"))
        aligned = []
        for _label, t, v in traces:
            interp_t, interp_v = uniform_trace(
                t,
                v,
                dt_ms=float(np.median(np.diff(first_t))) if len(first_t) > 1 else context.get("dt_ms"),
            )
            n = min(len(first_t), len(interp_t))
            aligned.append(interp_v[:n])
        n = min(len(values) for values in aligned)
        return first_t[:n], np.mean(np.vstack([values[:n] for values in aligned]), axis=0)

    return _NeuroinfraResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda signal: re.fullmatch(r"mean_([A-Z]+)_voltage", signal) is not None,
        resolve_fn=_resolve,
    )


def _soma_label_signal_provider() -> _NeuroinfraResultSignalProvider:
    """Provide direct per-soma trace signals by saved label."""

    def _list_names(result: dict[str, Any], context: dict[str, Any]) -> list[str]:
        if not bool(context.get("include_soma_labels", False)):
            return []
        return list_available_soma_labels(result)

    def _resolve(result: dict[str, Any], signal: str, context: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        for label, t, v in result["soma_vs"]:
            if label == signal:
                return uniform_trace(t, v, dt_ms=context.get("dt_ms"))
        raise KeyError(signal)

    return _NeuroinfraResultSignalProvider(
        list_names_fn=_list_names,
        matches_fn=lambda _signal: True,
        resolve_fn=_resolve,
    )


_OBGPU_RESULT_SIGNAL_PROVIDERS: tuple[_NeuroinfraResultSignalProvider, ...] = (
    _lfp_signal_provider(),
    _gc_output_rate_signal_provider(),
    _input_rate_signal_provider(),
    _mean_voltage_signal_provider(),
    _soma_label_signal_provider(),
)


def list_available_named_signals(
    result: dict[str, Any],
    *,
    include_soma_labels: bool = False,
) -> list[str]:
    """List named analysis signals currently resolvable for one loaded result."""
    return _neuroinfra_list_available_result_signals(
        result,
        _OBGPU_RESULT_SIGNAL_PROVIDERS,
        include_soma_labels=include_soma_labels,
    )


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


def plot_spike_frequency_kde_1d(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = ("TC", "MC"),
    threshold: float | None = None,
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a 1D KDE of detected soma spike frequencies."""
    plot_config = _coerce_frequency_plot_config(config)
    data = collect_spike_frequency_samples(
        result,
        indices=indices,
        cell_types=cell_types,
        modulus=plot_config.modulus,
        threshold=threshold,
    )
    label = "all" if not cell_types else "+".join(str(name) for name in cell_types)
    return _plot_frequency_kde_1d_from_samples(
        data["freqs"],
        config=plot_config,
        title=title or f"Soma Spike Frequency Distribution ({label})",
        ax=ax,
    )


def plot_spike_frequency_kde_2d(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = ("TC", "MC"),
    threshold: float | None = None,
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a 2D time/frequency KDE of detected soma spike frequencies."""
    plot_config = _coerce_frequency_plot_config(config)
    data = collect_spike_frequency_samples(
        result,
        indices=indices,
        cell_types=cell_types,
        modulus=plot_config.modulus,
        threshold=threshold,
    )
    label = "all" if not cell_types else "+".join(str(name) for name in cell_types)
    return _plot_frequency_kde_2d_from_samples(
        data["times"],
        data["freqs"],
        config=plot_config,
        title=title or f"Soma Spike Time/Frequency KDE ({label})",
        ax=ax,
    )


def plot_spike_frequency_time_binned(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    cell_types: list[str] | tuple[str, ...] | None = ("TC", "MC"),
    threshold: float | None = None,
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
    show_dots: bool = True,
    show_ridgeline_kde: bool = False,
) -> Any:
    """Plot time-binned soma spike-frequency distributions."""
    plot_config = _coerce_frequency_plot_config(config)
    data = collect_spike_frequency_samples(
        result,
        indices=indices,
        cell_types=cell_types,
        modulus=plot_config.modulus,
        threshold=threshold,
    )
    label = "all" if not cell_types else "+".join(str(name) for name in cell_types)
    return _plot_frequency_time_binned_from_samples(
        data["times"],
        data["freqs"],
        config=plot_config,
        title=title or f"Soma Spike Frequency Distributions ({label})",
        ax=ax,
        show_dots=show_dots,
        show_ridgeline_kde=show_ridgeline_kde,
    )


def plot_gc_output_frequency_kde_1d(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    target_types: list[str] | tuple[str, ...] | None = ("MC", "TC"),
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a 1D KDE of reciprocal GC inhibitory-output frequencies."""
    plot_config = _coerce_frequency_plot_config(config)
    data = collect_gc_output_frequency_samples(
        result,
        indices=indices,
        target_types=target_types,
        modulus=plot_config.modulus,
    )
    label = "all" if not target_types else "_".join(str(name) for name in target_types)
    return _plot_frequency_kde_1d_from_samples(
        data["freqs"],
        config=plot_config,
        title=title or f"GC Inhibitory Output Frequency Distribution ({label})",
        ax=ax,
    )


def plot_gc_output_frequency_kde_2d(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    target_types: list[str] | tuple[str, ...] | None = ("MC", "TC"),
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a 2D time/frequency KDE of reciprocal GC inhibitory-output frequencies."""
    plot_config = _coerce_frequency_plot_config(config)
    data = collect_gc_output_frequency_samples(
        result,
        indices=indices,
        target_types=target_types,
        modulus=plot_config.modulus,
    )
    label = "all" if not target_types else "_".join(str(name) for name in target_types)
    return _plot_frequency_kde_2d_from_samples(
        data["times"],
        data["freqs"],
        config=plot_config,
        title=title or f"GC Inhibitory Output Time/Frequency KDE ({label})",
        ax=ax,
    )


def plot_gc_output_frequency_time_binned(
    result: dict[str, Any],
    *,
    indices: list[int] | range | None = None,
    target_types: list[str] | tuple[str, ...] | None = ("MC", "TC"),
    config: FrequencyPlotConfig | dict[str, Any] | None = None,
    ax: Any = None,
    title: str | None = None,
    show_dots: bool = True,
    show_ridgeline_kde: bool = False,
) -> Any:
    """Plot time-binned reciprocal GC inhibitory-output frequency distributions."""
    plot_config = _coerce_frequency_plot_config(config)
    data = collect_gc_output_frequency_samples(
        result,
        indices=indices,
        target_types=target_types,
        modulus=plot_config.modulus,
    )
    label = "all" if not target_types else "_".join(str(name) for name in target_types)
    return _plot_frequency_time_binned_from_samples(
        data["times"],
        data["freqs"],
        config=plot_config,
        title=title or f"GC Inhibitory Output Frequency Distributions ({label})",
        ax=ax,
        show_dots=show_dots,
        show_ridgeline_kde=show_ridgeline_kde,
    )


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
    return _neuroinfra_smooth_rate_series(
        rate_hz,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
    )


def _event_rate_from_series(
    event_series: list[np.ndarray],
    *,
    t_stop: float,
    bin_ms: float,
    smooth_sigma_ms: float,
    denominator: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin one or more event series into a smoothed population-rate trace."""
    return _neuroinfra_binned_event_rate(
        event_series,
        t_stop=t_stop,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        denominator=denominator,
    )


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
    return _neuroinfra_rate_series_label(base_label, metadata)


def get_named_signal(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one named analysis signal into a uniform time/value trace."""
    return _neuroinfra_resolve_result_signal(
        result,
        signal,
        _OBGPU_RESULT_SIGNAL_PROVIDERS,
        dt_ms=dt_ms,
    )


def _recommended_raster_fontsize(n_rows: int, *, default: float = 7.0) -> float:
    """Choose a compact but readable y-label font size for dense rasters."""
    return _neuroinfra_recommended_raster_fontsize(n_rows, default=default)


def _recommended_raster_height(n_rows: int, *, min_height: float = 4.0) -> float:
    """Estimate a reasonable figure height for a raster plot."""
    return _neuroinfra_recommended_raster_height(n_rows, min_height=min_height)


def _ensure_raster_axis(
    ax: Any,
    n_rows: int,
    *,
    width: float = 14.0,
    min_height: float = 4.0,
    per_row_height: float = 0.22,
) -> Any:
    """Create a raster axis sized to the current row count when needed."""
    return _neuroinfra_ensure_raster_axis(
        ax,
        n_rows,
        width=width,
        min_height=min_height,
        per_row_height=per_row_height,
    )


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
    return _neuroinfra_style_raster_axis(
        ax,
        labels,
        ylabel=ylabel,
        title=title,
        fontsize=fontsize,
        line_spacing=line_spacing,
    )


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
    return _neuroinfra_fit_raster_labels(
        ax,
        offsets,
        min_fontsize=min_fontsize,
        target_ratio=target_ratio,
        min_height=min_height,
        max_iter=max_iter,
    )


def plot_input_raster(
    result: dict[str, Any],
    ax: Any = None,
    max_segments: int = 80,
    target_types: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Plot the saved odor-input event raster."""
    rows = sorted(filter_input_events(result, target_types=target_types), key=lambda row: row[0])[:max_segments]
    display_rows = [(row[0].replace("h.", ""), row[1]) for row in rows]
    return _neuroinfra_plot_event_raster_rows(
        display_rows,
        ax=ax,
        ylabel="Input Segment",
        title="Odor Input Raster",
        width=14.0,
        min_height=4.0,
        per_row_height=0.10,
        fontsize=_recommended_raster_fontsize(len(rows)),
        line_spacing=1.4,
        colors="black",
        no_data_message="No input events saved",
    )


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


def plot_voltage_traces(
    result: dict[str, Any],
    max_per_type: int = 4,
    ax: Any = None,
    *,
    combine_mt: bool = True,
) -> Any:
    """Plot a small representative subset of saved soma voltages."""
    ax = ax or plt.subplots(figsize=(14, 8))[1]
    grouped = split_traces_by_type(result)
    ordered_cell_types = _ordered_display_groups(
        [
            _display_group_for_cell_type(cell_type, combine_mt=combine_mt)
            for cell_type in _ordered_cell_types(grouped.keys())
        ],
        combine_mt=combine_mt,
    )
    buckets = {cell_type: [] for cell_type in ordered_cell_types}
    for label, t, v in result["soma_vs"]:
        try:
            group = _display_group_for_cell_type(cell_type_of(label), combine_mt=combine_mt)
        except ValueError:
            group = "other"
        if group in buckets:
            buckets[group].append((label, t, v))
    offset = 0.0
    for display_group in ordered_cell_types:
        traces = _truncate_display_rows_for_group(
            buckets.get(display_group, []),
            max_per_type,
            combine_mt=combine_mt,
            display_group=display_group,
        )
        if display_group == "MT":
            # Keep MC/TC color/identity while grouping them with the same banner.
            for label, t, v in traces:
                original_type = cell_type_of(label)
                ax.plot(t, v + offset, color=_cell_color(original_type), linewidth=1.0, label=label)
                offset += 120.0
            continue

        for label, t, v in traces:
            try:
                color_key = cell_type_of(label)
            except ValueError:
                color_key = "other"
            ax.plot(t, v + offset, color=_cell_color(color_key), linewidth=1.0, label=label)
            offset += 40.0 if display_group == "GC" else 120.0
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Offset Voltage")
    ax.set_title("Sample Soma Voltages" + (" (MT grouped)" if combine_mt else ""))
    if ax.lines:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    return ax


def plot_spike_raster(
    result: dict[str, Any],
    threshold: float | None = None,
    max_cells_per_type: int = 24,
    ax: Any = None,
    modulus: float | None = None,
    *,
    combine_mt: bool = True,
) -> Any:
    """Plot a soma-spike raster, preferring compact runtime spike artifacts."""
    saved_rows = _saved_soma_spike_rows_by_type(
        result,
        max_cells_per_type=max_cells_per_type,
        threshold=threshold,
        combine_mt=combine_mt,
    )
    if saved_rows is None:
        grouped = split_traces_by_type(result)
        raw_bucketed: dict[str, list[tuple[str, Any]]] = {}
        for cell_type in _ordered_cell_types(grouped.keys()):
            display_group = _display_group_for_cell_type(cell_type, combine_mt=combine_mt)
            raw_bucketed.setdefault(display_group, [])
            for trace in grouped[cell_type][:max_cells_per_type]:
                if isinstance(trace, tuple) and len(trace) == 3:
                    raw_bucketed[display_group].append(trace)
        grouped_rows = []
        for display_group in _ordered_display_groups(raw_bucketed.keys(), combine_mt=combine_mt):
            grouped_rows.extend(
                _truncate_display_rows_for_group(
                    raw_bucketed.get(display_group, []),
                    max_cells_per_type,
                    combine_mt=combine_mt,
                    display_group=display_group,
                )
            )

        raw_rows = []
        raw_rows.extend(grouped_rows)
        rows = [(label, detect_spikes(t, v, threshold=threshold)) for label, t, v in raw_rows]
    else:
        rows = saved_rows
    colors = [
        _cell_color(cell_type_of(label) if re.match(r"([A-Z]+)", normalize_cell_name(label)) else "other")
        for label, _spikes in rows
    ]
    return _neuroinfra_plot_event_raster_rows(
        rows,
        ax=ax,
        ylabel="Cell",
        title="Detected Soma Spike Raster" + (" (MT grouped)" if combine_mt else ""),
        width=14.0,
        min_height=4.5,
        per_row_height=0.10,
        fontsize=_recommended_raster_fontsize(len(rows)),
        line_spacing=1.3,
        modulus=modulus,
        colors=colors,
        no_data_message="No soma spikes saved",
    )


def plot_gc_output_event_raster(
    result: dict[str, Any],
    max_connections: int = 120,
    target_types: list[str] | tuple[str, ...] | None = None,
    ax: Any = None,
    *,
    fontsize: float = 7,
    line_spacing: float = 1.4,
    modulus: float | None = None,
) -> Any:
    """Plot the saved reciprocal GC inhibitory-output event raster."""
    rows = filter_gc_output_events(result, target_types=target_types)[:max_connections]
    display_rows = [
        f"{normalize_cell_name(row.get('source_section', 'GC'))}->{normalize_cell_name(row.get('dest_section', 'cell'))}"
        for row in rows
    ]
    prepared_rows = [
        (label, np.asarray(row.get("times", []), dtype=float))
        for label, row in zip(display_rows, rows)
    ]
    return _neuroinfra_plot_event_raster_rows(
        prepared_rows,
        ax=ax,
        ylabel="Reciprocal GABA Connection",
        title="GC Inhibitory Output Events",
        width=16.0,
        min_height=4.5,
        per_row_height=0.10,
        fontsize=min(float(fontsize), _recommended_raster_fontsize(len(rows), default=float(fontsize))),
        line_spacing=line_spacing,
        modulus=modulus,
        colors="black",
        no_data_message="No GC inhibitory-output events saved",
    )


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
    highcut_hz: float = 300.0,
    psd_xlim_hz: tuple[float, float] | None = None,
    *,
    show_psd_target_template: bool = True,
    psd_template_kind: str = "ketamine",
    psd_template_fit_band_hz: tuple[float, float] = DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ,
    psd_template_scale_method: str = "area",
    psd_template_floor: float = DEFAULT_PSD_TEMPLATE_FLOOR,
    psd_template_color: str = "tab:orange",
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
    axes[2].plot(freqs, power, color="tab:green", linewidth=1.0, label="Measured PSD")

    if show_psd_target_template:
        try:
            from olfactorybulb.hfo_optimizer import scaled_psd_template_curve

            template_freqs, template_power = scaled_psd_template_curve(
                psd_template_kind,
                freqs,
                power,
                fit_band_hz=psd_template_fit_band_hz,
                method=psd_template_scale_method,
                floor=psd_template_floor,
            )
            axes[2].plot(
                template_freqs,
                template_power,
                color=psd_template_color,
                linewidth=1.0,
                linestyle="--",
                label=f"Template ({psd_template_kind})",
            )
            axes[2].legend(loc="upper right", fontsize=9)
        except Exception:
            pass

    if psd_xlim_hz is None:
        psd_xlim_hz = (0.0, float(highcut_hz))
    axes[2].set_xlim(float(psd_xlim_hz[0]), float(psd_xlim_hz[1]))
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
    fig, axes = _neuroinfra_plot_band_power_summary(summary, signal_label=signal)
    return fig, axes, summary


def plot_named_signal(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    ax: Any = None,
    modulus: float | None = None,
) -> Any:
    """Plot one named analysis signal as a time trace."""
    t, y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    return _neuroinfra_plot_time_series(
        t,
        y,
        ax=ax,
        modulus=modulus,
        dt_ms=dt_ms,
        title=f"{signal} Trace",
        ylabel=signal,
    )


def _spectrogram_display_power(power: np.ndarray) -> np.ndarray:
    """Normalize spectrogram power into a stable display range."""
    values = np.log(np.asarray(power, dtype=float) + 1e-8)
    values -= values.min()
    return values


def plot_spectrogram(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    max_freq_hz: float = 250.0,
    nperseg: int = 256,
    noverlap: int = 192,
    ax: Any = None,
    modulus: float | None = None,
) -> Any:
    """Plot a spectrogram for a named analysis signal."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    times_ms, freqs, power = compute_spectrogram(
        signal_t,
        signal_y,
        dt_ms=dt_ms,
        max_freq_hz=max_freq_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return _neuroinfra_plot_time_frequency_map(
        times_ms,
        freqs,
        power,
        ax=ax,
        modulus=modulus,
        title=f"{signal.upper()} Spectrogram",
        colorbar_label="Power (dB)",
        power_transform=_spectrogram_display_power,
    )


def plot_wavelet(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    ax: Any = None,
    modulus: float | None = None,
) -> Any:
    """Plot the continuous wavelet power map for a named signal."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
    return _neuroinfra_plot_time_frequency_map(
        t,
        freqs,
        power,
        ax=ax,
        modulus=modulus,
        title=f"{signal.upper()} Wavelet Power",
        colorbar_label="log(1 + |cwt|)",
    )


def plot_wavelet_band_power(
    result: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    bands: dict[str, tuple[float, float]] | None = None,
    ax: Any = None,
    modulus: float | None = None,
) -> Any:
    """Plot band-collapsed wavelet power traces over time."""
    signal_t, signal_y = get_named_signal(result, signal=signal, dt_ms=dt_ms)
    t, _freqs, _power, traces = compute_wavelet_band_power(signal_t, signal_y, bands=bands, dt_ms=dt_ms)
    return _neuroinfra_plot_named_time_series(
        t,
        traces,
        ax=ax,
        modulus=modulus,
        dt_ms=dt_ms,
        title="Band Power Over Time",
        ylabel="Mean Wavelet Power",
    )


def _extract_figure_from_plot_result(plot_result: Any) -> Any:
    """Best-effort extraction of a Matplotlib figure from a plot return value."""
    return _neuroinfra_extract_figure_from_plot_result(plot_result)


_DEPRECATED_SWEEP_ANIMATION_PLOTS = {
    "gc_output_frequency_overview",
    "gc_output_overview",
    "input_overview",
    "lfp_overview",
    "spike_frequency_overview",
}


def get_builtin_sweep_plot_names() -> list[str]:
    """Return built-in plot names that can be rendered across a sweep."""
    return sorted([
        "gc_output_frequency_kde_1d",
        "gc_output_frequency_kde_2d",
        "gc_output_frequency_time_binned",
        "hfo_power_summary",
        "named_signal",
        "spectrogram",
        "spike_frequency_kde_1d",
        "spike_frequency_kde_2d",
        "spike_frequency_time_binned",
        "spike_raster",
        "voltage_traces",
        "wavelet",
        "wavelet_band_power",
    ])


def _get_builtin_sweep_plot(plot_name: str) -> Any:
    """Resolve a built-in sweep-plot name to a plotting helper."""
    mapping = {
        "voltage_traces": plot_voltage_traces,
        "spike_raster": plot_spike_raster,
        "hfo_power_summary": plot_hfo_power_summary,
        "named_signal": plot_named_signal,
        "spectrogram": plot_spectrogram,
        "wavelet": plot_wavelet,
        "wavelet_band_power": plot_wavelet_band_power,
        "spike_frequency_kde_1d": plot_spike_frequency_kde_1d,
        "spike_frequency_kde_2d": plot_spike_frequency_kde_2d,
        "spike_frequency_time_binned": plot_spike_frequency_time_binned,
        "gc_output_frequency_kde_1d": plot_gc_output_frequency_kde_1d,
        "gc_output_frequency_kde_2d": plot_gc_output_frequency_kde_2d,
        "gc_output_frequency_time_binned": plot_gc_output_frequency_time_binned,
    }
    if plot_name not in mapping:
        if plot_name in _DEPRECATED_SWEEP_ANIMATION_PLOTS:
            raise KeyError(
                f"Sweep animation plot {plot_name!r} is deprecated and intentionally "
                "not rendered. Use the separate KDE/signal plots instead."
            )
        available = ", ".join(get_builtin_sweep_plot_names())
        raise KeyError(f"Unknown built-in sweep plot {plot_name!r}. Available: {available}")
    return mapping[plot_name]


def make_sweep_plot_spec(
    plot: str | Any,
    *,
    name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    fps: int = 10,
    title_fn: Any = None,
) -> SweepPlotSpec:
    """Build a sweep-plot spec from a built-in plot name or custom callable."""
    return _neuroinfra_make_sweep_plot_spec(
        plot,
        name=name,
        plot_kwargs=plot_kwargs,
        filename=filename,
        figsize=figsize,
        interval=interval,
        fps=fps,
        title_fn=title_fn,
    )


def make_sweep_time_variant_specs(
    plot: str | Any,
    *,
    modulus: float | int | None,
    name: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    fps: int = 10,
    title_fn: Any = None,
    include_full: bool = True,
    include_modulus: bool = True,
) -> list[SweepPlotSpec]:
    """Build full-time and modulo-time variants for one time-axis sweep plot."""
    base_name = name
    if base_name is None:
        base_name = plot if isinstance(plot, str) else getattr(plot, "__name__", "custom_plot")
    base_filename = filename or str(base_name)
    kwargs = dict(plot_kwargs or {})
    specs: list[SweepPlotSpec] = []

    if include_full:
        full_kwargs = dict(kwargs)
        full_kwargs.pop("modulus", None)
        if "config" in full_kwargs:
            full_kwargs["config"] = frequency_plot_config_with_modulus(full_kwargs["config"], None)
        specs.append(
            make_sweep_plot_spec(
                plot,
                name=f"{base_name}_full",
                filename=f"{base_filename}_full",
                plot_kwargs=full_kwargs,
                figsize=figsize,
                interval=interval,
                fps=fps,
                title_fn=title_fn,
            )
        )

    modulus_value = _normalize_time_modulus(modulus)
    if include_modulus and modulus_value is not None:
        suffix = _safe_name(f"mod{_short_artifact_setting_value(modulus_value)}")
        mod_kwargs = dict(kwargs)
        if "config" in mod_kwargs:
            mod_kwargs["config"] = frequency_plot_config_with_modulus(mod_kwargs["config"], modulus_value)
        else:
            mod_kwargs["modulus"] = modulus_value
        specs.append(
            make_sweep_plot_spec(
                plot,
                name=f"{base_name}_{suffix}",
                filename=f"{base_filename}_{suffix}",
                plot_kwargs=mod_kwargs,
                figsize=figsize,
                interval=interval,
                fps=fps,
                title_fn=title_fn,
            )
        )

    return specs


def _normalize_sweep_plot_spec(plot_spec: SweepPlotSpec | str | Any | dict[str, Any]) -> SweepPlotSpec:
    """Accept ergonomic plot-spec forms and normalize them."""
    return _neuroinfra_normalize_sweep_plot_spec(plot_spec)


def _build_sweep_plot_callable(spec: SweepPlotSpec) -> tuple[Any, str]:
    """Resolve a plot spec into a figure-producing callable and filename stem."""
    return (
        _neuroinfra_build_sweep_plot_callable(spec, plot_resolver=_get_builtin_sweep_plot),
        _sweep_plot_artifact_stem(spec),
    )


def _is_deprecated_sweep_animation_spec(spec: SweepPlotSpec) -> bool:
    """Return True for retired sweep GIF specs that should be skipped."""
    return _neuroinfra_is_deprecated_sweep_animation_spec(
        spec,
        deprecated_names=_DEPRECATED_SWEEP_ANIMATION_PLOTS,
    )


def _format_sweep_value(value: Any) -> str:
    """Format a sweep value compactly for figure titles."""
    return _neuroinfra_format_sweep_value(value)


def _format_sweep_value_label(sweep: dict[str, Any], value: Any) -> str:
    """Format one sweep-path/value label for animation titles."""
    return _neuroinfra_format_sweep_value_label(sweep, value)


def _format_sweep_progress_label(frame_index: int, total_frames: int, *, width: int = 12) -> str:
    """Format one compact sweep-progress label without ASCII bar glyphs."""
    return _neuroinfra_format_sweep_progress_label(frame_index, total_frames, width=width)


def _format_sweep_frame_title(sweep: dict[str, Any], value: Any, frame_index: int, total_frames: int) -> str:
    """Build one default animation title with value and sweep progress."""
    return _neuroinfra_format_sweep_frame_title(sweep, value, frame_index, total_frames)


def _describe_unavailable_sweep_item(item: dict[str, Any]) -> str:
    """Return a compact reason for a missing/unrenderable sweep frame."""
    return _neuroinfra_describe_unavailable_sweep_item(item)


def _make_sweep_placeholder_figure(
    sweep: dict[str, Any],
    item: dict[str, Any],
    frame_index: int,
    total_frames: int,
    *,
    reason: str,
    figsize: tuple[float, float],
) -> Any:
    """Render an explicit placeholder frame instead of aborting partial sweeps."""
    return _neuroinfra_make_sweep_placeholder_figure(
        sweep,
        item,
        frame_index,
        total_frames,
        reason=str(reason)[:500],
        figsize=figsize,
    )


def _safe_name(name: Any) -> str:
    """Make a filesystem-safe artifact basename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "animation"


def _callable_artifact_label(value: Any) -> str:
    """Return a stable, compact label for functions used in artifact settings."""
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if qualname:
        return f"{module}.{qualname}" if module and module != "__main__" else str(qualname)
    return value.__class__.__name__


def _artifact_settings_ready(value: Any) -> Any:
    """Convert plot settings into deterministic JSON-compatible metadata."""
    if isinstance(value, FrequencyPlotConfig):
        config = asdict(value)
        if str(config.get("kde1d_engine", "histogram")).strip().lower() == "histogram":
            config.pop("kde1d_engine", None)
        return _artifact_settings_ready(config)
    if is_dataclass(value) and not isinstance(value, type):
        return _artifact_settings_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, range):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _artifact_settings_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_artifact_settings_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_artifact_settings_ready(item) for item in value)
    if callable(value):
        return _callable_artifact_label(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def _short_artifact_setting_value(value: Any, *, max_len: int = 32) -> str:
    """Format one settings value for a human-readable artifact filename suffix."""
    if value is None:
        text = "none"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, (int, str)):
        text = str(value)
    elif isinstance(value, list):
        if len(value) <= 8 and all(not isinstance(item, (dict, list)) for item in value):
            text = "-".join(_short_artifact_setting_value(item, max_len=12) for item in value)
        else:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
            text = f"list{len(value)}_{sha1(encoded.encode('utf-8')).hexdigest()[:8]}"
    elif isinstance(value, dict):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        text = f"cfg_{sha1(encoded.encode('utf-8')).hexdigest()[:8]}"
    else:
        text = str(value)
    safe = _safe_name(text)
    if len(safe) > max_len:
        safe = f"{safe[: max_len - 9]}_{sha1(safe.encode('utf-8')).hexdigest()[:8]}"
    return safe


def _flatten_artifact_settings(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten deterministic settings into key/value pairs for filename labels."""
    if isinstance(value, dict):
        pairs: list[tuple[str, Any]] = []
        for key in sorted(value):
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            pairs.extend(_flatten_artifact_settings(value[key], prefix=next_prefix))
        return pairs
    return [(prefix, value)]


def _sweep_plot_artifact_stem(spec: SweepPlotSpec) -> str:
    """Build a settings-aware, collision-resistant sweep animation filename stem."""
    base = str(spec.filename or spec.name or (spec.plot if isinstance(spec.plot, str) else "custom_plot"))
    settings_payload = {
        "plot": spec.plot if isinstance(spec.plot, str) else _callable_artifact_label(spec.plot),
        "plot_kwargs": _artifact_settings_ready(spec.plot_kwargs or {}),
        "figsize": _artifact_settings_ready(spec.figsize),
        "interval": int(spec.interval),
        "fps": int(spec.fps),
    }
    if spec.title_fn is not None:
        settings_payload["title_fn"] = _callable_artifact_label(spec.title_fn)
    encoded = json.dumps(settings_payload, sort_keys=True, separators=(",", ":"))
    digest = sha1(encoded.encode("utf-8")).hexdigest()[:10]

    flat_settings = _flatten_artifact_settings(settings_payload.get("plot_kwargs") or {})
    flat_settings.extend(
        [
            ("figsize", settings_payload["figsize"]),
            ("interval", settings_payload["interval"]),
            ("fps", settings_payload["fps"]),
        ]
    )
    if "title_fn" in settings_payload:
        flat_settings.append(("title", settings_payload["title_fn"]))

    parts = [
        f"{_safe_name(key)}-{_short_artifact_setting_value(value)}"
        for key, value in flat_settings
        if key
    ]
    suffix = "_".join(part for part in parts if part)
    max_suffix_len = 120
    if len(suffix) > max_suffix_len:
        suffix = suffix[:max_suffix_len].rstrip("_")
    if suffix:
        return _safe_name(f"{base}__{suffix}__{digest}")
    return _safe_name(f"{base}__{digest}")


def _fig_to_rgb_array(fig: Any) -> np.ndarray:
    """Render a matplotlib figure to an H×W×3 uint8 numpy array."""
    return _neuroinfra_fig_to_rgb_array(fig)


def _render_sweep_frame(
    sweep: dict[str, Any],
    item: dict[str, Any],
    frame_index: int,
    total_frames: int,
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
) -> tuple[np.ndarray, str]:
    """Render one sweep item to a frame array and title."""
    return _neuroinfra_render_sweep_frame(
        sweep,
        item,
        frame_index,
        total_frames,
        plot_fn,
        figsize=figsize,
        title_fn=title_fn,
        close_frames=close_frames,
    )


def _compose_sweep_display_frame(
    frame_rgb: np.ndarray,
    title: str,
    *,
    figsize: tuple[float, float],
    frame_index: int | None = None,
    total_frames: int | None = None,
) -> np.ndarray:
    """Compose one sweep frame with a baked-in title and visual progress bar."""
    return _neuroinfra_compose_sweep_display_frame(
        frame_rgb,
        title,
        figsize=figsize,
        frame_index=frame_index,
        total_frames=total_frames,
    )


def _iter_sweep_animation_frames(
    sweep: dict[str, Any],
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
) -> Any:
    """Yield rendered sweep animation frames one at a time."""
    yield from _neuroinfra_iter_sweep_animation_frames(
        sweep,
        plot_fn,
        figsize=figsize,
        title_fn=title_fn,
        close_frames=close_frames,
    )


def _default_sweep_animation_worker_count(frame_count: int) -> int:
    """Choose a safe default worker count for CPU-bound Matplotlib frame rendering."""
    return _neuroinfra_default_sweep_animation_worker_count(
        frame_count,
        env_var_name="OBGPU_SWEEP_RENDER_WORKERS",
    )


def _iter_parallel_sweep_display_frames(
    sweep: dict[str, Any],
    plot_fn: Any,
    *,
    figsize: tuple[float, float],
    title_fn: Any = None,
    close_frames: bool = True,
    workers: int | None = None,
) -> Any:
    """Yield composed frames in order while rendering independent frames concurrently."""
    yield from _neuroinfra_iter_parallel_sweep_display_frames(
        sweep,
        plot_fn,
        figsize=figsize,
        title_fn=title_fn,
        close_frames=close_frames,
        workers=workers,
        env_var_name="OBGPU_SWEEP_RENDER_WORKERS",
    )


def animate_sweep(
    sweep: dict[str, Any],
    plot_fn: Any,
    figsize: tuple[float, float] = (12, 5),
    interval: int = 100,
    title_fn: Any = None,
    close_frames: bool = True,
) -> animation.FuncAnimation:
    """Animate any plot function across a parameter sweep.

    ``plot_fn(result) -> matplotlib.Figure`` is called once per available
    sweep item. The figure is rendered to a pixel array so *any* plotting code
    works — multi-panel layouts, seaborn, custom axes, etc. Missing or failed
    items are rendered as explicit placeholder frames.

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
    return _neuroinfra_animate_sweep(
        sweep,
        plot_fn,
        figsize=figsize,
        interval=interval,
        title_fn=title_fn,
        close_frames=close_frames,
    )


def animate_lfp_sweep(
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    interval: int = 100,
) -> animation.FuncAnimation:
    """Animate trace-style outputs across a one-parameter sweep."""
    if signal != "lfp":
        return animate_sweep(
            sweep,
            lambda result: plot_named_signal(result, signal=signal, dt_ms=dt_ms),
            figsize=(12, 4),
            interval=interval,
        )

    return animate_sweep(
        sweep,
        lambda result: plot_lfp_overview(result, dt_ms=dt_ms),
        figsize=(12, 7),
        interval=interval,
    )


def animate_spectrogram_sweep(
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    max_freq_hz: float = 250.0,
    nperseg: int = 256,
    noverlap: int = 192,
    interval: int = 100,
) -> animation.FuncAnimation:
    """Animate spectrograms across a one-parameter sweep."""
    return animate_sweep(
        sweep,
        lambda result: plot_spectrogram(
            result,
            signal=signal,
            dt_ms=dt_ms,
            max_freq_hz=max_freq_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        ),
        figsize=(12, 4),
        interval=interval,
    )


def animate_wavelet_sweep(
    sweep: dict[str, Any],
    signal: str = "lfp",
    dt_ms: float = 0.1,
    interval: int = 100,
) -> animation.FuncAnimation:
    """Animate wavelet maps across a one-parameter sweep."""
    return animate_sweep(
        sweep,
        lambda result: plot_wavelet(result, signal=signal, dt_ms=dt_ms),
        figsize=(12, 4),
        interval=interval,
    )


def animate_sniff_average_sweep(
    sweep: dict[str, Any],
    dt_ms: float = 0.1,
    sniff_count: int = 8,
    interval: int = 100,
) -> animation.FuncAnimation:
    """Animate sniff-averaged wavelet views across a sweep."""
    def _plot(result: dict[str, Any]) -> Any:
        signal_t, signal_y = get_named_signal(result, signal="lfp", dt_ms=dt_ms)
        _t, _bp, freqs, power = compute_wavelet_map(signal_t, signal_y, dt_ms=dt_ms)
        sniff_duration_ms = 200.0
        skip_first_n_sniffs = 1
        step = max(1, int(round(sniff_duration_ms / dt_ms)))
        start_index = step * skip_first_n_sniffs
        available_columns = max(0, power.shape[1] - start_index)
        chunk_count = min(int(sniff_count), available_columns // step)
        if chunk_count > 0:
            chunks = [
                power[:, start_index + i * step : start_index + (i + 1) * step]
                for i in range(chunk_count)
            ]
            averaged = np.mean(np.asarray(chunks, dtype=float), axis=0)
        else:
            averaged = power[:, :step]
        plot_t = np.arange(averaged.shape[1], dtype=float) * dt_ms
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.contourf(
            plot_t,
            freqs,
            averaged,
            256,
            cmap="jet",
        )
        ax.set_ylim((20, 140))
        ax.set_xlabel("Time Since Sniff Onset [ms]")
        ax.set_ylabel("Frequency [Hz]")
        return fig

    return animate_sweep(sweep, _plot, figsize=(5, 5), interval=interval)


SWEEPS_BASE = DEFAULT_RESULTS_BASE / "sweeps"


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
    return _neuroinfra_save_sweep(
        sweep,
        name=name,
        base_dir=base_dir or SWEEPS_BASE,
        timestamp_factory=make_timestamp,
        safe_name=_safe_name,
        json_ready=_json_ready,
        resolve_git_head=_resolve_local_git_head,
    )


def load_sweep(path: str | Path) -> dict[str, Any]:
    """Reconstruct a sweep dict from a directory created by save_sweep.

    Results are loaded lazily (same as load_result) so re-animating old
    sweeps does not require loading all soma traces upfront.
    """
    return _neuroinfra_load_sweep(
        path,
        load_result_fn=lambda result_dir: load_result(result_dir, progress=False),
        safe_name=_safe_name,
    )


def list_sweeps(
    prefix: str | None = None,
    base_dir: str | Path | None = None,
) -> list[Path]:
    """Return saved sweep directories sorted from oldest to newest."""
    return _neuroinfra_list_sweeps(base_dir=base_dir or SWEEPS_BASE, prefix=prefix)


def save_animation(
    anim: animation.FuncAnimation,
    name: str,
    output_dir: str | Path | None = None,
    sweep: dict[str, Any] | None = None,
    fps: int = 10,
) -> Path:
    """Save an animation as a GIF and return the written path.

    When ``sweep`` is provided and has a ``sweep_dir``, the GIF is saved to
    ``sweep_dir/animations/`` automatically (``output_dir`` is ignored).
    """
    return _neuroinfra_save_animation(
        anim,
        name,
        safe_name=_safe_name,
        output_dir=output_dir,
        sweep=sweep,
        fps=fps,
        default_output_dir_factory=lambda: DEFAULT_RESULTS_BASE / "animations" / make_timestamp(),
    )


def save_sweep_animation_stream(
    sweep: dict[str, Any],
    plot_fn: Any,
    name: str,
    *,
    output_dir: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    interval: int = 100,
    title_fn: Any = None,
    close_frames: bool = True,
    fps: int = 10,
    workers: int | None = None,
) -> Path:
    """Render and save a sweep GIF without retaining all frames in memory."""
    progress = _ProgressBar(
        total=len(sweep["items"]),
        desc=f"[OBGPU load] Render {name}",
        unit="frame",
        unit_scale=False,
        display_step=max(1, len(sweep["items"]) // 100),
    )
    try:
        return _neuroinfra_save_sweep_animation_stream(
            sweep,
            plot_fn,
            name,
            safe_name=_safe_name,
            output_dir=output_dir,
            figsize=figsize,
            title_fn=title_fn,
            close_frames=close_frames,
            fps=fps,
            workers=workers,
            env_var_name="OBGPU_SWEEP_RENDER_WORKERS",
            progress_callback=lambda current, total: progress.update_to(current),
            default_output_dir_factory=lambda: DEFAULT_RESULTS_BASE / "animations" / make_timestamp(),
        )
    finally:
        progress.close()


def animate_sweep_plots(
    sweep: dict[str, Any],
    plots: list[SweepPlotSpec | str | Any | dict[str, Any]],
    *,
    close_frames: bool = True,
    stream: bool = True,
    workers: int | None = None,
) -> dict[str, Path]:
    """Render and save multiple sweep animations from one completed sweep.

    Each entry in ``plots`` may be:

    - a built-in plot name from :func:`get_builtin_sweep_plot_names`
    - a custom ``plot_fn(result) -> fig`` callable defined in the notebook
    - a :class:`SweepPlotSpec`
    - a dict like ``{"plot": "spike_frequency_kde_2d", "plot_kwargs": {...}}``
    """
    deprecated_names = set(_DEPRECATED_SWEEP_ANIMATION_PLOTS)
    for raw_spec in plots:
        spec = _normalize_sweep_plot_spec(raw_spec)
        if _is_deprecated_sweep_animation_spec(spec):
            _progress_write(
                f"[OBGPU load] Skipping deprecated sweep animation plot {spec.name!r}."
            )
    return _neuroinfra_animate_sweep_plots(
        sweep,
        plots,
        plot_builder=_build_sweep_plot_callable,
        safe_name=_safe_name,
        deprecated_names=deprecated_names,
        close_frames=close_frames,
        stream=stream,
        workers=workers,
        env_var_name="OBGPU_SWEEP_RENDER_WORKERS",
        default_output_dir_factory=lambda: DEFAULT_RESULTS_BASE / "animations" / make_timestamp(),
    )


def run_sweep_with_animations(
    base_config: dict[str, Any],
    sweep_path: str | list[str] | dict[str, list[Any]],
    values: list[Any] | tuple[Any, ...] | None = None,
    *,
    plots: list[SweepPlotSpec | str | Any | dict[str, Any]] | None = None,
    use_grid: bool = False,
    close_frames: bool = True,
    workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run a sweep once, then emit one or more animations over the same results."""
    if use_grid:
        if not isinstance(sweep_path, dict):
            raise TypeError("Grid sweeps require sweep_path to be a dict of {path: values}")
        sweep = run_grid_sweep(base_config, sweep_path)
    else:
        sweep = run_parameter_sweep(base_config, sweep_path, values)

    artifacts: dict[str, Path] = {}
    if plots:
        artifacts = animate_sweep_plots(sweep, plots, close_frames=close_frames, workers=workers)
    return sweep, artifacts


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
    show_raw_voltage_traces = bool(config.get("show_voltage_traces", False))
    show_psd_template = bool(config.get("lfp_show_psd_target_template", True))
    psd_template_kind = str(config.get("lfp_psd_template_kind", "ketamine"))
    psd_template_fit = config.get("lfp_psd_template_fit_band_hz", DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ)
    psd_template_floor = config.get("lfp_psd_template_floor", DEFAULT_PSD_TEMPLATE_FLOOR)
    if isinstance(psd_template_fit, (list, tuple)) and len(psd_template_fit) == 2:
        psd_template_fit = (float(psd_template_fit[0]), float(psd_template_fit[1]))
    else:
        psd_template_fit = DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ
    try:
        psd_template_floor = float(psd_template_floor)
    except (TypeError, ValueError):
        psd_template_floor = DEFAULT_PSD_TEMPLATE_FLOOR
    psd_xlim_hz = config.get("lfp_psd_xlim_hz", (0.0, 300.0))
    if isinstance(psd_xlim_hz, (list, tuple)) and len(psd_xlim_hz) == 2:
        psd_xlim_hz = (float(psd_xlim_hz[0]), float(psd_xlim_hz[1]))
    else:
        psd_xlim_hz = None
    spectrogram_max_freq_hz = float(config.get("spectrogram_max_freq_hz", 250.0))
    spectrogram_nperseg = int(config.get("spectrogram_nperseg", 256))
    spectrogram_noverlap = int(config.get("spectrogram_noverlap", 192))

    plot_input_overview(
        result,
        bin_ms=input_bin_ms,
        smooth_sigma_ms=input_smooth_ms,
        max_segments=input_max_segments,
        normalization=input_norm,
    )
    plt.show()

    if show_raw_voltage_traces:
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

    plot_lfp_overview(
        result,
        dt_ms=dt_ms,
        show_psd_target_template=show_psd_template,
        psd_template_kind=psd_template_kind,
        psd_template_fit_band_hz=psd_template_fit,
        psd_template_floor=psd_template_floor,
        psd_xlim_hz=psd_xlim_hz,
    )
    plt.show()

    plot_spectrogram(
        result,
        signal=config.get("spectrogram_signal", "lfp"),
        dt_ms=dt_ms,
        max_freq_hz=spectrogram_max_freq_hz,
        nperseg=spectrogram_nperseg,
        noverlap=spectrogram_noverlap,
    )
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
