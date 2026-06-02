"""Machine-readable inventory for reusable infrastructure extraction candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExtractionCandidate:
    """One subsystem that may be extracted into a reusable framework package."""

    key: str
    title: str
    target_module: str
    source_paths: tuple[str, ...]
    generic_capabilities: tuple[str, ...]
    repo_specific_couplings: tuple[str, ...]
    extraction_confidence: str
    proposed_phase: int
    current_status: str
    recommended_action: str

    def source_path_status(self) -> dict[str, bool]:
        return {
            path: (REPO_ROOT / path).exists()
            for path in self.source_paths
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_paths_exist"] = self.source_path_status()
        return payload


EXTRACTION_CANDIDATES: tuple[ExtractionCandidate, ...] = (
    ExtractionCandidate(
        key="audit_framework",
        title="Audit framework",
        target_module="neuroinfra.audit",
        source_paths=(
            "olfactorybulb/audit/core.py",
            "olfactorybulb/audit/cli.py",
            "olfactorybulb/audit/registry.py",
            "olfactorybulb/audit/neuron_protocols.py",
        ),
        generic_capabilities=(
            "audit item and report datatypes",
            "colored and JSON audit rendering",
            "registry-based discovery",
            "compound multi-audit runs",
        ),
        repo_specific_couplings=(
            "individual audit implementations remain domain-specific",
            "some protocol helpers assume NEURON workloads",
        ),
        extraction_confidence="high",
        proposed_phase=1,
        current_status="candidate",
        recommended_action="Extract core/cli/registry first and leave current olfactory-bulb audits as plugin consumers.",
    ),
    ExtractionCandidate(
        key="artifact_and_output_paths",
        title="Result artifact and output-path helpers",
        target_module="neuroinfra.artifacts",
        source_paths=(
            "neuroinfra/artifacts/loading.py",
            "neuroinfra/artifacts/result_view.py",
            "neuroinfra/artifacts/output_paths.py",
            "neuroinfra/artifacts/result_artifacts.py",
            "olfactorybulb/result_artifacts.py",
            "olfactorybulb/output_paths.py",
        ),
        generic_capabilities=(
            "timestamped run labels",
            "artifact versioning",
            "compact NPZ/PKL persistence",
            "saved trace discovery",
            "spike detection from saved voltages",
            "lazy artifact result containers",
            "timed local artifact load plans with progress reporting",
            "result-view planning with eager and deferred artifact wiring",
            "configurable result schemas for default fields and artifact application",
        ),
        repo_specific_couplings=(
            "artifact names still use OBGPU terminology like soma_vs and lfp",
        ),
        extraction_confidence="high",
        proposed_phase=1,
        current_status="internal_shim_extracted",
        recommended_action="The first internal extraction has been done behind compatibility shims, and the local artifact-loading helpers plus the generic result-view planner/schema now live under neuroinfra.artifacts; next remove remaining OBGPU-specific naming and separate the last notebook-specific signal analysis and presentation policy from the OBGPU artifact plan.",
    ),
    ExtractionCandidate(
        key="notebook_run_catalog",
        title="Notebook run catalog and metadata",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/runs.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "saved run metadata datatypes",
            "run directory listing and prefix filtering",
            "prefix and index based run resolution",
            "captured stdout and stderr recovery",
            "saved config snapshot reload",
        ),
        repo_specific_couplings=(
            "the wrapper still chooses the default OBGPU results base",
            "config normalization after reload remains domain-specific",
            "summary and run_info filename conventions still come from this repo's run layout",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic run-directory catalog and metadata loader now live under neuroinfra.notebooks.runs; next separate the remaining config normalization and simulation/result wiring from the notebook helper.",
    ),
    ExtractionCandidate(
        key="notebook_config_store",
        title="Notebook config persistence and catalog",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/config_store.py",
            "olfactorybulb/notebook_configs.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "JSON-ready conversion for notebook config payloads",
            "config save and reload helpers",
            "saved config file discovery",
        ),
        repo_specific_couplings=(
            "odor-schedule normalization after reload remains olfactory-bulb-specific",
            "the built-in paramset catalog is domain-specific",
            "effective-param diffing still depends on the repo's paramset semantics",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic config persistence helpers now live under neuroinfra.notebooks.config_store, while the olfactory-bulb-specific normalization and paramset catalog live in olfactorybulb.notebook_configs; next separate the remaining simulation/result wiring from the notebook helper.",
    ),
    ExtractionCandidate(
        key="notebook_reporting",
        title="Notebook reporting and figure output",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/reporting.py",
            "olfactorybulb/notebook_reports.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "nested payload flattening for diff reports",
            "stable nested value diff generation",
            "human-readable diff section rendering",
            "figure save helpers with run and sweep aware output directories",
        ),
        repo_specific_couplings=(
            "run-summary content and section ordering remain olfactory-bulb-specific",
            "default figure output roots still come from the notebook helper",
            "effective-param and runtime-control summaries still depend on repo-specific config semantics",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic notebook diff/report/save helpers now live under neuroinfra.notebooks.reporting, while the olfactory-bulb-specific run-summary presentation lives in olfactorybulb.notebook_reports; next keep shrinking the notebook helper by separating the remaining simulation/result entrypoint glue.",
    ),
    ExtractionCandidate(
        key="notebook_sweep_planning",
        title="Notebook sweep planning",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/sweeps.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "nested config path splitting with indexed list support",
            "nested config value assignment for dict and list payloads",
            "single-axis sweep expansion",
            "joint sweep expansion",
            "grid sweep expansion",
            "hook-driven sweep label and timestamp policy",
        ),
        repo_specific_couplings=(
            "label and timestamp policy still come from the notebook helper",
            "base config normalization still depends on the repo's run config defaults",
            "actual local and remote sweep execution still remain in obgpu_experiment_helpers.py",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic sweep-planning layer now lives under neuroinfra.notebooks.sweeps, while the notebook helper still owns concrete run-config normalization, sweep execution, and persistence; next separate the remaining local/remote sweep runner orchestration from planning.",
    ),
    ExtractionCandidate(
        key="notebook_local_run_execution",
        title="Notebook local subprocess execution",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/local_runs.py",
            "olfactorybulb/notebook_local_runs.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "local subprocess execution with captured stdout and stderr",
            "command and capture artifact persistence",
            "required summary-file enforcement",
            "standard failure-message rendering",
            "hook-driven run metadata persistence and return-value construction",
        ),
        repo_specific_couplings=(
            "local env, override-file payload, and return-value semantics still remain repo-specific",
            "the concrete run-info and effective-param hooks still come from the olfactory-bulb notebook layer",
            "higher-level local versus remote dispatch still remains outside this layer",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic local notebook subprocess runner now lives under neuroinfra.notebooks.local_runs, while the concrete olfactory-bulb local payload and hook adapters now live in olfactorybulb.notebook_local_runs; next keep shrinking the notebook helper by separating the remaining higher-level runner selection and execution orchestration.",
    ),
    ExtractionCandidate(
        key="notebook_run_info_protocol",
        title="Notebook run-info protocol",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/run_info.py",
            "olfactorybulb/notebook_run_info.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "run_info payload building with preserved existing fields",
            "env subset capture for persisted run metadata",
            "hook-driven override, execution-mode, and effective-param payloads",
            "run_info merge helpers for post-load artifact metadata",
        ),
        repo_specific_couplings=(
            "the concrete env key set is still chosen by the olfactory-bulb notebook layer",
            "effective-param and execution-mode semantics still depend on the repo's config model",
            "the notebook helper still chooses when local, remote, and sweep paths persist metadata",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic notebook run-info protocol now lives under neuroinfra.notebooks.run_info, while the olfactory-bulb-specific payload semantics live in olfactorybulb.notebook_run_info; next keep shrinking the helper by separating the remaining local/remote runner orchestration that decides when those payloads are written.",
    ),
    ExtractionCandidate(
        key="notebook_workflows",
        title="Notebook run and sweep workflows",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/workflows.py",
            "olfactorybulb/notebook_workflows.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "run and load workflow composition",
            "saved run pair loading",
            "local sweep-plan execution loops",
            "hook-driven post-load metadata merging",
        ),
        repo_specific_couplings=(
            "progress messages still come from the notebook helper",
            "local sweep path policy still comes from the repo's results layout",
            "concrete workflow hook assembly still remains repo-specific",
            "remote sweep and remote run orchestration remain outside this layer",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic notebook workflow layer now lives under neuroinfra.notebooks.workflows, while the concrete olfactory-bulb hook assembly now lives in olfactorybulb.notebook_workflows; next separate the remaining remote runner orchestration from the notebook helper.",
    ),
    ExtractionCandidate(
        key="notebook_remote_jobs",
        title="Notebook remote job session and submit protocol",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/remote_jobs.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "remote git publication and preflight session lifecycle",
            "helper-cache and reusable-allocation preparation",
            "submit stdout/stderr persistence with JSON response parsing",
        ),
        repo_specific_couplings=(
            "remote payload construction still depends on the repo's benchmark command conventions",
            "run_info failure handling still depends on the notebook helper's reporting policy",
            "remote monitoring, finalization, and result loading still remain in the notebook facade",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic notebook remote session lifecycle and JSON submit protocol now live under neuroinfra.notebooks.remote_jobs, while the helper still owns concrete remote payload construction, failure reporting, monitoring, finalization, and result loading; next separate the remaining remote run and remote sweep orchestration above this session layer.",
    ),
    ExtractionCandidate(
        key="notebook_remote_run_workflow",
        title="Notebook remote single-run workflow",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/remote_runs.py",
            "olfactorybulb/notebook_remote_runs.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "remote preflight failure persistence and error reporting",
            "remote single-run submit and monitor workflow composition",
            "remote final artifact handling and run-info persistence orchestration",
        ),
        repo_specific_couplings=(
            "remote payload construction still depends on the repo's benchmark command conventions",
            "monitor and artifact hook wiring still depends on the repo's remote helper set",
            "returned run record shape still depends on the repo's domain model",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic notebook remote single-run workflow now lives under neuroinfra.notebooks.remote_runs, and the concrete olfactory-bulb run payload/workflow adapters now live under olfactorybulb.notebook_remote_runs; next keep shrinking obgpu_experiment_helpers.py by moving the last notebook-entrypoint glue and shared domain adapters out of it.",
    ),
    ExtractionCandidate(
        key="notebook_remote_sweep_workflow",
        title="Notebook remote sweep workflow",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/remote_sweeps.py",
            "olfactorybulb/notebook_remote_sweeps.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "remote sweep manifest upload and submit workflow composition",
            "remote sweep monitoring and compact final-sync orchestration",
            "partial-result bookkeeping and saved sweep persistence",
        ),
        repo_specific_couplings=(
            "manifest item construction still depends on the repo's benchmark command conventions",
            "incremental item sync and item-finalization hooks still depend on the repo's result layout",
            "returned sweep item payloads still depend on the repo's domain result model",
        ),
        extraction_confidence="high",
        proposed_phase=2,
        current_status="internal_shim_extracted",
        recommended_action="The generic notebook remote sweep workflow now lives under neuroinfra.notebooks.remote_sweeps, and the concrete olfactory-bulb sweep payload/workflow adapters now live under olfactorybulb.notebook_remote_sweeps; next separate the analogous remote single-run domain adapters and keep shrinking obgpu_experiment_helpers.py toward notebook entrypoint glue.",
    ),
    ExtractionCandidate(
        key="remote_slurm_execution",
        title="Remote Slurm execution layer",
        target_module="neuroinfra.remote.slurm",
        source_paths=(
            "neuroinfra/remote/config.py",
            "neuroinfra/remote/command_launch.py",
            "neuroinfra/remote/helper_bundle.py",
            "neuroinfra/remote/allocation_runtime.py",
            "neuroinfra/remote/notebook_runtime.py",
            "neuroinfra/remote/paramiko_transport.py",
            "neuroinfra/remote/sftp_sync.py",
            "neuroinfra/remote/archive_stream.py",
            "neuroinfra/remote/stream_sync.py",
            "neuroinfra/remote/result_sync.py",
            "neuroinfra/remote/deferred_artifacts.py",
            "neuroinfra/remote/status_poll.py",
            "neuroinfra/remote/run_artifacts.py",
            "neuroinfra/remote/run_monitor.py",
            "neuroinfra/remote/sweep_monitor.py",
            "neuroinfra/remote/sweep_artifacts.py",
            "neuroinfra/remote/slurm_launch.py",
            "neuroinfra/remote/slurm_state.py",
            "neuroinfra/remote_script_common.py",
            "neuroinfra/remote_script_submit.py",
            "neuroinfra/remote_script_polling.py",
            "neuroinfra/remote_script_allocations.py",
            "neuroinfra/remote_script_sweeps.py",
            "neuroinfra/remote/git_sync.py",
            "neuroinfra/remote/helper_cache.py",
            "neuroinfra/remote/allocation_cache.py",
            "tools/remote/slurm_common.py",
            "tools/remote/submit_sol_run.py",
            "tools/remote/submit_slurm_allocation.py",
            "tools/remote/remote_sweep_driver.py",
            "tools/remote/poll_sol_run.py",
        ),
        generic_capabilities=(
            "sbatch directive generation",
            "remote wrapper scripting",
            "batch fanout inside one allocation",
            "per-run worktree path relocation",
            "MPI preflight generation",
        ),
        repo_specific_couplings=(
            "OBGPU-prefixed environment variables",
            "benchmark command conventions",
            "cluster bootstrap defaults from this repo",
        ),
        extraction_confidence="medium-high",
        proposed_phase=2,
        current_status="config_sync_launch_state_transport_allocation_runtime_sftp_stream_result_sync_deferred_artifact_status_poll_run_artifacts_run_monitor_sweep_monitor_sweep_artifacts_script_common_submit_polling_allocations_sweeps_git_helper_cache_and_allocation_seams_standardized",
        recommended_action="The remote config-normalization, helper-bundle packaging, helper-cache lifecycle, allocation cache policy, notebook-managed reusable-allocation orchestration, low-level Paramiko archive/direct stream helpers, higher-level result-sync retry/fallback policy, deferred remote-artifact sync policy, shared JSON status-poll retry/parsing, remote single-run final sync and artifact-collection policy, remote single-run live monitoring policy, remote sweep live monitoring policy, remote sweep compact final-sync/finalization policy, notebook runtime/session policy, Paramiko transport/session logic, SFTP sync loops, archive-stream builders, Slurm state/preflight helpers, remote-script common helpers, remote single-run submit helpers, remote polling/status helpers, remote allocation lifecycle helpers, remote sweep runner helpers, Slurm helper argv/launch assembly, and local Git publication/base-resolution helpers now live under neuroinfra; next focus on the remaining notebook-facade orchestration in obgpu_experiment_helpers.py rather than low-level remote plumbing.",
    ),
    ExtractionCandidate(
        key="campaign_archive_framework",
        title="Campaign and optimizer archive framework",
        target_module="neuroinfra.campaigns",
        source_paths=(
            "neuroinfra/campaigns/store.py",
            "olfactorybulb/hfo_optimizer.py",
            "tools/run_hfo_campaign.py",
        ),
        generic_capabilities=(
            "campaign directory layout",
            "batch plan/run/score lifecycle",
            "archive and state persistence",
            "Latin-hypercube seeding",
            "elite and exploration proposal patterns",
        ),
        repo_specific_couplings=(
            "HFO scoring semantics",
            "paired ketamine/control condition logic",
            "olfactory-bulb-specific metrics and penalties",
        ),
        extraction_confidence="medium",
        proposed_phase=4,
        current_status="store_seams_standardized",
        recommended_action="The generic campaign filesystem/state/archive layer now lives under neuroinfra.campaigns; next separate proposal/storage interfaces from the HFO-specific scorer and candidate metrics.",
    ),
    ExtractionCandidate(
        key="contracts_and_registries",
        title="Metadata contract and registry pattern",
        target_module="neuroinfra.contracts",
        source_paths=(
            "neuroinfra/contracts/parameters.py",
            "neuroinfra/contracts/visuals.py",
            "olfactorybulb/hfo_features.py",
            "olfactorybulb/hfo_visuals.py",
            "olfactorybulb/audit/hfo_feature_contracts.py",
            "prev_ob_models/cell_registry.py",
        ),
        generic_capabilities=(
            "single-source-of-truth parameter registries",
            "visual contract snapshots",
            "model registry metadata",
            "contract drift audits",
        ),
        repo_specific_couplings=(
            "concrete HFO parameter set",
            "concrete olfactory-bulb cell catalog",
        ),
        extraction_confidence="high",
        proposed_phase=3,
        current_status="internal_shim_extracted",
        recommended_action="The generic parameter-space and visual-contract helpers now live under neuroinfra.contracts; next reduce the remaining HFO-specific coupling by separating concrete plot families from the shared manifest schema.",
    ),
    ExtractionCandidate(
        key="analysis_signal_registry",
        title="Result analysis and signal registry",
        target_module="neuroinfra.analysis",
        source_paths=(
            "neuroinfra/analysis/events.py",
            "neuroinfra/analysis/frequency_plots.py",
            "neuroinfra/analysis/catalog.py",
            "neuroinfra/analysis/overview.py",
            "neuroinfra/analysis/phase_locking.py",
            "neuroinfra/analysis/plotting.py",
            "neuroinfra/analysis/profiles.py",
            "neuroinfra/analysis/grouped_views.py",
            "neuroinfra/analysis/signal_views.py",
            "neuroinfra/analysis/spectral.py",
            "neuroinfra/analysis/sweeps.py",
            "neuroinfra/analysis/signals.py",
            "olfactorybulb/analysis_data.py",
            "olfactorybulb/analysis_hfo_views.py",
            "olfactorybulb/analysis_presentations.py",
            "olfactorybulb/analysis_profile.py",
            "olfactorybulb/analysis_views.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "stable category and label cataloging",
            "stable ordered-name helpers with preferred ordering and unknown-last handling",
            "fair round-robin subgroup truncation for merged display buckets",
            "ordered group-row flattening across display buckets with per-bucket limits",
            "result-overview context and summary builders",
            "shared plotting primitives for traces, time-frequency maps, and band-power summaries",
            "stacked labeled trace plotting with configurable offsets and styling",
            "grouped row-display policies for stable bucketed ordering and per-bucket limits",
            "grouped stacked-trace and event-raster suites built on those row policies",
            "domain analysis profiles that aggregate concrete signal, event, frequency, and sweep suites",
            "uniform-trace interpolation and modulus folding",
            "shared spectrogram, wavelet, and band-power analysis",
            "named-signal trace, band-pass, PSD overview, spectrogram, wavelet, and band-power view helpers",
            "phase-locking summaries from resolved signals and labeled spike-time rows",
            "frequency KDE and time-binned plotting from precomputed sample arrays",
            "result-backed frequency plot families for 1D KDE, 2D KDE, and time-binned rendering",
            "family-bound result-frequency plotting suites",
            "instantaneous frequency sample collection from labeled event rows",
            "trace-derived instantaneous frequency sample collection from labeled continuous-trace rows",
            "event-frequency conversion, binned event-rate analysis, rate-plot helpers, and raster-plot primitives",
            "row filtering by label-prefix families",
            "normalization-driven event-rate computation from arbitrary event rows",
            "result-backed event-family specs for filtering, frequency sample collection, and normalized rate computation",
            "result-backed event family suites with reusable t_stop inference hooks",
            "result-backed event plot suites for raster, rate, and overview composition",
            "reusable event-rate series assembly for named subset plots",
            "prepared labeled event display rows and overview-layout derivation from them",
            "shared raster-plus-rate overview layout",
            "sweep plot specification and placeholder rendering",
            "named sweep plot registries with deprecation-aware resolution",
            "sweep metadata persistence and reload",
            "streamed and in-memory GIF rendering for sweep frames",
            "ordered named-signal providers",
            "ordered named-signal registries",
            "registry-backed resolved-signal view suites",
            "dynamic signal enumeration",
            "provider-based signal resolution",
            "provider factories for keyed traces, suffix variants, pattern-matched signals, and labeled traces",
            "aligned mean traces from grouped time/value rows",
            "decoupled analysis-signal catalogs from notebook facades",
        ),
        repo_specific_couplings=(
            "the concrete olfactory-bulb result semantics now live in olfactorybulb.analysis_data",
            "the concrete olfactory-bulb HFO/LFP overview policy now lives in olfactorybulb.analysis_hfo_views",
            "the concrete olfactory-bulb notebook presentation presets now live in olfactorybulb.analysis_presentations",
            "the concrete olfactory-bulb analysis profile now lives in olfactorybulb.analysis_profile",
            "the concrete olfactory-bulb grouped presentation policy now lives in olfactorybulb.analysis_views",
            "concrete OBGPU signal families like lfp and gc_output_rate now live in explicit olfactorybulb domain modules rather than only the notebook helper",
            "signal semantics still assume this repository's saved result structure",
        ),
        extraction_confidence="medium",
        proposed_phase=3,
        current_status="internal_shim_extracted",
        recommended_action="The generic result-catalog helpers, stable ordered-name helpers, fair round-robin subgroup truncation, ordered group-row flattening with per-bucket limits, grouped row-display policies, grouped stacked-trace and event-raster suites, result-overview builders, shared plotting primitives including stacked labeled traces, named-signal trace, band-pass, PSD overview, time-frequency view helpers, resolved-signal phase-locking summaries, frequency KDE/time-binned sample renderers, result-backed frequency plot families plus family-bound suites, labeled-row and trace-derived frequency sample collectors, label-prefix row filtering, normalization-driven event-rate computation, result-backed event-family specs plus family suites and event plot suites, reusable event-rate series assembly, prepared labeled event display rows, overview-layout derivation from them, plus event-rate and raster-analysis helpers, shared rate plotting and overview layout, spectral analysis core, sweep plot protocol, named sweep plot registries, sweep persistence and animation pipeline, the named-signal provider/registry/view layer, and domain analysis profiles that aggregate concrete repo definitions now live under neuroinfra.analysis, with the current concrete OBGPU result semantics, grouped soma presentation, HFO/LFP overview policy, notebook presentation presets, and profile now assembled in olfactorybulb.analysis_data, olfactorybulb.analysis_views, olfactorybulb.analysis_hfo_views, olfactorybulb.analysis_presentations, and olfactorybulb.analysis_profile; next keep shrinking obgpu_experiment_helpers.py toward notebook entrypoint glue rather than analysis ownership.",
    ),
    ExtractionCandidate(
        key="dashboard_and_packets",
        title="Dashboard and packet runtime",
        target_module="neuroinfra.dashboard",
        source_paths=(
            "neuroinfra/dashboard/packets.py",
            "neuroinfra/dashboard/runtime.py",
            "tools/analysis/hfo_visual_dashboard.py",
            "tools/analysis/generate_hfo_candidate_packet.py",
            "tools/analysis/regenerate_hfo_packet_psd.py",
            "tools/analysis/hfo_tensorboard_dashboard.py",
        ),
        generic_capabilities=(
            "manifest-backed packet generation",
            "runtime supervision",
            "best/recent candidate views",
            "stale packet detection",
            "background packet refresh",
        ),
        repo_specific_couplings=(
            "HFO-specific packet schema",
            "PSD overlays and score summaries",
            "fixed plot families and filenames",
        ),
        extraction_confidence="medium",
        proposed_phase=5,
        current_status="packet_and_runtime_protocol_extracted",
        recommended_action="The generic packet manifest and sidecar/runtime process helpers now live under neuroinfra.dashboard; next separate the remaining HFO-specific command assembly and HTML/server policy from the shared supervision shell.",
    ),
    ExtractionCandidate(
        key="cell_model_registry",
        title="Cell-model registry",
        target_module="neuroinfra.models",
        source_paths=(
            "neuroinfra/models/registry.py",
            "prev_ob_models/cell_registry.py",
            "prev_ob_models/utils.py",
        ),
        generic_capabilities=(
            "discoverable model metadata",
            "family and role resolution",
            "dynamic import and instantiation",
            "default family-role mapping",
        ),
        repo_specific_couplings=(
            "registered model families are olfactory-bulb-specific",
            "role vocabulary is still domain-specific",
        ),
        extraction_confidence="medium-high",
        proposed_phase=3,
        current_status="internal_shim_extracted",
        recommended_action="The generic registry skeleton now lives under neuroinfra.models; next move more concrete catalogs behind provider interfaces instead of direct repo imports.",
    ),
    ExtractionCandidate(
        key="slice_geometry_connectivity",
        title="Slice geometry and connectivity evaluator",
        target_module="neuroinfra.geometry",
        source_paths=(
            "olfactorybulb/slice_connectivity_optimizer.py",
            "tools/optimize_slice_connectivity.py",
        ),
        generic_capabilities=(
            "section and terminal geometry datatypes",
            "offline connectivity scoring",
            "candidate rule evaluation against exported geometry",
        ),
        repo_specific_couplings=(
            "slice JSON schema is repo-specific",
            "section-family assumptions are olfactory-bulb-specific",
            "group and synapse-set naming is domain-specific",
        ),
        extraction_confidence="medium",
        proposed_phase=5,
        current_status="later",
        recommended_action="Do not extract first-wave. Stabilize the exported geometry schema first, then generalize.",
    ),
    ExtractionCandidate(
        key="notebook_helper_surface",
        title="Notebook helper surface",
        target_module="neuroinfra.notebooks",
        source_paths=(
            "neuroinfra/notebooks/runs.py",
            "obgpu_experiment_helpers.py",
        ),
        generic_capabilities=(
            "none in current file as a whole because responsibilities are mixed",
        ),
        repo_specific_couplings=(
            "run config defaults",
            "remote SSH logic",
            "sweep execution",
            "artifact loading",
            "plotting",
            "dashboard glue",
        ),
        extraction_confidence="low",
        proposed_phase=2,
        current_status="blocked_by_refactor",
        recommended_action="The generic notebook run catalog/metadata layer now lives under neuroinfra.notebooks.runs, but the rest of this helper still mixes config defaults, remote execution, result loading, and presentation; keep splitting it by responsibility because it remains the main architectural blocker.",
    ),
)


REPO_SPECIFIC_AREAS: tuple[dict[str, object], ...] = (
    {
        "key": "olfactory_bulb_domain_model",
        "title": "Olfactory-bulb domain model and biology",
        "source_paths": (
            "olfactorybulb/model.py",
            "olfactorybulb/inputs.py",
            "olfactorybulb/epli.py",
            "olfactorybulb/paramsets/base.py",
            "olfactorybulb/paramsets/case_studies.py",
            "olfactorybulb/paramsets/sensitivity.py",
            "olfactorybulb/slicebuilder/blender.py",
            "olfactorybulb/slicebuilder/nrn.py",
            "prev_ob_models/Birgiolas2020/isolated_cells.py",
        ),
        "reason": "These files are the science application and should remain the first domain plugin rather than move into the reusable framework.",
    },
)


def target_module_index() -> dict[str, list[ExtractionCandidate]]:
    grouped: dict[str, list[ExtractionCandidate]] = {}
    for candidate in EXTRACTION_CANDIDATES:
        grouped.setdefault(candidate.target_module, []).append(candidate)
    return grouped


def repo_specific_areas() -> tuple[dict[str, object], ...]:
    return REPO_SPECIFIC_AREAS


def inventory_rows() -> list[dict[str, object]]:
    return [candidate.to_dict() for candidate in EXTRACTION_CANDIDATES]


def _text_summary(candidates: Iterable[ExtractionCandidate]) -> str:
    lines = ["Neuroinfra extraction candidates", "==============================", ""]
    for candidate in sorted(candidates, key=lambda item: (item.proposed_phase, item.target_module, item.key)):
        lines.append(f"- {candidate.target_module} :: {candidate.title}")
        lines.append(f"  phase={candidate.proposed_phase} confidence={candidate.extraction_confidence} status={candidate.current_status}")
        lines.append(f"  sources={', '.join(candidate.source_paths)}")
        lines.append(f"  action={candidate.recommended_action}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the extraction inventory as JSON.")
    args = parser.parse_args(argv)

    if args.json:
        print(
            json.dumps(
                {
                    "candidates": inventory_rows(),
                    "repo_specific_areas": list(REPO_SPECIFIC_AREAS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(_text_summary(EXTRACTION_CANDIDATES), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
