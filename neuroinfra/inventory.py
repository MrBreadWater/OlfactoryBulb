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
            "neuroinfra/analysis/signal_views.py",
            "neuroinfra/analysis/spectral.py",
            "neuroinfra/analysis/sweeps.py",
            "neuroinfra/analysis/signals.py",
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
            "uniform-trace interpolation and modulus folding",
            "shared spectrogram, wavelet, and band-power analysis",
            "named-signal trace, band-pass, PSD overview, spectrogram, wavelet, and band-power view helpers",
            "phase-locking summaries from resolved signals and labeled spike-time rows",
            "frequency KDE and time-binned plotting from precomputed sample arrays",
            "result-backed frequency plot families for 1D KDE, 2D KDE, and time-binned rendering",
            "instantaneous frequency sample collection from labeled event rows",
            "trace-derived instantaneous frequency sample collection from labeled continuous-trace rows",
            "event-frequency conversion, binned event-rate analysis, rate-plot helpers, and raster-plot primitives",
            "row filtering by label-prefix families",
            "normalization-driven event-rate computation from arbitrary event rows",
            "result-backed event-family specs for filtering, frequency sample collection, and normalized rate computation",
            "reusable event-rate series assembly for named subset plots",
            "prepared labeled event display rows and overview-layout derivation from them",
            "shared raster-plus-rate overview layout",
            "sweep plot specification and placeholder rendering",
            "named sweep plot registries with deprecation-aware resolution",
            "sweep metadata persistence and reload",
            "streamed and in-memory GIF rendering for sweep frames",
            "ordered named-signal providers",
            "dynamic signal enumeration",
            "provider-based signal resolution",
            "provider factories for keyed traces, suffix variants, pattern-matched signals, and labeled traces",
            "aligned mean traces from grouped time/value rows",
            "decoupled analysis-signal catalogs from notebook facades",
        ),
        repo_specific_couplings=(
            "concrete OBGPU signal families like lfp and gc_output_rate still live in obgpu_experiment_helpers.py",
            "signal semantics still assume this repository's saved result structure",
        ),
        extraction_confidence="medium",
        proposed_phase=3,
        current_status="internal_shim_extracted",
        recommended_action="The generic result-catalog helpers, stable ordered-name helpers, fair round-robin subgroup truncation, ordered group-row flattening with per-bucket limits, result-overview builders, shared plotting primitives including stacked labeled traces, named-signal trace, band-pass, PSD overview, time-frequency view helpers, resolved-signal phase-locking summaries, frequency KDE/time-binned sample renderers, result-backed frequency plot families, labeled-row and trace-derived frequency sample collectors, label-prefix row filtering, normalization-driven event-rate computation, result-backed event-family specs, reusable event-rate series assembly, prepared labeled event display rows, overview-layout derivation from them, plus event-rate and raster-analysis helpers, shared rate plotting and overview layout, spectral analysis core, sweep plot protocol, named sweep plot registries, sweep persistence and animation pipeline, and the named-signal provider registry and provider-factory layer now live under neuroinfra.analysis; next move more concrete signal families and notebook presentation and domain-specific plot composition out of obgpu_experiment_helpers.py and behind provider-style interfaces.",
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
        source_paths=("obgpu_experiment_helpers.py",),
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
        recommended_action="Split this file by responsibility before extraction; it is the main architectural blocker.",
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
