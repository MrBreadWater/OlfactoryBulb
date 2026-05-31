"""Audit that HFO feature and visualization contracts stay centralized."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _configure_parent_cache_dirs() -> None:
    cache_root = Path("/tmp") / f"olfactorybulb-audit-cache-{os.getuid()}"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_cache = cache_root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))


_configure_parent_cache_dirs()

import obgpu_experiment_helpers as hlp
from olfactorybulb.audit.core import AuditItem, AuditReport, collect_items
from olfactorybulb.hfo_features import (
    default_hfo_search_space,
    hfo_control_help,
    hfo_run_config_defaults,
    parameter_contract_snapshot,
)
import olfactorybulb.hfo_visuals as hfo_visuals
import tools.analysis.hfo_visual_dashboard as hfo_dashboard


def _contract_item(
    *,
    check_id: str,
    status: str,
    title: str,
    criterion: str,
    description: str,
    acceptable: str,
    acceptable_basis: str,
    evidence: dict[str, object] | None = None,
    note: str = "",
) -> AuditItem:
    return AuditItem(
        check_id=check_id,
        status=status,
        title=title,
        criterion=criterion,
        description=description,
        acceptable=acceptable,
        acceptable_basis=acceptable_basis,
        evidence=evidence or {},
        note=note,
    )


def audit_parameter_contracts() -> list[AuditItem]:
    items: list[AuditItem] = []
    search_space = default_hfo_search_space()
    search_paths = [spec.path for spec in search_space]
    unique_paths = list(dict.fromkeys(search_paths))
    items.append(
        _contract_item(
            check_id="hfo_search_space_unique_paths",
            status="PASS" if search_paths and len(unique_paths) == len(search_paths) else "FAIL",
            title="HFO search-space paths are unique",
            criterion="The central HFO registry should define one unique search-space path per optimizer dimension.",
            description="This check verifies that the optimizer search space is not accidentally tuning the same underlying configuration path through two different dimension entries.",
            acceptable="Every registered search-space path appears exactly once. The total path count must equal the unique path count.",
            acceptable_basis="The rule comes directly from the canonical search-space registry returned by default_hfo_search_space(). Duplicate paths would create ambiguous optimizer behavior after the registry refactor.",
            evidence={"count": len(search_paths), "unique_count": len(unique_paths)},
        )
    )

    contract = parameter_contract_snapshot(search_space=search_space)
    items.append(
        _contract_item(
            check_id="hfo_parameter_contract_matches_search_space",
            status="PASS" if contract.get("search_space_paths") == search_paths else "FAIL",
            title="Parameter contract snapshot matches the central search space",
            criterion="The packet/dashboard parameter contract must derive from the same search-space registry used by the optimizer.",
            description="This check confirms that packet manifests and dashboard parameter tables are snapshotting the exact same ordered parameter list that drives optimization.",
            acceptable="The parameter contract snapshot must reproduce the search-space path list exactly, in the same order.",
            acceptable_basis="The rule is a direct equality comparison between parameter_contract_snapshot(...) and default_hfo_search_space(). If those diverge, the UI can silently omit or reorder active optimizer knobs.",
            evidence={"contract_paths": contract.get("search_space_paths", []), "search_space_paths": search_paths},
        )
    )

    defaults = hfo_run_config_defaults()
    config = hlp.build_run_config()
    missing_defaults = sorted(key for key in defaults if key not in config)
    items.append(
        _contract_item(
            check_id="hfo_helper_defaults_cover_registry",
            status="PASS" if not missing_defaults else "FAIL",
            title="Notebook run-config defaults cover the central HFO registry",
            criterion="Every registered HFO runtime knob should be present in build_run_config defaults.",
            description="This check validates that notebook-facing run configuration helpers expose every registered HFO knob, so the live notebook path cannot silently miss a newly added parameter.",
            acceptable="No registered default key is missing from build_run_config(). The missing-key list must be empty.",
            acceptable_basis="The rule compares hfo_run_config_defaults() against the actual build_run_config() output because that notebook helper is the maintained user-facing entrypoint.",
            evidence={"missing_keys": missing_defaults},
        )
    )

    control_catalog = hlp.available_controls()
    missing_help = sorted(key for key in hfo_control_help() if key not in control_catalog)
    mismatched_help = sorted(
        key for key, value in hfo_control_help().items() if control_catalog.get(key) != value
    )
    items.append(
        _contract_item(
            check_id="hfo_helper_help_matches_registry",
            status="PASS" if not missing_help and not mismatched_help else "FAIL",
            title="Notebook control help mirrors the central HFO registry",
            criterion="Every registered HFO runtime knob should expose the same help text through available_controls.",
            description="This check ensures the notebook control catalog does not drift away from the centralized HFO registry descriptions that explain each tunable parameter to the user.",
            acceptable="There are no missing help entries and no mismatched help strings between the registry and available_controls().",
            acceptable_basis="The rule comes from exact key and value comparisons between hfo_control_help() and the control catalog returned by obgpu_experiment_helpers.available_controls().",
            evidence={"missing_keys": missing_help, "mismatched_keys": mismatched_help},
        )
    )

    probe_overrides = hlp.build_param_overrides(
        hlp.build_run_config(
            input_syn_tau1_ms=7.0,
            gap_tc=13.0,
            ampa_nmda_gmax=63.0,
            gaba_tau2_ms=105.0,
            kar_mt_gmax=0.02,
            enable_gc_kar=True,
        )
    )
    probe_ok = (
        probe_overrides.get("input_syn_tau1") == 7.0
        and probe_overrides.get("gap_juction_gmax", {}).get("TC") == 13.0
        and probe_overrides.get("synapse_properties", {}).get("AmpaNmdaSyn", {}).get("gmax") == 63.0
        and probe_overrides.get("synapse_properties", {}).get("GabaSyn", {}).get("tau2") == 105.0
        and probe_overrides.get("kar_mt_gmax") == 0.02
        and probe_overrides.get("enable_gc_kar") is True
    )
    items.append(
        _contract_item(
            check_id="hfo_runtime_override_wiring",
            status="PASS" if probe_ok else "FAIL",
            title="Representative HFO runtime overrides still reach the benchmark override payload",
            criterion="Centralized HFO registry wiring must still drive the same benchmark override keys as before the refactor.",
            description="This check exercises representative conductance, time-constant, gap-junction, and boolean knobs to make sure the centralized registry still maps them into the benchmark override payload correctly.",
            acceptable="Each probe value lands at the expected override key and value in the generated benchmark payload.",
            acceptable_basis="The acceptance rule is a direct probe against build_param_overrides(build_run_config(...)) using representative HFO knobs that previously lived in decentralized wiring code.",
            evidence={"probe_overrides": probe_overrides},
        )
    )
    return items


def audit_visual_contracts() -> list[AuditItem]:
    items: list[AuditItem] = []
    packet_files = hfo_visuals.packet_manifest_files()
    required_files = {
        hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION["control"],
        hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION["ketamine"],
        hfo_visuals.kde_filename("1d", "control", "MT"),
        hfo_visuals.kde_filename("1d", "control", "EPLI"),
        hfo_visuals.kde_filename("2d", "ketamine", "MT"),
        hfo_visuals.kde_filename("2d", "ketamine", "EPLI"),
    }
    missing_files = sorted(required_files.difference(packet_files))
    items.append(
        _contract_item(
            check_id="hfo_packet_manifest_files_cover_visual_contract",
            status="PASS" if not missing_files else "FAIL",
            title="Packet manifest file list covers the shared visual contract",
            criterion="Shared packet file enumeration should include spectrograms plus separate MT/EPLI frequency KDE artifacts.",
            description="This check verifies that packet manifests enumerate the core visualization artifacts expected by the dashboard, including separate mitral-cell and external plexiform layer interneuron spike-density plots.",
            acceptable="All required spectrogram and KDE filenames are present in the shared packet manifest file list. The missing-file list must be empty.",
            acceptable_basis="The acceptance rule comes from the canonical packet artifact contract in hfo_visuals.packet_manifest_files() and the currently supported spectrogram/KDE filenames.",
            evidence={"missing_files": missing_files},
        )
    )

    spectrogram_contract_ok = (
        dict(hfo_dashboard.EXPECTED_SPECTROGRAM_FILES) == dict(hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION)
        and str(hfo_dashboard.EXPECTED_SPECTROGRAM_PIPELINE) == str(hfo_visuals.SPECTROGRAM_PIPELINE["generator"])
        and tuple(hfo_dashboard.PRIMARY_PSD_NAME_ORDER) == tuple(hfo_visuals.PRIMARY_PSD_NAME_ORDER)
    )
    items.append(
        _contract_item(
            check_id="hfo_dashboard_visual_contract_alignment",
            status="PASS" if spectrogram_contract_ok else "FAIL",
            title="Dashboard visual expectations derive from the shared visual contract",
            criterion="Dashboard refresh checks should agree with the packet generator about filenames, PSD priority, and spectrogram provenance.",
            description="This check guards against the dashboard and packet generator drifting apart on which spectrogram files are authoritative, which PSD overlay to prioritize, and which pipeline produced the plots.",
            acceptable="Dashboard constants and the shared visual contract must match exactly for spectrogram filenames, primary PSD order, and the spectrogram generator identity.",
            acceptable_basis="The rule compares the dashboard module constants against the centralized hfo_visuals contract because those values must stay mechanically linked after the refactor.",
            evidence={
                "dashboard_spectrogram_files": dict(hfo_dashboard.EXPECTED_SPECTROGRAM_FILES),
                "contract_spectrogram_files": dict(hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION),
                "dashboard_psd_order": list(hfo_dashboard.PRIMARY_PSD_NAME_ORDER),
                "contract_psd_order": list(hfo_visuals.PRIMARY_PSD_NAME_ORDER),
            },
        )
    )

    tab_keys = [tab.key for tab in hfo_visuals.dashboard_tabs()]
    items.append(
        _contract_item(
            check_id="hfo_dashboard_tabs_registered",
            status="PASS" if tab_keys == ["best", "recent"] else "FAIL",
            title="Dashboard tabs are registered through the shared visual contract",
            criterion="Dashboard tab identity should live in one shared contract instead of being hardcoded in multiple render paths.",
            description="This check confirms that the live dashboard navigation is driven by the shared tab contract rather than by ad hoc hardcoded tab names in separate render code paths.",
            acceptable="The shared tab registry exposes the expected tab sequence: best, then recent.",
            acceptable_basis="The current maintained dashboard design expects those two tabs. The rule compares the rendered registry order against that canonical sequence to catch drift quickly.",
            evidence={"tab_keys": tab_keys},
        )
    )
    return items


def configure_parser(parser: argparse.ArgumentParser) -> None:
    del parser


def run(args: argparse.Namespace) -> AuditReport:
    del args
    items = collect_items(audit_parameter_contracts(), audit_visual_contracts())
    return AuditReport(
        audit_id="hfo_feature_contracts",
        title="HFO feature/visual contract audit",
        items=items,
    )
