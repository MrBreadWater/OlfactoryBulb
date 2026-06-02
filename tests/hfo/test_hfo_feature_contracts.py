"""Regression checks for centralized HFO feature and visual contracts."""

from __future__ import annotations

import argparse

from olfactorybulb.audit.hfo_feature_contracts import run
from olfactorybulb.hfo_features import (
    PARAMETER_CONTRACT_VERSION,
    default_hfo_search_space,
    hfo_run_config_defaults,
    parameter_contract_snapshot,
)
import olfactorybulb.hfo_visuals as hfo_visuals
import tools.analysis.generate_hfo_candidate_packet as packet_script
import tools.analysis.hfo_visual_dashboard as dashboard
import tools.analysis.regenerate_hfo_packet_psd as psd_script


report = run(argparse.Namespace())
assert report.worst_status == "PASS", report.to_json()

contract = parameter_contract_snapshot(search_space=default_hfo_search_space())
assert contract["version"] == PARAMETER_CONTRACT_VERSION
assert contract["search_space_paths"] == [spec.path for spec in default_hfo_search_space()]
assert set(contract["runtime_parameter_keys"]) == set(hfo_run_config_defaults())

packet_files = hfo_visuals.packet_manifest_files()
assert hfo_visuals.kde_filename("1d", "control", "MT") in packet_files
assert hfo_visuals.kde_filename("1d", "control", "EPLI") in packet_files
assert hfo_visuals.kde_filename("2d", "ketamine", "GC") in packet_files

assert packet_script.VISUAL_STYLE_VERSION == hfo_visuals.VISUAL_STYLE_VERSION
assert packet_script.SPECTROGRAM_FILE_CONTROL == hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION["control"]
assert packet_script.SPECTROGRAM_FILE_KETAMINE == hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION["ketamine"]
assert psd_script.PSD_PACKET_RENDER_VERSION == hfo_visuals.PSD_PACKET_RENDER_VERSION
assert dashboard.EXPECTED_SPECTROGRAM_FILES == dict(hfo_visuals.SPECTROGRAM_FILE_BY_CONDITION)
assert tuple(dashboard.PRIMARY_PSD_NAME_ORDER) == tuple(hfo_visuals.PRIMARY_PSD_NAME_ORDER)
