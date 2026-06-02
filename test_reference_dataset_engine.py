"""Smoke tests for the generic declarative reference-data dataset engine."""

from __future__ import annotations

from pathlib import Path

from olfactorybulb.audit.reference_dataset_config import (
    dataset_config_path,
    dataset_output_path,
    load_dataset_config,
)
from olfactorybulb.audit.reference_dataset_engine import extract_reference_dataset, write_reference_dataset_outputs


template_path = Path("research_context/reference_datasets/TEMPLATE.dataset.toml")
assert template_path.exists(), template_path
template_config = load_dataset_config(path=template_path)
assert template_config["dataset_id"] == "example_dataset"
assert template_config["outputs"]["ephys"] == "EXAMPLE_ephys.csv"
assert template_config["sources"][0]["source_id"] == "example_primary_table"

config_path = dataset_config_path("pv_crh_epl_fsi")
assert config_path.exists(), config_path
config = load_dataset_config(dataset_id="pv_crh_epl_fsi")
assert config["dataset_id"] == "pv_crh_epl_fsi"
assert config["source_data_subdir"] == "epl_fsi"
assert len(config["sources"]) >= 6

result = extract_reference_dataset(dataset_id="pv_crh_epl_fsi")
assert result["rows"]["ephys"], "expected ephys rows"
assert result["rows"]["fi_curve"], "expected fi_curve rows"
assert result["rows"]["protocols"], "expected protocol rows"
assert result["rows"]["identity"], "expected identity rows"
assert result["rows"]["notes"], "expected note rows"

written = write_reference_dataset_outputs(dataset_id="pv_crh_epl_fsi")
for output_key in ("ephys", "fi_curve", "protocols", "identity", "notes", "manual", "readme"):
    assert dataset_output_path(config, output_key).exists(), output_key

assert written["rows"]["fi_curve"][0]["protocol_id"] == "BMU2024_EPL_FSI_500ms_50_600pA_50pA"

gc_config_path = dataset_config_path("granule_cells")
assert gc_config_path.exists(), gc_config_path
gc_config = load_dataset_config(dataset_id="granule_cells")
assert gc_config["dataset_id"] == "granule_cells"
assert gc_config["source_data_subdir"] == "granule_cells"
assert len(gc_config["sources"]) >= 7

gc_result = extract_reference_dataset(dataset_id="granule_cells")
assert gc_result["rows"]["ephys"], "expected GC ephys rows"
assert gc_result["rows"]["subtype_ephys"], "expected GC subtype ephys rows"
assert gc_result["rows"]["protocols"], "expected GC protocol rows"
assert gc_result["rows"]["identity"], "expected GC identity rows"
assert gc_result["rows"]["modulation"], "expected GC modulation rows"
assert gc_result["rows"]["synaptic_latency"], "expected GC latency rows"
assert gc_result["rows"]["notes"], "expected GC note rows"

gc_written = write_reference_dataset_outputs(dataset_id="granule_cells")
for output_key in (
    "ephys",
    "fi_curve",
    "subtype_ephys",
    "subtype_fi_curve",
    "protocols",
    "identity",
    "synaptic_latency",
    "modulation",
    "notes",
    "manual",
    "readme",
):
    assert dataset_output_path(gc_config, output_key).exists(), output_key

assert gc_written["rows"]["protocols"][0]["protocol_id"] == "BU2014_MC_TC_2s_0_300pA_50pA"

print("reference_dataset_engine: OK")
