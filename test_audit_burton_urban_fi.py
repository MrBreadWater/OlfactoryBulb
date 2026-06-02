"""Smoke tests for the Burton & Urban f-I validation audit.

Run with:
    python test_audit_burton_urban_fi.py
"""

from __future__ import annotations

import json
import subprocess
import sys

from olfactorybulb.audit.burton_urban_fi import (
    BURTON_CSV_REFERENCES,
    BurtonUrbanProtocol,
    _resolved_jobs,
    build_validation_items,
    find_spike_times_milliseconds,
    summarize_metrics,
)


fixture_metrics = [
    {
        "cell_name": "MC1",
        "cell_type": "MC",
        "resting_potential_mV": -66.0,
        "bias_current_pA": 100.0,
        "zero_step_rate_Hz": 0.0,
        "membrane_time_constant_ms": 21.3,
        "cell_capacitance_pF": 236.4,
        "sag_amplitude_mV": -2.0,
        "rebound_potential_presence": 0.0,
        "rheobase_pA": 100.0,
        "spike_latency_ms": 500.0,
        "peak_rate_Hz": 60.0,
        "fi_gain_Hz_per_50pA": 10.0,
        "spike_accommodation_hz": -9.0,
        "spike_accommodation_time_constant_ms": 398.0,
        "cv_isi": 0.4,
        "cv_isi_step_pA": 150.0,
        "cv_isi_mean_rate_Hz": 20.0,
        "input_resistance_MOhm": 100.0,
        "AP_onset_mV": -42.0,
        "Amplitude_mV": 76.0,
        "FWHM_ms": 1.1,
        "Rise_slope_mV_per_ms": 240.0,
        "Fall_slope_mV_per_ms": -70.0,
        "AHP_amplitude_mV": 15.0,
        "T_AHP50_ms": 60.0,
    },
    {
        "cell_name": "TC1",
        "cell_type": "TC",
        "resting_potential_mV": -64.0,
        "bias_current_pA": 150.0,
        "zero_step_rate_Hz": 0.0,
        "membrane_time_constant_ms": 18.8,
        "cell_capacitance_pF": 188.8,
        "sag_amplitude_mV": -4.4,
        "rebound_potential_presence": 1.0,
        "rheobase_pA": 90.0,
        "spike_latency_ms": 400.0,
        "peak_rate_Hz": 120.0,
        "fi_gain_Hz_per_50pA": 20.0,
        "spike_accommodation_hz": -20.0,
        "spike_accommodation_time_constant_ms": 585.0,
        "cv_isi": 0.8,
        "cv_isi_step_pA": 100.0,
        "cv_isi_mean_rate_Hz": 20.0,
        "input_resistance_MOhm": 110.0,
        "AP_onset_mV": -42.5,
        "Amplitude_mV": 72.0,
        "FWHM_ms": 0.9,
        "Rise_slope_mV_per_ms": 200.0,
        "Fall_slope_mV_per_ms": -90.0,
        "AHP_amplitude_mV": 17.0,
        "T_AHP50_ms": 20.0,
    },
]

summary = summarize_metrics(fixture_metrics)
assert summary["MC"]["fi_gain_Hz_per_50pA"] == 10.0
assert summary["TC"]["peak_rate_Hz"] == 120.0

items = build_validation_items(fixture_metrics, BurtonUrbanProtocol())
item_by_id = {item.check_id: item for item in items}
assert item_by_id["uploaded_burton_reference_coverage"].status == "PASS"
assert item_by_id["fi_protocol_caveats"].status == "WARN"
assert item_by_id["tc_fi_gain_higher"].status == "PASS"
assert item_by_id["rheobase_in_paper_regime"].status == "PASS"
assert item_by_id["tc_cv_isi_higher"].status == "PASS"
assert item_by_id["mc_membrane_time_constant_ms_within_uploaded_reference_band"].status == "PASS"
assert "two standard deviations" in item_by_id["mc_membrane_time_constant_ms_within_uploaded_reference_band"].acceptable_basis
assert item_by_id["mc_cv_isi_within_uploaded_reference_band"].evidence["accepted_low"] > 0.0
assert item_by_id["mc_cv_isi_within_uploaded_reference_band"].evidence["accepted_interval_standard"] == "lognormal-reconstructed dispersion band"
assert "not a formal confidence interval" in item_by_id["mc_cv_isi_within_uploaded_reference_band"].acceptable_basis
assert "N_FI_PROTOCOL_DIFFERENCE" in item_by_id["fi_protocol_caveats"].evidence["note_ids"]
assert callable(find_spike_times_milliseconds)
assert _resolved_jobs(10, 0, use_gpu=False) >= 1
assert _resolved_jobs(10, 99, use_gpu=False) == 10
assert _resolved_jobs(10, 8, use_gpu=True) == 1

tight_reference = BURTON_CSV_REFERENCES["MC"]["membrane_time_constant_ms"]
custom_metrics = [dict(metric) for metric in fixture_metrics]
custom_metrics[0]["membrane_time_constant_ms"] = tight_reference.mean + 1.5 * tight_reference.std
default_band_items = {
    item.check_id: item
    for item in build_validation_items(custom_metrics, BurtonUrbanProtocol(), reference_sigma_multiplier=2.0)
}
one_sigma_items = {
    item.check_id: item
    for item in build_validation_items(custom_metrics, BurtonUrbanProtocol(), reference_sigma_multiplier=1.0)
}
assert default_band_items["mc_membrane_time_constant_ms_within_uploaded_reference_band"].status == "PASS"
assert one_sigma_items["mc_membrane_time_constant_ms_within_uploaded_reference_band"].status == "FAIL"

skip = subprocess.run(
    [sys.executable, "tools/audit_burton_urban_fi.py", "--skip-neuron", "--jobs", "4", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert skip.returncode == 0, skip
payload = json.loads(skip.stdout)
assert payload["audit_id"] == "burton_urban_fi"
item_by_id_json = {item["check_id"]: item for item in payload["items"]}
assert payload["summary"]["WARN"] == 2
assert payload["summary"]["PASS"] >= 4
assert item_by_id_json["baseline_slice_population_counts"]["status"] == "PASS"
assert item_by_id_json["requested_birgiolas_models_registered"]["status"] == "PASS"
assert item_by_id_json["birgiolas_model_morphology_skipped"]["status"] == "WARN"
assert item_by_id_json["burton_urban_fi_skipped"]["evidence"]["jobs"] == 4
assert item_by_id_json["burton_urban_fi_skipped"]["evidence"]["reference_sigma_multiplier"] == 2.0

generic = subprocess.run(
    [sys.executable, "tools/run_audit.py", "burton_urban_fi", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert generic.returncode == 0, generic
assert json.loads(generic.stdout)["audit_id"] == "burton_urban_fi"


def _assert_new_sweep_payload(run: subprocess.CompletedProcess[str]) -> None:
    assert run.returncode in (0, 1), run
    payload = json.loads(run.stdout)
    assert payload["audit_id"] == "new_sweep"
    assert payload["title"] == "New sweep"
    check_ids = {item["check_id"] for item in payload["items"]}
    assert any(check_id.startswith("env_install.") for check_id in check_ids)
    assert "burton_urban_fi.burton_urban_fi_skipped" in check_ids
    assert "burton_urban_fi.baseline_slice_population_counts" in check_ids
    assert "burton_urban_fi.requested_birgiolas_models_registered" in check_ids
    assert any(check_id.startswith("epli_correctness.") for check_id in check_ids)
    expected_code = 1 if payload["summary"]["FAIL"] else 0
    assert run.returncode == expected_code


default_new_sweep = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
_assert_new_sweep_payload(default_new_sweep)

explicit_all = subprocess.run(
    [sys.executable, "tools/run_audit.py", "all", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
_assert_new_sweep_payload(explicit_all)

explicit_new_sweep = subprocess.run(
    [sys.executable, "tools/run_audit.py", "new_sweep", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
_assert_new_sweep_payload(explicit_new_sweep)

listed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--list"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed.returncode == 0, listed
assert "burton_urban_fi" in listed.stdout

print("audit_burton_urban_fi smoke test: OK")
