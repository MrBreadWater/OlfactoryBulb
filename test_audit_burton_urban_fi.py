"""Smoke tests for the Burton & Urban f-I validation audit.

Run with:
    python test_audit_burton_urban_fi.py
"""

from __future__ import annotations

import json
import subprocess
import sys

from olfactorybulb.audit.burton_urban_fi import (
    BurtonUrbanProtocol,
    build_validation_items,
    summarize_metrics,
)


fixture_metrics = [
    {
        "cell_name": "MC1",
        "cell_type": "MC",
        "resting_potential_mV": -66.0,
        "bias_current_pA": 100.0,
        "zero_step_rate_Hz": 0.0,
        "rheobase_pA": 100.0,
        "spike_latency_ms": 500.0,
        "peak_rate_Hz": 60.0,
        "fi_gain_Hz_per_50pA": 10.0,
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
        "rheobase_pA": 90.0,
        "spike_latency_ms": 400.0,
        "peak_rate_Hz": 120.0,
        "fi_gain_Hz_per_50pA": 20.0,
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
assert item_by_id["tc_fi_gain_higher"].status == "PASS"
assert item_by_id["rheobase_in_paper_regime"].status == "PASS"
assert item_by_id["tc_cv_isi_higher"].status == "PASS"

skip = subprocess.run(
    [sys.executable, "tools/audit_burton_urban_fi.py", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert skip.returncode == 0, skip
payload = json.loads(skip.stdout)
assert payload["audit_id"] == "burton_urban_fi"
assert payload["summary"]["WARN"] == 1

generic = subprocess.run(
    [sys.executable, "tools/run_audit.py", "burton_urban_fi", "--skip-neuron", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert generic.returncode == 0, generic
assert json.loads(generic.stdout)["audit_id"] == "burton_urban_fi"

listed = subprocess.run(
    [sys.executable, "tools/run_audit.py", "--list"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed.returncode == 0, listed
assert "burton_urban_fi" in listed.stdout

print("audit_burton_urban_fi smoke test: OK")
