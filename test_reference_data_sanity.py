"""Heuristic sanity checks for generated reference-data bundles.

These checks are intentionally broad and conservative. They are not a substitute
for source-by-source review, but they are meant to catch obvious extraction
mistakes such as unit blow-ups or impossible sign errors.
"""

from __future__ import annotations

import csv
from pathlib import Path


REFERENCE_DIR = Path("research_context")
CSV_FILES = sorted(REFERENCE_DIR.glob("*.csv"))


def _float_or_none(value: str):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


for path in CSV_FILES:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        continue
    for row in rows:
        prop = str(row.get("Property", ""))
        unit = str(row.get("unit", ""))
        mean = _float_or_none(row.get("mean", ""))
        sd = _float_or_none(row.get("sd", ""))
        q_low = _float_or_none(row.get("q_low", ""))
        q_high = _float_or_none(row.get("q_high", ""))

        if sd is not None:
            assert sd >= 0.0, (path.name, prop, "negative sd", sd)
        if q_low is not None and q_high is not None:
            assert q_low <= q_high, (path.name, prop, "quantiles reversed", q_low, q_high)

        if mean is None:
            continue
        if unit == "mV":
            assert abs(mean) <= 200.0, (path.name, prop, unit, mean)
        elif unit == "Hz":
            assert abs(mean) <= 1000.0, (path.name, prop, unit, mean)
        elif unit == "ms":
            assert abs(mean) <= 10000.0, (path.name, prop, unit, mean)
        elif unit == "pA":
            assert abs(mean) <= 5000.0, (path.name, prop, unit, mean)
        elif unit == "MOhm":
            assert abs(mean) <= 5000.0, (path.name, prop, unit, mean)
        elif unit == "pF":
            assert abs(mean) <= 5000.0, (path.name, prop, unit, mean)
        elif unit == "Hz/nA":
            assert abs(mean) <= 10000.0, (path.name, prop, unit, mean)

        if "Coefficient of Variation" in prop:
            assert mean >= 0.0, (path.name, prop, "negative coefficient of variation", mean)
        if "Probability" in prop:
            assert 0.0 <= mean <= 1.0, (path.name, prop, "probability outside [0,1]", mean)
        if "Capacitance" in prop:
            assert mean >= 0.0, (path.name, prop, "negative capacitance", mean)
        if "Resistance" in prop:
            assert mean >= 0.0, (path.name, prop, "negative resistance", mean)
        if "Time Constant" in prop:
            assert mean >= 0.0, (path.name, prop, "negative time constant", mean)
        if "Rheobase" in prop:
            assert mean >= 0.0, (path.name, prop, "negative rheobase", mean)


print("reference_data_sanity: OK")
