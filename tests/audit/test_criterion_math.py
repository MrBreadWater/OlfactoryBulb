"""Regression tests for reusable audit criterion-math builders."""

from __future__ import annotations

from olfactorybulb.audit import (
    criterion_math_for_absolute_difference,
    criterion_math_for_exact_metric,
    criterion_math_for_reference_band,
)
from olfactorybulb.audit.reference_validation_rules import compute_reference_acceptance_band


lognormal_band = compute_reference_acceptance_band(
    reference_mean=0.45,
    reference_sd=0.29,
    sigma_multiplier=2.0,
    band_mode="lognormal_sd",
    lower_bound=0.0,
)
lognormal_math = criterion_math_for_reference_band(
    "mitral cell",
    "ISI Coefficient of Variation",
    lognormal_band,
)
assert (
    lognormal_math.latex
    == r"\left|\ln\!\left(\overline{\mathrm{CV}}_{\mathrm{ISI}}\right) - \mu_{\log}\right| \leq 2\sigma_{\log}"
)
assert [definition["symbol"] for definition in lognormal_math.definitions] == [
    r"\overline{\mathrm{CV}}_{\mathrm{ISI}}",
    r"\mu_{\mathrm{ref}}",
    r"\sigma_{\mathrm{ref}}",
    r"\mu_{\log}",
    r"\sigma_{\log}",
]

tolerance_math = criterion_math_for_exact_metric("zero_step_rate_Hz")
assert tolerance_math.latex == r"\forall i,\ \left|x_i - c\right| \leq \epsilon"

diff_math = criterion_math_for_absolute_difference(
    r"\bar{x}_{\mathrm{MC}}",
    r"\bar{x}_{\mathrm{TC}}",
    5.0,
)
assert diff_math.latex == r"\left|\bar{x}_{\mathrm{TC}} - \bar{x}_{\mathrm{MC}}\right| \leq 5"
