"""Reusable LaTeX criterion builders for audit items."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import re


@dataclass(frozen=True)
class CriterionMath:
    """Rendered criterion math plus the definitions needed to read it."""

    latex: str
    definitions: list[dict[str, Any]] = field(default_factory=list)
    formulae: list[str] = field(default_factory=list)


def latex_number(value: float) -> str:
    return f"{float(value):g}"


def latex_token(value: str, *, fallback: str = "x") -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", str(value or "").strip())
    return token or fallback


def group_mean_symbol(group: str) -> str:
    return rf"\bar{{x}}_{{\mathrm{{{latex_token(group, fallback='g')}}}}}"


_PROPERTY_OBSERVED_SYMBOLS: dict[str, str] = {
    "ISI Coefficient of Variation": r"\overline{\mathrm{CV}}_{\mathrm{ISI}}",
    "Rheobase Current": r"\bar{I}_{\mathrm{rh}}",
    "Input Resistance": r"\bar{R}_{\mathrm{in}}",
    "Membrane Time Constant": r"\bar{\tau}_m",
    "Capacitance": r"\bar{C}_m",
    "Membrane Resting Voltage": r"\bar{V}_{\mathrm{rest}}",
    "AP Threshold": r"\bar{V}_{\mathrm{th}}",
    "AP Amplitude": r"\bar{A}_{\mathrm{AP}}",
    "AP Half-Width": r"\overline{\mathrm{FWHM}}",
    "AP Width at Half-height": r"\overline{\mathrm{FWHM}}",
    "AP Rising Slope": r"\bar{s}_{\mathrm{rise}}",
    "AP Falling Slope": r"\bar{s}_{\mathrm{fall}}",
    "First Spike Latency": r"\bar{t}_{\mathrm{lat}}",
    "FI Curve Slope": r"\bar{g}_{\mathrm{FI}}",
    "Peak Instantaneous Rate": r"\bar{f}_{\max}",
    "Max FI Rate": r"\bar{f}_{\max}",
    "Spontaneous Firing Rate": r"\bar{f}_{\mathrm{spont}}",
    "AHP Amplitude": r"\bar{A}_{\mathrm{AHP}}",
    "AHP Duration": r"\bar{t}_{\mathrm{AHP50}}",
    "Spiking Rate Accommodation": r"\overline{\mathrm{SRA}}",
    "Spiking Rate Accom. Time Constant": r"\bar{\tau}_{\mathrm{SRA}}",
}


def observed_symbol_for_property(property_name: str) -> str:
    return _PROPERTY_OBSERVED_SYMBOLS.get(str(property_name or "").strip(), r"\bar{x}")


def criterion_math_for_reference_band(
    group: str,
    property_name: str,
    band: Any,
) -> CriterionMath:
    observed_label = f"{group} mean {property_name.lower()}"
    observed_symbol = observed_symbol_for_property(property_name)
    sigma_multiplier = latex_number(float(band.sigma_multiplier))
    formulae: list[str] = []
    definitions: list[dict[str, Any]] = [{"symbol": observed_symbol, "definition": observed_label}]
    if band.mode == "quantile_interval":
        latex = rf"L \leq {observed_symbol} \leq U"
        definitions.extend(
            [
                {"symbol": "L", "definition": "reported lower quantile bound"},
                {"symbol": "U", "definition": "reported upper quantile bound"},
                {"symbol": r"q_{\mathrm{low}}", "definition": "reported lower quantile"},
                {"symbol": r"q_{\mathrm{high}}", "definition": "reported upper quantile"},
            ]
        )
        formulae.extend([r"L = q_{\mathrm{low}}", r"U = q_{\mathrm{high}}"])
    elif band.mode == "beta_sd":
        latex = rf"L \leq {observed_symbol} \leq U"
        definitions.extend(
            [
                {
                    "symbol": "L",
                    "definition": "lower beta-distribution quantile reconstructed from the uploaded mean and standard deviation",
                },
                {
                    "symbol": "U",
                    "definition": "upper beta-distribution quantile reconstructed from the uploaded mean and standard deviation",
                },
                {"symbol": "q", "definition": "tail probability matched to the configured sigma multiplier"},
                {"symbol": r"\alpha", "definition": "beta-shape parameter"},
                {"symbol": r"\beta", "definition": "beta-shape parameter"},
                {"symbol": r"\kappa", "definition": "beta concentration parameter"},
                {"symbol": r"\mu", "definition": "uploaded reference mean"},
                {"symbol": r"\sigma", "definition": "uploaded reference standard deviation"},
            ]
        )
        formulae.extend(
            [
                r"\kappa = \frac{\mu (1 - \mu)}{\sigma^2} - 1",
                r"\alpha = \mu \kappa",
                r"\beta = (1 - \mu)\kappa",
                r"L = Q_{\mathrm{Beta}}(q;\alpha,\beta)",
                r"U = Q_{\mathrm{Beta}}(1 - q;\alpha,\beta)",
            ]
        )
    elif band.mode == "binary_indicator":
        latex = rf"{observed_symbol} = b"
        definitions.append({"symbol": "b", "definition": "uploaded binary reference indicator"})
    elif band.mode == "lognormal_sd":
        reference_mean_symbol = r"\mu_{\mathrm{ref}}"
        reference_sd_symbol = r"\sigma_{\mathrm{ref}}"
        log_mean_symbol = r"\mu_{\log}"
        log_sd_symbol = r"\sigma_{\log}"
        latex = (
            rf"\left|\ln\!\left({observed_symbol}\right) - {log_mean_symbol}\right| "
            rf"\leq {sigma_multiplier}{log_sd_symbol}"
        )
        definitions.extend(
            [
                {"symbol": reference_mean_symbol, "definition": "uploaded arithmetic reference mean"},
                {"symbol": reference_sd_symbol, "definition": "uploaded arithmetic reference standard deviation"},
                {"symbol": log_mean_symbol, "definition": "reconstructed log-space mean"},
                {"symbol": log_sd_symbol, "definition": "reconstructed log-space standard deviation"},
            ]
        )
        formulae.extend(
            [
                rf"{log_mean_symbol} = \ln({reference_mean_symbol}) - \frac{{1}}{{2}}{log_sd_symbol}^2",
                rf"{log_sd_symbol} = \sqrt{{\ln\!\left(1 + \left(\frac{{{reference_sd_symbol}}}{{{reference_mean_symbol}}}\right)^2\right)}}",
            ]
        )
    else:
        reference_mean_symbol = r"\mu_{\mathrm{ref}}"
        reference_sd_symbol = r"\sigma_{\mathrm{ref}}"
        latex = (
            rf"\left|{observed_symbol} - {reference_mean_symbol}\right| "
            rf"\leq {sigma_multiplier}{reference_sd_symbol}"
        )
        definitions.extend(
            [
                {"symbol": reference_mean_symbol, "definition": "uploaded reference mean"},
                {"symbol": reference_sd_symbol, "definition": "uploaded reference standard deviation"},
            ]
        )
    return CriterionMath(latex=latex, definitions=definitions, formulae=formulae)


def criterion_math_for_exact_metric(metric_key: str) -> CriterionMath:
    return CriterionMath(
        latex=r"\forall i,\ \left|x_i - c\right| \leq \epsilon",
        definitions=[
            {"symbol": r"x_i", "definition": f"{metric_key} value for each audited row"},
            {"symbol": "c", "definition": "expected value"},
            {"symbol": r"\epsilon", "definition": "tolerance"},
        ],
    )


def criterion_math_for_absolute_difference(
    left_symbol: str,
    right_symbol: str,
    max_difference: float,
    *,
    definitions: list[dict[str, Any]] | None = None,
) -> CriterionMath:
    return CriterionMath(
        latex=rf"\left|{right_symbol} - {left_symbol}\right| \leq {latex_number(max_difference)}",
        definitions=list(definitions or []),
    )


def criterion_math_for_ordering(
    left_symbol: str,
    operator: str,
    right_symbol: str,
    *,
    definitions: list[dict[str, Any]] | None = None,
) -> CriterionMath:
    return CriterionMath(
        latex=rf"{left_symbol} {operator} {right_symbol}",
        definitions=list(definitions or []),
    )


def criterion_math_for_lower_bound(
    observed_symbol: str,
    minimum: float,
    *,
    definitions: list[dict[str, Any]] | None = None,
) -> CriterionMath:
    return CriterionMath(
        latex=rf"{observed_symbol} \geq {latex_number(minimum)}",
        definitions=list(definitions or []),
    )


def criterion_math_for_upper_bound(
    observed_symbol: str,
    maximum: float,
    *,
    definitions: list[dict[str, Any]] | None = None,
) -> CriterionMath:
    return CriterionMath(
        latex=rf"{observed_symbol} \leq {latex_number(maximum)}",
        definitions=list(definitions or []),
    )


def criterion_math_for_closed_range(
    observed_symbol: str,
    minimum: float,
    maximum: float,
    *,
    definitions: list[dict[str, Any]] | None = None,
) -> CriterionMath:
    definition_rows = list(definitions or [])
    if math.isfinite(float(minimum)) and math.isfinite(float(maximum)):
        if float(minimum) == float(maximum):
            return CriterionMath(
                latex=rf"{observed_symbol} = {latex_number(minimum)}",
                definitions=definition_rows,
            )
        center = (float(minimum) + float(maximum)) / 2.0
        radius = (float(maximum) - float(minimum)) / 2.0
        if center == 0.0:
            return CriterionMath(
                latex=rf"\left|{observed_symbol}\right| \leq {latex_number(radius)}",
                definitions=definition_rows,
            )
        return CriterionMath(
            latex=rf"\left|{observed_symbol} - {latex_number(center)}\right| \leq {latex_number(radius)}",
            definitions=definition_rows,
        )
    if math.isfinite(float(minimum)):
        return criterion_math_for_lower_bound(
            observed_symbol,
            minimum,
            definitions=definition_rows,
        )
    if math.isfinite(float(maximum)):
        return criterion_math_for_upper_bound(
            observed_symbol,
            maximum,
            definitions=definition_rows,
        )
    return CriterionMath(
        latex=rf"{observed_symbol} \in \mathbb{{R}}",
        definitions=definition_rows,
    )
