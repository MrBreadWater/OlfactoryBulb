"""Generic extraction engine for declarative reference-data datasets."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from .reference_data import (
    NEEDS_MANUAL_EXTRACTION_COLUMNS,
    PV_CRH_EPL_FSI_EPHYS_COLUMNS,
    PV_CRH_EPL_FSI_FI_CURVE_COLUMNS,
    PV_CRH_EPL_FSI_IDENTITY_COLUMNS,
    PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS,
    VALIDATION_NOTES_COLUMNS,
)
from .reference_dataset_config import (
    dataset_output_filenames,
    dataset_output_path,
    dataset_section,
    dataset_sources,
    load_dataset_config,
)
from .reference_sources import ensure_reference_sources, local_source_path, source_entry, stable_source_url


OUTPUT_COLUMNS = {
    "ephys": PV_CRH_EPL_FSI_EPHYS_COLUMNS,
    "fi_curve": PV_CRH_EPL_FSI_FI_CURVE_COLUMNS,
    "protocols": PV_CRH_EPL_FSI_PROTOCOLS_COLUMNS,
    "identity": PV_CRH_EPL_FSI_IDENTITY_COLUMNS,
    "notes": VALIDATION_NOTES_COLUMNS,
    "manual": NEEDS_MANUAL_EXTRACTION_COLUMNS,
}


def _bool_csv(value: bool) -> str:
    return "true" if value else "false"


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _clean_label(text: object) -> str:
    return " ".join(str(text).split())


def _format_mean_plus_minus(mean: float | None, spread: float | None) -> str:
    if mean is None or spread is None:
        return ""
    return f"{mean:g} +/- {spread:g}"


def _join_note_ids(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value if str(item).strip())
    return str(value)


def _stats(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        raise ValueError("Cannot compute statistics for an empty value list")
    mean_value = statistics.fmean(values)
    sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
    sem_value = sd_value / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean_value, sd_value, sem_value, len(values)


class TableCache:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def load(self, source_id: str, *, sheet_name: str = "") -> pd.DataFrame:
        key = (source_id, sheet_name)
        if key in self._cache:
            return self._cache[key]

        path = local_source_path(source_id, config=self.config)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xls", ".xlsx"}:
            selected_sheet: int | str = sheet_name if sheet_name else 0
            df = pd.read_excel(path, sheet_name=selected_sheet)
        else:
            raise ValueError(f"Unsupported tabular source format for {path}")

        df.columns = [_clean_label(column) for column in df.columns]
        self._cache[key] = df
        return df


def _apply_row_filters(df: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    filtered = df
    filters: list[dict[str, Any]] = []
    if rule.get("row_filter_column"):
        filters.append({"column": rule.get("row_filter_column"), "equals": rule.get("row_filter_equals")})
    filters.extend(list(rule.get("row_filters", []) or []))

    for spec in filters:
        column = _clean_label(spec.get("column", ""))
        equals = spec.get("equals")
        if not column:
            continue
        if column not in filtered.columns:
            raise KeyError(f"Missing filter column {column!r} in rule {rule.get('property_name', rule.get('source_id'))}")
        series = filtered[column].astype(str).str.strip()
        filtered = filtered[series == str(equals).strip()]
    return filtered


def _build_ephys_row(data: dict[str, Any]) -> dict[str, object]:
    mean = data.get("mean")
    sd = data.get("sd")
    sem = data.get("sem")
    return {
        "Property": data.get("property_name", data.get("Property", "")),
        "mean +/- sd": data.get("mean_plus_minus_sd", _format_mean_plus_minus(mean, sd if sd is not None else sem)),
        "n": data.get("n", ""),
        "Source": data.get("source", data.get("Source", "")),
        "Notes": data.get("notes", data.get("Notes", "")),
        "cell_type": data.get("cell_type", ""),
        "marker_profile": data.get("marker_profile", ""),
        "protocol_id": data.get("protocol_id", ""),
        "mean": mean if mean is not None else "",
        "sd": sd if sd is not None else "",
        "sem": sem if sem is not None else "",
        "stat_type": data.get("stat_type", ""),
        "unit": data.get("unit", ""),
        "source_file": data.get("source_file", ""),
        "source_location": data.get("source_location", ""),
        "source_url": data.get("source_url", ""),
        "data_kind": data.get("data_kind", ""),
        "extraction_method": data.get("extraction_method", ""),
        "include_in_validation": _bool_csv(bool(data.get("include_in_validation", False))),
        "include_in_fi_validation": _bool_csv(bool(data.get("include_in_fi_validation", False))),
        "confidence": data.get("confidence", ""),
        "note_ids": _join_note_ids(data.get("note_ids", "")),
        "reported_value_raw": data.get("reported_value_raw", ""),
    }


def _build_identity_row(data: dict[str, Any]) -> dict[str, object]:
    return {
        "source": data.get("source", ""),
        "source_file": data.get("source_file", ""),
        "source_location": data.get("source_location", ""),
        "cell_type": data.get("cell_type", ""),
        "marker_profile": data.get("marker_profile", ""),
        "identity_kind": data.get("identity_kind", ""),
        "Property": data.get("property_name", data.get("Property", "")),
        "source_url": data.get("source_url", ""),
        "mean": data.get("mean", "") if data.get("mean") is not None else "",
        "sd": data.get("sd", "") if data.get("sd") is not None else "",
        "sem": data.get("sem", "") if data.get("sem") is not None else "",
        "stat_type": data.get("stat_type", ""),
        "unit": data.get("unit", ""),
        "n": data.get("n", ""),
        "data_kind": data.get("data_kind", ""),
        "extraction_method": data.get("extraction_method", ""),
        "include_in_validation": _bool_csv(bool(data.get("include_in_validation", False))),
        "confidence": data.get("confidence", ""),
        "note_ids": _join_note_ids(data.get("note_ids", "")),
        "notes": data.get("notes", ""),
        "reported_value_raw": data.get("reported_value_raw", ""),
    }


def _build_fi_curve_row(data: dict[str, Any]) -> dict[str, object]:
    return {
        "source": data.get("source", ""),
        "source_file": data.get("source_file", ""),
        "source_location": data.get("source_location", ""),
        "cell_type": data.get("cell_type", ""),
        "cell_id": data.get("cell_id", ""),
        "marker_profile": data.get("marker_profile", ""),
        "protocol_id": data.get("protocol_id", ""),
        "source_url": data.get("source_url", ""),
        "current_pA": data.get("current_pA", ""),
        "firing_rate_Hz": data.get("firing_rate_Hz", ""),
        "rate_definition": data.get("rate_definition", ""),
        "step_duration_ms": data.get("step_duration_ms", ""),
        "current_start_pA": data.get("current_start_pA", ""),
        "current_stop_pA": data.get("current_stop_pA", ""),
        "current_step_pA": data.get("current_step_pA", ""),
        "baseline_or_holding_vm_mV": data.get("baseline_or_holding_vm_mV", ""),
        "synaptic_blockers": data.get("synaptic_blockers", ""),
        "temperature_C": data.get("temperature_C", ""),
        "sample_scope": data.get("sample_scope", ""),
        "extraction_method": data.get("extraction_method", ""),
        "include_in_validation": _bool_csv(bool(data.get("include_in_validation", False))),
        "confidence": data.get("confidence", ""),
        "note_ids": _join_note_ids(data.get("note_ids", "")),
        "notes": data.get("notes", ""),
    }


def _build_protocol_row(data: dict[str, Any]) -> dict[str, object]:
    return {
        "protocol_id": data.get("protocol_id", ""),
        "source": data.get("source", ""),
        "cell_type": data.get("cell_type", ""),
        "marker_profile": data.get("marker_profile", ""),
        "stimulus_type": data.get("stimulus_type", ""),
        "step_duration_ms": data.get("step_duration_ms", ""),
        "current_start_pA": data.get("current_start_pA", ""),
        "current_stop_pA": data.get("current_stop_pA", ""),
        "current_step_pA": data.get("current_step_pA", ""),
        "current_values_pA": data.get("current_values_pA", ""),
        "rate_definition": data.get("rate_definition", ""),
        "spike_detection_rule": data.get("spike_detection_rule", ""),
        "baseline_or_holding_vm_mV": data.get("baseline_or_holding_vm_mV", ""),
        "synaptic_blockers": data.get("synaptic_blockers", ""),
        "temperature_C": data.get("temperature_C", ""),
        "compatible_group": data.get("compatible_group", ""),
        "notes": data.get("notes", ""),
    }


def _build_note_row(data: dict[str, Any]) -> dict[str, object]:
    return {
        "note_id": data.get("note_id", ""),
        "severity": data.get("severity", ""),
        "scope": data.get("scope", ""),
        "target_type": data.get("target_type", ""),
        "target": data.get("target", ""),
        "message": data.get("message", ""),
        "display_order": data.get("display_order", ""),
        "source": data.get("source", ""),
        "source_location": data.get("source_location", ""),
    }


def _build_manual_row(data: dict[str, Any]) -> dict[str, object]:
    return {
        "source": data.get("source", ""),
        "source_file": data.get("source_file", ""),
        "figure_or_table": data.get("figure_or_table", ""),
        "target_metric": data.get("target_metric", ""),
        "reason": data.get("reason", ""),
        "suggested_action": data.get("suggested_action", ""),
    }


def _summary_rule_rows(
    config: dict[str, Any],
    cache: TableCache,
    rule: dict[str, Any],
) -> list[dict[str, object]]:
    output = str(rule["output"])
    df = _apply_row_filters(cache.load(str(rule["source_id"]), sheet_name=str(rule.get("sheet_name", ""))), rule)
    column = _clean_label(rule["column"])
    if column not in df.columns:
        raise KeyError(f"Missing column {column!r} in summary rule for {rule.get('property_name')}")
    scale = float(rule.get("transform_scale", 1.0))
    values = [
        float(value) * scale
        for value in (_float_or_none(item) for item in df[column].tolist())
        if value is not None
    ]
    mean_value, sd_value, sem_value, count = _stats(values)
    source_id = str(rule["source_id"])
    source_meta = source_entry(source_id, config=config)
    base_row = {
        "source": rule.get("source", source_meta.get("source", "")),
        "source_file": source_meta.get("filename", ""),
        "source_url": stable_source_url(source_id, config=config),
        "source_location": str(
            rule.get(
                "source_location",
                f"{_clean_label(source_meta.get('label', source_id))}, sheet '{rule.get('sheet_name', '')}', column '{column}'",
            )
        ),
        "cell_type": rule.get("cell_type", ""),
        "marker_profile": rule.get("marker_profile", ""),
        "property_name": rule.get("property_name", ""),
        "mean": mean_value,
        "sd": sd_value,
        "sem": sem_value,
        "n": count,
        "stat_type": rule.get("stat_type", "mean_sd"),
        "unit": rule.get("unit", ""),
        "data_kind": rule.get("data_kind", ""),
        "extraction_method": rule.get("extraction_method", "source_spreadsheet"),
        "include_in_validation": bool(rule.get("include_in_validation", True)),
        "include_in_fi_validation": bool(rule.get("include_in_fi_validation", False)),
        "confidence": rule.get("confidence", "high"),
        "protocol_id": rule.get("protocol_id", ""),
        "note_ids": rule.get("note_ids", []),
        "notes": rule.get("notes", ""),
        "reported_value_raw": rule.get(
            "reported_value_template",
            f"Computed from {_clean_label(source_meta.get('label', source_id))} rows for {rule.get('reported_definition', rule.get('property_name', 'metric'))} "
            f"(n = {count}, mean = {mean_value:.6g}, sd = {sd_value:.6g}, sem = {sem_value:.6g})",
        ),
    }
    if output == "ephys":
        return [_build_ephys_row(base_row)]
    if output == "identity":
        base_row["identity_kind"] = rule.get("identity_kind", "")
        return [_build_identity_row(base_row)]
    raise ValueError(f"Unsupported summary output type: {output}")


def _point_rule_rows(config: dict[str, Any], cache: TableCache, rule: dict[str, Any]) -> list[dict[str, object]]:
    df = _apply_row_filters(cache.load(str(rule["source_id"]), sheet_name=str(rule.get("sheet_name", ""))), rule)
    current_column = _clean_label(rule["current_column"])
    value_column = _clean_label(rule["value_column"])
    if current_column not in df.columns or value_column not in df.columns:
        raise KeyError(f"Missing point-rule columns for source {rule['source_id']}")
    current_min = _float_or_none(rule.get("current_min_pA"))
    current_max = _float_or_none(rule.get("current_max_pA"))
    source_id = str(rule["source_id"])
    source_meta = source_entry(source_id, config=config)

    rows: list[dict[str, object]] = []
    for _, record in df.iterrows():
        current_value = _float_or_none(record[current_column])
        point_value = _float_or_none(record[value_column])
        if current_value is None or point_value is None:
            continue
        if current_min is not None and current_value < current_min:
            continue
        if current_max is not None and current_value > current_max:
            continue
        row = _build_fi_curve_row(
            {
                "source": rule.get("source", source_meta.get("source", "")),
                "source_file": source_meta.get("filename", ""),
                "source_location": rule.get("source_location", f"sheet '{rule.get('sheet_name', '')}'"),
                "cell_type": rule.get("cell_type", ""),
                "cell_id": rule.get("cell_id", ""),
                "marker_profile": rule.get("marker_profile", ""),
                "protocol_id": rule.get("protocol_id", ""),
                "source_url": stable_source_url(source_id, config=config),
                "current_pA": current_value,
                "firing_rate_Hz": point_value,
                "rate_definition": rule.get("rate_definition", ""),
                "step_duration_ms": rule.get("step_duration_ms", ""),
                "current_start_pA": rule.get("current_start_pA", ""),
                "current_stop_pA": rule.get("current_stop_pA", ""),
                "current_step_pA": rule.get("current_step_pA", ""),
                "baseline_or_holding_vm_mV": rule.get("baseline_or_holding_vm_mV", ""),
                "synaptic_blockers": rule.get("synaptic_blockers", ""),
                "temperature_C": rule.get("temperature_C", ""),
                "sample_scope": rule.get("sample_scope", ""),
                "extraction_method": rule.get("extraction_method", "source_spreadsheet"),
                "include_in_validation": bool(rule.get("include_in_validation", True)),
                "confidence": rule.get("confidence", "high"),
                "note_ids": rule.get("note_ids", []),
                "notes": rule.get("notes", ""),
            }
        )
        rows.append(row)
    return rows


def _evaluate_condition(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    condition = str(rule.get("condition", "")).strip()
    if not condition:
        return True
    if condition == "missing_source":
        return str(rule.get("source_id", "")) in set(context.get("missing_source_ids", []))
    if condition == "output_empty":
        output = str(rule.get("output", "")).strip()
        return len(context.get("rows", {}).get(output, [])) == 0
    raise ValueError(f"Unsupported conditional rule: {condition}")


def _static_rows(section: list[dict[str, Any]], builder) -> list[dict[str, object]]:
    return [builder(row) for row in section]


def _conditional_rows(section: list[dict[str, Any]], builder, context: dict[str, Any]) -> list[dict[str, object]]:
    return [builder(row) for row in section if _evaluate_condition(row, context)]


def _render_readme(config: dict[str, Any], context: dict[str, Any]) -> str:
    readme = dict(config.get("readme", {}))
    title = str(readme.get("title", f"{config.get('dataset_name', config.get('dataset_id'))} reference-data extraction"))
    lines = [f"# {title}", ""]
    summary = str(readme.get("summary", "")).strip()
    if summary:
        lines.append(summary)
        lines.append("")

    def add_list_section(header: str, values: list[str]) -> None:
        if not values:
            return
        lines.append(f"## {header}")
        lines.append("")
        for value in values:
            lines.append(f"- {value}")
        lines.append("")

    add_list_section("Source summary", list(readme.get("source_summary", []) or []))
    add_list_section("Suitable now", list(readme.get("suitable_now", []) or []))
    add_list_section("Caveats", list(readme.get("caveats", []) or []))

    lines.append("## Extraction status")
    lines.append("")
    for output_key, rows in context["rows"].items():
        lines.append(f"- `{output_key}` rows: {len(rows)}")
    missing = list(context.get("missing_source_ids", []))
    lines.append(f"- Missing required sources after acquisition: {', '.join(missing) if missing else 'none'}")
    lines.append("")
    return "\n".join(lines)


def extract_reference_dataset(*, dataset_id: str, config_path: Path | None = None) -> dict[str, Any]:
    config = load_dataset_config(dataset_id=dataset_id, path=config_path)
    source_ids = [
        str(source["source_id"])
        for source in dataset_sources(config)
        if bool(source.get("required", False)) and bool(source.get("downloadable", True))
    ]
    downloaded, download_errors = ensure_reference_sources(
        config=config,
        source_ids=source_ids,
        include_optional=False,
        strict=False,
    )
    missing_source_ids = [source_id for source_id in source_ids if source_id not in downloaded]

    cache = TableCache(config)
    rows: dict[str, list[dict[str, object]]] = {
        "ephys": _static_rows(dataset_section(config, "static_ephys_rows"), _build_ephys_row),
        "fi_curve": _static_rows(dataset_section(config, "static_fi_curve_rows"), _build_fi_curve_row),
        "protocols": _static_rows(dataset_section(config, "static_protocol_rows"), _build_protocol_row),
        "identity": _static_rows(dataset_section(config, "static_identity_rows"), _build_identity_row),
        "notes": _static_rows(dataset_section(config, "static_note_rows"), _build_note_row),
        "manual": _static_rows(dataset_section(config, "static_manual_rows"), _build_manual_row),
    }

    for rule in dataset_section(config, "summary_rules"):
        output = str(rule["output"])
        rows[output].extend(_summary_rule_rows(config, cache, rule))
    for rule in dataset_section(config, "point_rules"):
        rows["fi_curve"].extend(_point_rule_rows(config, cache, rule))

    context = {
        "config": config,
        "download_errors": download_errors,
        "missing_source_ids": missing_source_ids,
        "rows": rows,
    }
    rows["notes"].extend(_conditional_rows(dataset_section(config, "conditional_note_rows"), _build_note_row, context))
    rows["manual"].extend(_conditional_rows(dataset_section(config, "conditional_manual_rows"), _build_manual_row, context))
    context["rows"] = rows

    readme_text = _render_readme(config, context)
    return {
        "config": config,
        "rows": rows,
        "download_errors": download_errors,
        "missing_source_ids": missing_source_ids,
        "readme_text": readme_text,
    }


def write_reference_dataset_outputs(*, dataset_id: str, config_path: Path | None = None) -> dict[str, Any]:
    result = extract_reference_dataset(dataset_id=dataset_id, config_path=config_path)
    config = result["config"]
    for output_key, fieldnames in OUTPUT_COLUMNS.items():
        path = dataset_output_path(config, output_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in result["rows"][output_key]:
                writer.writerow(row)
    readme_path = dataset_output_path(config, "readme")
    readme_path.write_text(result["readme_text"])
    return result
