"""Generic extraction engine for declarative reference-data datasets."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd

from .reference_data import PROPERTY_UNITS
from .reference_dataset_config import (
    dataset_output_path,
    dataset_output_specs,
    dataset_section,
    dataset_sources,
    load_dataset_config,
    primary_output_key,
)
from .reference_sources import ensure_reference_sources, local_source_path, source_entry, stable_source_url


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
    mean_value = _float_or_none(mean)
    spread_value = _float_or_none(spread)
    if mean_value is None or spread_value is None:
        return ""
    return f"{mean_value:g} +/- {spread_value:g}"


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


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile for an empty value list")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError(f"Quantile must be between 0 and 1 inclusive, got {quantile}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _quantile_label(quantile: float) -> str:
    percent = float(quantile) * 100.0
    if percent.is_integer():
        percent_text = str(int(percent))
    else:
        percent_text = f"{percent:g}"
    return f"{percent_text}th percentile"


def _parse_summary_stat_cell(text: object) -> tuple[float | None, float | None, float | None, int | None]:
    raw = " ".join(str(text or "").replace("−", "-").replace("±", "+/-").split())
    if not raw:
        return None, None, None, None
    import re

    match = re.match(
        r"^(?P<mean>-?\d+(?:\.\d+)?)\s*\+/-\s*(?P<spread>\d+(?:\.\d+)?)\s*\((?P<n>\d+)\)$",
        raw,
    )
    if match:
        return float(match.group("mean")), float(match.group("spread")), None, int(match.group("n"))

    match = re.match(
        r"^(?P<mean>-?\d+(?:\.\d+)?)\s*\((?P<n>\d+)\)$",
        raw,
    )
    if match:
        return float(match.group("mean")), None, None, int(match.group("n"))

    return None, None, None, None


class TableCache:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def load(self, source_id: str, *, sheet_name: str = "", table_index: int | None = None) -> pd.DataFrame:
        cache_selector = str(table_index) if table_index is not None else sheet_name
        key = (source_id, cache_selector)
        if key in self._cache:
            return self._cache[key]

        path = local_source_path(source_id, config=self.config)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xls", ".xlsx"}:
            selected_sheet: int | str = sheet_name if sheet_name else 0
            df = pd.read_excel(path, sheet_name=selected_sheet)
        elif suffix in {".htm", ".html"}:
            tables = pd.read_html(path)
            selected_index = table_index if table_index is not None else int(sheet_name or 0)
            df = tables[selected_index]
        elif suffix == ".docx":
            tables = self._read_docx_tables(path)
            selected_index = (table_index if table_index is not None else int(sheet_name or 1)) - 1
            df = tables[selected_index]
        else:
            raise ValueError(f"Unsupported tabular source format for {path}")

        df.columns = [_clean_label(column) for column in df.columns]
        self._cache[key] = df
        return df

    @staticmethod
    def _read_docx_tables(path: Path) -> list[pd.DataFrame]:
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))

        tables: list[pd.DataFrame] = []
        for table in root.findall(".//w:tbl", ns):
            rows: list[list[str]] = []
            for tr in table.findall("./w:tr", ns):
                cells: list[str] = []
                for tc in tr.findall("./w:tc", ns):
                    texts = [node.text for node in tc.findall(".//w:t", ns) if node.text]
                    cells.append(" ".join(texts).strip())
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue

            header_row_index = 0
            if "supplementary table" in rows[0][0].lower() and len(rows) > 1:
                header_row_index = 1
            headers = rows[header_row_index]
            width = len(headers)
            normalized_rows: list[list[str]] = []
            for row in rows[header_row_index + 1 :]:
                padded = row + [""] * (width - len(row))
                normalized_rows.append(padded[:width])
            tables.append(pd.DataFrame(normalized_rows, columns=headers))
        return tables


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


def _resolve_row_value(column: str, data: dict[str, Any]) -> object:
    if column in data:
        value = data[column]
    elif column == "Property":
        value = data.get("property_name", data.get("Property", ""))
    elif column == "mean +/- sd":
        mean = data.get("mean")
        sd = data.get("sd")
        sem = data.get("sem")
        value = data.get("mean_plus_minus_sd", _format_mean_plus_minus(mean, sd if sd is not None else sem))
    elif column == "Source":
        value = data.get("source", data.get("Source", ""))
    elif column == "Notes":
        value = data.get("notes", data.get("Notes", ""))
    else:
        value = data.get(column, "")

    if value is None:
        return ""
    if column == "note_ids":
        return _join_note_ids(value)
    if column.startswith("include_in_") and isinstance(value, bool):
        return _bool_csv(value)
    return value


def _build_row(columns: list[str], data: dict[str, Any]) -> dict[str, object]:
    return {column: _resolve_row_value(column, data) for column in columns}


def _table_selector(rule: dict[str, Any]) -> tuple[str, int | None]:
    sheet_name = str(rule.get("sheet_name", ""))
    table_index = rule.get("table_index")
    return sheet_name, int(table_index) if table_index is not None else None


def _summary_rule_rows(
    config: dict[str, Any],
    cache: TableCache,
    output_specs: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> list[dict[str, object]]:
    output_key = str(rule["output"])
    try:
        output_spec = output_specs[output_key]
    except KeyError as exc:
        raise KeyError(f"Unknown output key {output_key!r} in summary rule") from exc
    sheet_name, table_index = _table_selector(rule)
    df = _apply_row_filters(cache.load(str(rule["source_id"]), sheet_name=sheet_name, table_index=table_index), rule)
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
    quantile_low = _float_or_none(rule.get("quantile_low"))
    quantile_high = _float_or_none(rule.get("quantile_high"))
    q_low_value: float | str = ""
    q_high_value: float | str = ""
    q_low_label = ""
    q_high_label = ""
    if quantile_low is not None and quantile_high is not None:
        q_low_value = _quantile(values, quantile_low)
        q_high_value = _quantile(values, quantile_high)
        q_low_label = str(rule.get("quantile_low_label") or _quantile_label(quantile_low))
        q_high_label = str(rule.get("quantile_high_label") or _quantile_label(quantile_high))
    source_id = str(rule["source_id"])
    source_meta = source_entry(source_id, config=config)
    base_row = {
        "source": rule.get("source", source_meta.get("source", "")),
        "source_file": source_meta.get("filename", ""),
        "source_url": stable_source_url(source_id, config=config),
        "source_location": str(
            rule.get(
                "source_location",
                f"{_clean_label(source_meta.get('label', source_id))}, selector '{sheet_name or table_index or 0}', column '{column}'",
            )
        ),
        "cell_type": rule.get("cell_type", ""),
        "marker_profile": rule.get("marker_profile", ""),
        "property_name": rule.get("property_name", ""),
        "mean": mean_value,
        "sd": sd_value,
        "sem": sem_value,
        "q_low": q_low_value,
        "q_high": q_high_value,
        "q_low_label": q_low_label,
        "q_high_label": q_high_label,
        "n": count,
        "stat_type": rule.get("stat_type", "mean_sd"),
        "unit": rule.get("unit") or PROPERTY_UNITS.get(str(rule.get("property_name", "")), ""),
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
    if output_spec["row_type"] == "identity":
        base_row["identity_kind"] = rule.get("identity_kind", "")
    if output_spec["row_type"] not in {"ephys", "identity"}:
        raise ValueError(f"Unsupported summary output row type: {output_spec['row_type']}")
    return [_build_row(output_spec["columns"], base_row)]


def _point_rule_rows(
    config: dict[str, Any],
    cache: TableCache,
    output_specs: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> list[dict[str, object]]:
    output_key = str(rule.get("output") or primary_output_key(config, "fi_curve"))
    try:
        output_spec = output_specs[output_key]
    except KeyError as exc:
        raise KeyError(f"Unknown output key {output_key!r} in point rule") from exc
    if output_spec["row_type"] != "fi_curve":
        raise ValueError(f"Point rules require a fi_curve output, got {output_spec['row_type']!r} for {output_key!r}")
    sheet_name, table_index = _table_selector(rule)
    df = _apply_row_filters(cache.load(str(rule["source_id"]), sheet_name=sheet_name, table_index=table_index), rule)
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
        row = _build_row(
            output_spec["columns"],
            {
                "source": rule.get("source", source_meta.get("source", "")),
                "source_file": source_meta.get("filename", ""),
                "source_location": rule.get("source_location", f"selector '{sheet_name or table_index or 0}'"),
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


def _formatted_summary_rule_rows(
    config: dict[str, Any],
    cache: TableCache,
    output_specs: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> list[dict[str, object]]:
    output_key = str(rule["output"])
    try:
        output_spec = output_specs[output_key]
    except KeyError as exc:
        raise KeyError(f"Unknown output key {output_key!r} in formatted summary rule") from exc
    if output_spec["row_type"] not in {"ephys", "identity"}:
        raise ValueError(
            f"Formatted summary rules require ephys or identity outputs, got {output_spec['row_type']!r}"
        )

    sheet_name, table_index = _table_selector(rule)
    df = _apply_row_filters(cache.load(str(rule["source_id"]), sheet_name=sheet_name, table_index=table_index), rule)
    property_column = _clean_label(rule.get("property_column", ""))
    value_column = _clean_label(rule.get("value_column", ""))
    if property_column not in df.columns or value_column not in df.columns:
        raise KeyError(f"Missing formatted-summary columns in source {rule['source_id']}")

    property_map = {_clean_label(key): str(value) for key, value in dict(rule.get("property_map", {})).items()}
    unit_map = {_clean_label(key): str(value) for key, value in dict(rule.get("unit_map", {})).items()}
    include_unmapped = bool(rule.get("include_unmapped", False))
    default_scale = float(rule.get("transform_scale", 1.0))
    property_transform_scales = {
        _clean_label(key): float(value)
        for key, value in dict(rule.get("property_transform_scales", {})).items()
    }
    stat_type = str(rule.get("stat_type", "mean_sd"))
    source_id = str(rule["source_id"])
    source_meta = source_entry(source_id, config=config)

    rows: list[dict[str, object]] = []
    for _, record in df.iterrows():
        raw_property = _clean_label(record[property_column])
        if not raw_property:
            continue
        property_name = property_map.get(raw_property, raw_property if include_unmapped else "")
        if not property_name:
            continue
        scale = property_transform_scales.get(raw_property, default_scale)

        mean_value, spread_value, sem_value, count = _parse_summary_stat_cell(record[value_column])
        if mean_value is None:
            continue
        if mean_value is not None:
            mean_value *= scale
        if spread_value is not None:
            spread_value *= scale
        if sem_value is not None:
            sem_value *= scale

        sd_field: float | str = ""
        sem_field: float | str = ""
        if spread_value is not None:
            if stat_type == "mean_sd":
                sd_field = spread_value
            elif stat_type == "mean_sem":
                sem_field = spread_value
            elif stat_type == "sd_from_sem":
                sem_field = spread_value
                sd_field = spread_value * math.sqrt(count) if count else ""
            else:
                sd_field = spread_value

        base_row = {
            "source": rule.get("source", source_meta.get("source", "")),
            "source_file": source_meta.get("filename", ""),
            "source_url": stable_source_url(source_id, config=config),
            "source_location": rule.get(
                "source_location",
                f"{_clean_label(source_meta.get('label', source_id))}, selector '{sheet_name or table_index or 0}'",
            ),
            "Property": property_name,
            "property_name": property_name,
            "mean": mean_value,
            "sd": sd_field,
            "sem": sem_field if sem_field != "" else (sem_value if sem_value is not None else ""),
            "n": count if count is not None else "",
            "stat_type": stat_type,
            "unit": rule.get("unit") or unit_map.get(raw_property, PROPERTY_UNITS.get(property_name, "")),
            "data_kind": rule.get("data_kind", ""),
            "extraction_method": rule.get("extraction_method", "source_table_summary"),
            "include_in_validation": bool(rule.get("include_in_validation", True)),
            "include_in_fi_validation": bool(rule.get("include_in_fi_validation", False)),
            "confidence": rule.get("confidence", "high"),
            "protocol_id": rule.get("protocol_id", ""),
            "note_ids": rule.get("note_ids", []),
            "notes": rule.get("notes", ""),
            "reported_value_raw": str(record[value_column]).strip(),
        }
        for key in (
            "cell_type",
            "marker_profile",
            "gc_subtype",
            "species",
            "age",
            "maturity",
            "layer_or_location",
            "recording_temperature_C",
            "temperature_C",
            "identity_kind",
            "sample_scope",
        ):
            if key in rule:
                base_row[key] = rule[key]
        rows.append(_build_row(output_spec["columns"], base_row))
    return rows


def _evaluate_condition(rule: dict[str, Any], context: dict[str, Any]) -> bool:
    condition = str(rule.get("condition", "")).strip()
    if not condition:
        return True
    if condition == "missing_source":
        return str(rule.get("source_id", "")) in set(context.get("missing_source_ids", []))
    if condition == "output_empty":
        output = str(rule.get("condition_output", rule.get("output", ""))).strip()
        return len(context.get("rows", {}).get(output, [])) == 0
    raise ValueError(f"Unsupported conditional rule: {condition}")


def _static_rows(
    config: dict[str, Any],
    output_specs: dict[str, dict[str, Any]],
    section: list[dict[str, Any]],
    row_type: str,
) -> dict[str, list[dict[str, object]]]:
    rows_by_output: dict[str, list[dict[str, object]]] = {key: [] for key in output_specs}
    default_output_key = primary_output_key(config, row_type)
    for row in section:
        output_key = str(row.get("output") or default_output_key)
        try:
            output_spec = output_specs[output_key]
        except KeyError as exc:
            raise KeyError(f"Unknown output key {output_key!r} in static {row_type} rows") from exc
        if output_spec["row_type"] != row_type:
            raise ValueError(
                f"Static row output {output_key!r} has row type {output_spec['row_type']!r}, expected {row_type!r}"
            )
        rows_by_output[output_key].append(_build_row(output_spec["columns"], row))
    return rows_by_output


def _conditional_rows(
    config: dict[str, Any],
    output_specs: dict[str, dict[str, Any]],
    section: list[dict[str, Any]],
    row_type: str,
    context: dict[str, Any],
) -> dict[str, list[dict[str, object]]]:
    rows_by_output: dict[str, list[dict[str, object]]] = {key: [] for key in output_specs}
    default_output_key = primary_output_key(config, row_type)
    for row in section:
        if not _evaluate_condition(row, context):
            continue
        output_key = str(row.get("output") or default_output_key)
        try:
            output_spec = output_specs[output_key]
        except KeyError as exc:
            raise KeyError(f"Unknown output key {output_key!r} in conditional {row_type} rows") from exc
        if output_spec["row_type"] != row_type:
            raise ValueError(
                f"Conditional row output {output_key!r} has row type {output_spec['row_type']!r}, expected {row_type!r}"
            )
        rows_by_output[output_key].append(_build_row(output_spec["columns"], row))
    return rows_by_output


def _extend_rows(rows: dict[str, list[dict[str, object]]], additions: dict[str, list[dict[str, object]]]) -> None:
    for output_key, values in additions.items():
        rows.setdefault(output_key, []).extend(values)


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
    output_specs = dataset_output_specs(config)
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
        output_key: [] for output_key, spec in output_specs.items() if spec["row_type"] != "readme"
    }
    _extend_rows(rows, _static_rows(config, output_specs, dataset_section(config, "static_ephys_rows"), "ephys"))
    _extend_rows(rows, _static_rows(config, output_specs, dataset_section(config, "static_fi_curve_rows"), "fi_curve"))
    _extend_rows(rows, _static_rows(config, output_specs, dataset_section(config, "static_protocol_rows"), "protocols"))
    _extend_rows(rows, _static_rows(config, output_specs, dataset_section(config, "static_identity_rows"), "identity"))
    _extend_rows(rows, _static_rows(config, output_specs, dataset_section(config, "static_note_rows"), "notes"))
    _extend_rows(rows, _static_rows(config, output_specs, dataset_section(config, "static_manual_rows"), "manual"))

    for rule in dataset_section(config, "summary_rules"):
        output_key = str(rule["output"])
        rows[output_key].extend(_summary_rule_rows(config, cache, output_specs, rule))
    for rule in dataset_section(config, "formatted_summary_rules"):
        output_key = str(rule["output"])
        rows[output_key].extend(_formatted_summary_rule_rows(config, cache, output_specs, rule))
    for rule in dataset_section(config, "point_rules"):
        output_key = str(rule.get("output") or primary_output_key(config, "fi_curve"))
        rows[output_key].extend(_point_rule_rows(config, cache, output_specs, rule))

    context = {
        "config": config,
        "download_errors": download_errors,
        "missing_source_ids": missing_source_ids,
        "rows": rows,
    }
    _extend_rows(rows, _conditional_rows(config, output_specs, dataset_section(config, "conditional_note_rows"), "notes", context))
    _extend_rows(rows, _conditional_rows(config, output_specs, dataset_section(config, "conditional_manual_rows"), "manual", context))
    context["rows"] = rows

    readme_text = _render_readme(config, context)
    return {
        "config": config,
        "output_specs": output_specs,
        "rows": rows,
        "download_errors": download_errors,
        "missing_source_ids": missing_source_ids,
        "readme_text": readme_text,
    }


def write_reference_dataset_outputs(*, dataset_id: str, config_path: Path | None = None) -> dict[str, Any]:
    result = extract_reference_dataset(dataset_id=dataset_id, config_path=config_path)
    config = result["config"]
    output_specs = result["output_specs"]
    for output_key, spec in output_specs.items():
        if spec["row_type"] == "readme":
            continue
        path = dataset_output_path(config, output_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=spec["columns"])
            writer.writeheader()
            for row in result["rows"][output_key]:
                writer.writerow(row)
    if "readme" in output_specs:
        readme_path = dataset_output_path(config, "readme")
        readme_path.write_text(result["readme_text"])
    return result
