"""Audit that one declarative reference dataset satisfies maintained contract checks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from olfactorybulb.audit.core import AuditItem, AuditReport
from olfactorybulb.audit.reference_data import (
    dataset_output_columns,
    dataset_output_keys,
    dataset_output_path,
    dataset_output_row_type,
    load_dataset_output_rows,
)
from olfactorybulb.audit.reference_dataset_config import DEFAULT_REFERENCE_DATASET_ID, load_dataset_config


NUMERIC_COLUMNS = {
    "mean",
    "sd",
    "sem",
    "q_low",
    "q_high",
    "n",
    "current_pA",
    "firing_rate_Hz",
    "step_duration_ms",
    "current_start_pA",
    "current_stop_pA",
    "current_step_pA",
    "recording_temperature_C",
}
PROVENANCE_FIELDS = ("source_file", "source_url", "source_location")
DIGITIZED_REQUIRED_FIELDS = ("source_location", "confidence", "note_ids")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = __doc__
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_REFERENCE_DATASET_ID,
        help="Reference dataset id to inspect.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional explicit dataset config path. Overrides --dataset-id when provided.",
    )


def _item(
    *,
    check_id: str,
    status: str,
    title: str,
    criterion: str,
    description: str,
    acceptable: str,
    acceptable_basis: str,
    evidence: dict[str, object] | None = None,
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
        validation_design_review_status="not_applicable",
    )


def _header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _boolish(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes"}


def _floatable(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number)


def _missing_rows(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[int]:
    missing: list[int] = []
    for index, row in enumerate(rows, start=2):
        if any(not str(row.get(field, "")).strip() for field in fields):
            missing.append(index)
    return missing


def _bad_numeric_rows(rows: list[dict[str, str]], columns: list[str]) -> dict[str, list[int]]:
    bad: dict[str, list[int]] = {}
    for column in columns:
        failures: list[int] = []
        for index, row in enumerate(rows, start=2):
            if not _floatable(row.get(column, "")):
                failures.append(index)
        if failures:
            bad[column] = failures
    return bad


def run(args: argparse.Namespace) -> AuditReport:
    raw_config_path = getattr(args, "config_path", None)
    config_path = Path(str(raw_config_path)).resolve() if raw_config_path and str(raw_config_path).strip() else None
    config = load_dataset_config(dataset_id=None if config_path else str(args.dataset_id), path=config_path)
    dataset_id = str(config["dataset_id"])
    dataset_name = str(config.get("dataset_name", dataset_id))
    items: list[AuditItem] = []

    for output_key in dataset_output_keys(dataset_id=dataset_id):
        path = dataset_output_path(dataset_id=dataset_id, output_key=output_key)
        row_type = dataset_output_row_type(dataset_id=dataset_id, output_key=output_key)
        expected_columns = dataset_output_columns(dataset_id=dataset_id, output_key=output_key)

        if row_type == "readme":
            continue

        header = _header(path)
        items.append(
            _item(
                check_id=f"{output_key}_schema_columns_match",
                status="PASS" if header == expected_columns else "FAIL",
                title=f"{output_key} output matches its declared schema columns",
                criterion="Every generated reference output should match the columns declared by its dataset config and schema preset exactly.",
                description="This prevents drift between the declarative dataset config, the extraction engine, and the generated canonical files.",
                acceptable="The CSV header matches the configured schema columns exactly.",
                acceptable_basis="The expected columns come from the dataset config output mapping and the shared schema preset registry.",
                evidence={
                    "dataset_id": dataset_id,
                    "output_key": output_key,
                    "row_type": row_type,
                    "path": str(path),
                    "expected_columns": expected_columns,
                    "actual_columns": header,
                },
            )
        )

        rows = load_dataset_output_rows(dataset_id=dataset_id, output_key=output_key)

        required_provenance = [field for field in PROVENANCE_FIELDS if field in header]
        if row_type == "manual":
            required_provenance = [field for field in required_provenance if field != "source_url"]
        if required_provenance:
            missing = _missing_rows(rows, tuple(required_provenance))
            items.append(
                _item(
                    check_id=f"{output_key}_provenance_fields_populated",
                    status="PASS" if not missing else "FAIL",
                    title=f"{output_key} rows carry populated provenance fields",
                    criterion="Generated reference rows should preserve source file, stable source URL, and source location whenever those fields are part of the schema.",
                    description="This enforces row-level traceability so suspicious values can be audited back to a specific paper asset or location later.",
                    acceptable="Every populated row has non-empty provenance fields present in its schema.",
                    acceptable_basis="These are maintained repository rules for reproducible literature-backed validation, not optional display metadata.",
                    evidence={
                        "dataset_id": dataset_id,
                        "output_key": output_key,
                        "required_fields": required_provenance,
                        "missing_row_numbers": missing[:25],
                        "missing_row_count": len(missing),
                    },
                )
            )

        numeric_columns = [column for column in header if column in NUMERIC_COLUMNS]
        if numeric_columns:
            bad_numeric = _bad_numeric_rows(rows, numeric_columns)
            items.append(
                _item(
                    check_id=f"{output_key}_numeric_columns_parse",
                    status="PASS" if not bad_numeric else "FAIL",
                    title=f"{output_key} numeric contract fields parse as numeric where populated",
                    criterion="Declared numeric fields in generated reference outputs should contain numeric values wherever they are populated.",
                    description="This catches extraction corruption such as wrong unit scaling, text leaks into numeric columns, and malformed empty-value handling.",
                    acceptable="Every populated numeric field parses cleanly as a finite number.",
                    acceptable_basis="The numeric field set is derived from the maintained shared reference-data schema rather than from ad hoc dataset-specific heuristics.",
                    evidence={
                        "dataset_id": dataset_id,
                        "output_key": output_key,
                        "bad_numeric_rows": bad_numeric,
                    },
                )
            )

        if row_type == "notes":
            missing = _missing_rows(rows, ("note_id", "severity", "scope", "target_type", "message"))
            note_ids = [str(row.get("note_id", "")).strip() for row in rows if str(row.get("note_id", "")).strip()]
            duplicates = sorted({note_id for note_id in note_ids if note_ids.count(note_id) > 1})
            items.extend(
                [
                    _item(
                        check_id=f"{output_key}_required_fields_populated",
                        status="PASS" if not missing else "FAIL",
                        title=f"{output_key} note rows carry required metadata",
                        criterion="Validation-note rows should expose complete identifiers, scope, target type, and message fields.",
                        description="The note system is the maintained vehicle for protocol and provenance caveats, so partial note rows are treated as contract breakage.",
                        acceptable="Every notes row has non-empty required note metadata.",
                        acceptable_basis="These fields are required by the maintained note loader and downstream note-rendering surfaces.",
                        evidence={
                            "dataset_id": dataset_id,
                            "output_key": output_key,
                            "missing_row_numbers": missing[:25],
                            "missing_row_count": len(missing),
                        },
                    ),
                    _item(
                        check_id=f"{output_key}_note_ids_unique",
                        status="PASS" if not duplicates else "FAIL",
                        title=f"{output_key} note identifiers are unique",
                        criterion="Each notes table should assign one unique row to each note identifier.",
                        description="Allowing duplicate note identifiers makes downstream matching and human review ambiguous, especially once note ids become stable references across audits.",
                        acceptable="No note identifier appears more than once in the notes table.",
                        acceptable_basis="The maintained note system de-duplicates by note identifier downstream, so duplicate ids would silently collapse distinct rows.",
                        evidence={
                            "dataset_id": dataset_id,
                            "output_key": output_key,
                            "duplicate_note_ids": duplicates,
                        },
                    ),
                ]
            )

        if row_type == "fi_curve" and rows:
            missing_protocol = _missing_rows(rows, ("protocol_id",))
            missing_notes = _missing_rows(rows, ("note_ids",))
            items.extend(
                [
                    _item(
                        check_id=f"{output_key}_protocol_ids_present",
                        status="PASS" if not missing_protocol else "FAIL",
                        title=f"{output_key} current-rate rows declare protocol identifiers",
                        criterion="Every current-versus-rate row should declare the protocol it came from.",
                        description="Exact current-rate comparisons are only meaningful when the reference rows retain explicit protocol identity.",
                        acceptable="Every f-I point row has a non-empty protocol_id.",
                        acceptable_basis="The maintained reference-validation rules compare f-I rows by protocol id and surface explicit caveats when protocols differ.",
                        evidence={
                            "dataset_id": dataset_id,
                            "output_key": output_key,
                            "missing_row_numbers": missing_protocol[:25],
                            "missing_row_count": len(missing_protocol),
                        },
                    ),
                    _item(
                        check_id=f"{output_key}_note_ids_present",
                        status="PASS" if not missing_notes else "FAIL",
                        title=f"{output_key} current-rate rows carry note identifiers",
                        criterion="Every current-versus-rate row should carry note identifiers when downstream outputs may need protocol caveats.",
                        description="This prevents caveats from being dropped when the raw CSV is used outside the richer HTML or CLI renderers.",
                        acceptable="Every f-I point row has non-empty note_ids.",
                        acceptable_basis="The repository now treats note propagation as part of the reference-data contract, not just a display detail.",
                        evidence={
                            "dataset_id": dataset_id,
                            "output_key": output_key,
                            "missing_row_numbers": missing_notes[:25],
                            "missing_row_count": len(missing_notes),
                        },
                    ),
                ]
            )

        if "include_in_fi_validation" in header:
            fi_summary_rows = [row for row in rows if _boolish(row.get("include_in_fi_validation", ""))]
            if fi_summary_rows:
                missing_protocol = _missing_rows(fi_summary_rows, ("protocol_id",))
                missing_notes = _missing_rows(fi_summary_rows, ("note_ids",))
                items.extend(
                    [
                        _item(
                            check_id=f"{output_key}_fi_summary_protocol_ids_present",
                            status="PASS" if not missing_protocol else "FAIL",
                            title=f"{output_key} firing-rate summary rows declare protocol identifiers",
                            criterion="Reference rows that participate in f-I validation should retain protocol identifiers even when they are summary metrics rather than point curves.",
                            description="Summary f-I metrics still need protocol identity so validation can distinguish exact matches from caveated cross-protocol comparisons.",
                            acceptable="Every row marked include_in_fi_validation=true has a non-empty protocol_id.",
                            acceptable_basis="The declarative reference-validation layer filters and renders f-I comparisons by protocol identity.",
                            evidence={
                                "dataset_id": dataset_id,
                                "output_key": output_key,
                                "row_count": len(fi_summary_rows),
                                "missing_row_numbers": missing_protocol[:25],
                                "missing_row_count": len(missing_protocol),
                            },
                        ),
                        _item(
                            check_id=f"{output_key}_fi_summary_note_ids_present",
                            status="PASS" if not missing_notes else "FAIL",
                            title=f"{output_key} firing-rate summary rows carry note identifiers",
                            criterion="Reference rows that participate in f-I validation should preserve their note identifiers.",
                            description="This keeps protocol caveats attached when only summary-level firing-rate metrics are available.",
                            acceptable="Every row marked include_in_fi_validation=true has non-empty note_ids.",
                            acceptable_basis="The maintained reference-data contract requires caveats to travel with the data rather than living only in prose.",
                            evidence={
                                "dataset_id": dataset_id,
                                "output_key": output_key,
                                "row_count": len(fi_summary_rows),
                                "missing_row_numbers": missing_notes[:25],
                                "missing_row_count": len(missing_notes),
                            },
                        ),
                    ]
                )

        if "extraction_method" in header:
            digitized_rows = [row for row in rows if str(row.get("extraction_method", "")).strip() == "figure_digitized"]
            if digitized_rows:
                missing_digitized = _missing_rows(digitized_rows, DIGITIZED_REQUIRED_FIELDS)
                items.append(
                    _item(
                        check_id=f"{output_key}_digitized_rows_carry_provenance",
                        status="PASS" if not missing_digitized else "FAIL",
                        title=f"{output_key} digitized rows carry explicit digitization metadata",
                        criterion="Figure-digitized rows should expose provenance, confidence, and note identifiers explicitly.",
                        description="Digitized rows are intentionally lower confidence than machine-readable rows, so the maintained contract requires them to remain visible and inspectable downstream.",
                        acceptable="Every figure-digitized row has non-empty source_location, confidence, and note_ids.",
                        acceptable_basis="This is the repository-wide rule for lower-confidence figure-derived literature rows.",
                        evidence={
                            "dataset_id": dataset_id,
                            "output_key": output_key,
                            "digitized_row_count": len(digitized_rows),
                            "missing_row_numbers": missing_digitized[:25],
                            "missing_row_count": len(missing_digitized),
                        },
                    )
                )

    return AuditReport(
        audit_id="reference_dataset_contracts",
        title=f"Reference dataset contract audit ({dataset_name})",
        items=items,
    )
