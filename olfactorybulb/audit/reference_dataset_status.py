"""Audit generated outputs for a declarative reference-data dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from olfactorybulb.audit.core import AuditItem, AuditReport
from olfactorybulb.audit.reference_data import csv_rows
from olfactorybulb.audit.reference_dataset_config import (
    DEFAULT_REFERENCE_DATASET_ID,
    dataset_output_path,
    dataset_output_specs,
    load_dataset_config,
)
from olfactorybulb.audit.reference_notes import load_notes


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_REFERENCE_DATASET_ID,
        help="Reference dataset id to inspect.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional explicit dataset config path.",
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
    evidence: dict[str, object],
    note: str = "",
) -> AuditItem:
    return AuditItem(
        check_id=check_id,
        status=status,
        title=title,
        criterion=criterion,
        description=description,
        acceptable=acceptable,
        acceptable_basis=acceptable_basis,
        evidence=evidence,
        note=note,
        human_review_status="not_applicable",
    )


def run(args: argparse.Namespace) -> AuditReport:
    config = load_dataset_config(dataset_id=args.dataset_id, path=args.config_path)
    dataset_id = str(config.get("dataset_id", args.dataset_id))
    dataset_name = str(config.get("dataset_name", dataset_id)).strip() or dataset_id
    output_specs = dataset_output_specs(config)

    items: list[AuditItem] = []
    for output_key, spec in output_specs.items():
        path = dataset_output_path(config, output_key)
        row_type = str(spec["row_type"])
        evidence: dict[str, object] = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "output_key": output_key,
            "schema_name": spec["schema_name"],
            "row_type": row_type,
            "path": str(path),
        }
        if not path.exists():
            items.append(
                _item(
                    check_id=f"{output_key}_exists",
                    status="FAIL",
                    title=f"{dataset_name} {output_key} output exists",
                    criterion="Every declared reference-dataset output should exist on disk after extraction.",
                    description="This checks that the generated canonical output file declared in the dataset config was actually written.",
                    acceptable="The output path exists on disk.",
                    acceptable_basis="Declared outputs come from the centralized dataset config, so a missing file means the extraction surface is incomplete or stale.",
                    evidence=evidence,
                )
            )
            continue

        if row_type == "readme":
            text = path.read_text(encoding="utf-8")
            evidence["character_count"] = len(text)
            items.append(
                _item(
                    check_id=f"{output_key}_loadable",
                    status="PASS" if text.strip() else "FAIL",
                    title=f"{dataset_name} {output_key} output is non-empty",
                    criterion="The dataset README should exist and contain explanatory text.",
                    description="This confirms that the human-facing provenance/readme file was generated and is not empty.",
                    acceptable="The README file contains non-whitespace text.",
                    acceptable_basis="The extractor owns README generation, so an empty file means the generated bundle explanation is missing.",
                    evidence=evidence,
                )
            )
            continue

        if row_type == "notes":
            notes = load_notes(path)
            evidence["row_count"] = len(notes)
            evidence["note_ids"] = [note.note_id for note in notes]
        else:
            rows = csv_rows(path)
            evidence["row_count"] = len(rows)
            if rows:
                evidence["columns"] = list(rows[0].keys())
            else:
                evidence["columns"] = list(spec["columns"])

        items.append(
            _item(
                check_id=f"{output_key}_loadable",
                status="PASS",
                title=f"{dataset_name} {output_key} output loads cleanly",
                criterion="Every generated reference-dataset output should load according to its declared schema shape.",
                description="This is a lightweight status audit for generated dataset artifacts. It checks that the output exists and that the file shape can be read back through the maintained loader path.",
                acceptable="The file exists and loads without raising an error.",
                acceptable_basis="The dataset config is the source of truth for declared outputs, and the maintained extraction pipeline is expected to regenerate them reproducibly.",
                evidence=evidence,
            )
        )

    return AuditReport(
        audit_id="reference_dataset_status",
        title=f"Reference dataset status audit ({dataset_name})",
        items=items,
    )

