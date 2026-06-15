"""Declarative intrinsic validation audit for granule-cell models."""

from __future__ import annotations

import argparse

from olfactorybulb.audit.reference_validation_engine import (
    add_reference_validation_common_args,
    add_reference_validation_protocol_args,
    load_reference_validation_plan,
    run_reference_validation,
)


VALIDATION_ID = "gc_intrinsic_validation"


def _validation():
    return load_reference_validation_plan(validation_id=VALIDATION_ID)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    validation = _validation()
    add_reference_validation_common_args(parser)
    add_reference_validation_protocol_args(parser, validation=validation)


def run(args: argparse.Namespace):
    validation = _validation()
    return run_reference_validation(
        args=args,
        validation=validation,
        audit_id=VALIDATION_ID,
        title=validation.title,
    )


__all__ = ["VALIDATION_ID", "configure_parser", "run"]
