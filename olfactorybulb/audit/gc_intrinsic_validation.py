"""Declarative intrinsic validation audit for granule-cell models."""

from __future__ import annotations

import argparse

from olfactorybulb.audit.reference_validation_config import load_reference_validation_config, validation_title
from olfactorybulb.audit.reference_validation_engine import (
    add_reference_validation_common_args,
    add_reference_validation_protocol_args,
    run_reference_validation,
)


VALIDATION_ID = "gc_intrinsic_validation"


def _config():
    return load_reference_validation_config(validation_id=VALIDATION_ID)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    config = _config()
    add_reference_validation_common_args(parser)
    add_reference_validation_protocol_args(parser, config=config)


def run(args: argparse.Namespace):
    config = _config()
    return run_reference_validation(
        args=args,
        config=config,
        audit_id=VALIDATION_ID,
        title=validation_title(config),
    )


__all__ = ["VALIDATION_ID", "configure_parser", "run"]
