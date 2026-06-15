"""Verify the SciUnit / NeuronUnit overhaul import surface in OBGPU.

This is intentionally narrower than the main OBGPU import audit. It exists so
the overhaul branch can gate the fork-side scientific-core work without
pretending the entire repo has already migrated.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


THIRD_PARTY_IMPORTS = [
    "quantities",
    "neo",
    "sciunit",
    "neuronunit",
]

REPO_IMPORTS = [
    "olfactorybulb.audit.protocol_evidence",
    "olfactorybulb.audit.reference_validation_document",
    "olfactorybulb.audit.reference_validation_plan",
    "olfactorybulb.audit.reference_validation_specs",
    "olfactorybulb.audit.reference_validation_engine",
    "olfactorybulb.neuronunit.capabilities",
    "olfactorybulb.neuronunit.provenance",
    "olfactorybulb.neuronunit.tests.publications",
    "olfactorybulb.neuronunit.tests.tests",
    "olfactorybulb.neuronunit.models.neuron_cell",
    "olfactorybulb.neuronunit.reference_bands",
    "olfactorybulb.neuronunit.comparison_validation_suite",
    "olfactorybulb.neuronunit.series_validation_suite",
    "olfactorybulb.neuronunit.suite_scores",
    "olfactorybulb.neuronunit.suite_presentation",
    "olfactorybulb.neuronunit.summary_validation_suite",
    "olfactorybulb.neuronunit.reference_validation_suite",
    "olfactorybulb.audit.reference_validation_rules",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to validate.",
    )
    return parser.parse_args()


def import_module(name: str) -> None:
    importlib.import_module(name)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    failures: list[dict[str, str]] = []
    for name in THIRD_PARTY_IMPORTS:
        try:
            import_module(name)
        except Exception as exc:
            failures.append({"kind": "third_party", "target": name, "error": repr(exc)})
    for name in REPO_IMPORTS:
        try:
            import_module(name)
        except Exception as exc:
            failures.append({"kind": "repo", "target": name, "error": repr(exc)})

    if failures:
        print(json.dumps({"ok": False, "failures": failures}, indent=2))
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "ok": True,
                "third_party_checked": THIRD_PARTY_IMPORTS,
                "repo_checked": REPO_IMPORTS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
