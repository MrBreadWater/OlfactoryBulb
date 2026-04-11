"""Verify that the maintained OBGPU Python surface imports cleanly.

This is intentionally scoped to the supported OBGPU workflow:

- setup/build helpers
- benchmark runner
- notebook helper module
- active ``olfactorybulb`` runtime modules

It does not try to validate Blender-only code paths or the older neuronunit
stack, which have separate dependencies and are not part of the Sol OBGPU
target.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path


THIRD_PARTY_IMPORTS = [
    "numpy",
    "scipy",
    "pywt",
    "sympy",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.animation",
    "mpi4py.MPI",
    "peewee",
    "lmfit",
    "requests",
    "pexpect",
    "validators",
    "git",
    "networkx",
    "tables",
    "lxml",
    "lxml.etree",
    "cerberus",
    "execnet",
    "blenderneuron",
    "LFPsimpy",
    "natsort",
    "neuron",
]

REPO_IMPORTS = [
    "olfactorybulb.output_paths",
    "olfactorybulb.database",
    "olfactorybulb.paramsets.base",
    "olfactorybulb.paramsets.case_studies",
    "olfactorybulb.paramsets.sensitivity",
    "olfactorybulb.model",
    "modify_model",
    "obgpu_experiment_helpers",
]


def import_module(name: str) -> None:
    """Import one module while tolerating warnings from optional preload paths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importlib.import_module(name)


def import_module_from_path(name: str, path: Path) -> None:
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to validate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the import verification and exit non-zero on failure."""
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    failures: list[dict[str, str]] = []

    for name in THIRD_PARTY_IMPORTS:
        try:
            import_module(name)
        except Exception as exc:  # pragma: no cover - exercised in setup workflows
            failures.append({"kind": "third_party", "target": name, "error": repr(exc)})

    for name in REPO_IMPORTS:
        try:
            import_module(name)
        except Exception as exc:  # pragma: no cover - exercised in setup workflows
            failures.append({"kind": "repo", "target": name, "error": repr(exc)})

    benchmark_path = repo_root / "tools" / "benchmarks" / "benchmark_ob.py"
    try:
        import_module_from_path("benchmark_ob", benchmark_path)
    except Exception as exc:  # pragma: no cover - exercised in setup workflows
        failures.append({"kind": "repo_file", "target": str(benchmark_path), "error": repr(exc)})

    if failures:
        print(json.dumps({"ok": False, "failures": failures}, indent=2))
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "ok": True,
                "third_party_checked": THIRD_PARTY_IMPORTS,
                "repo_checked": REPO_IMPORTS,
                "repo_file_checked": str(benchmark_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
