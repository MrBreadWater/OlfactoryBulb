"""Focused tests for generic notebook run metadata helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from neuroinfra.notebooks.runs import (
    RunRecord,
    list_run_dirs,
    load_run_config,
    load_run_record,
    read_json_if_present,
    resolve_run_dir,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        older = base / "2026-01-01_alpha"
        newer = base / "2026-01-02_beta"
        fallback = base / "2026-01-03_gamma"
        for path in (older, newer, fallback):
            path.mkdir(parents=True, exist_ok=True)

        _write_json(
            older / "run_info.json",
            {
                "label": "alpha",
                "timestamp": "2026-01-01T00:00:00",
                "config": {"paramset": "GammaSignature", "gaba_tau2_ms": 36.0},
                "overrides": {"gaba_tau2_ms": 36.0},
                "command": ["python", "demo.py"],
            },
        )
        _write_json(older / "summary.json", {"label": "alpha-summary"})
        (older / "stdout.txt").write_text("alpha stdout\n")
        (older / "stderr.txt").write_text("alpha stderr\n")

        _write_json(
            newer / "run_info.json",
            {
                "requested_label": "beta-requested",
                "timestamp": "2026-01-02T00:00:00",
                "config": {"paramset": "GammaSignature", "gap_mc": 32.0},
                "command": ["python", "beta.py", "--flag"],
            },
        )
        _write_json(newer / "summary.json", {"label": "beta-summary"})
        (newer / "stdout.txt").write_text("beta stdout\n")

        _write_json(fallback / "summary.json", {"timestamp": "2026-01-03T00:00:00"})

        assert read_json_if_present(base / "missing.json") is None
        ordered = list_run_dirs(results_base=base)
        assert ordered == [older, newer, fallback]
        assert list_run_dirs(prefix="2026-01-02", results_base=base) == [newer]

        assert resolve_run_dir(prefix="2026-01-02", results_base=base) == newer
        older_record = load_run_record(older, results_base=base)
        assert resolve_run_dir(older_record, results_base=base) == older
        assert isinstance(older_record, RunRecord)
        assert older_record.label == "alpha"
        assert older_record.timestamp == "2026-01-01T00:00:00"
        assert older_record.command == ["python", "demo.py"]
        assert older_record.stdout.strip() == "alpha stdout"
        assert older_record.stderr.strip() == "alpha stderr"

        newer_record = load_run_record(prefix="2026-01-02", results_base=base)
        assert newer_record.label == "beta-summary"
        assert newer_record.command == ["python", "beta.py", "--flag"]
        assert newer_record.stderr == ""

        fallback_record = load_run_record(fallback, results_base=base)
        assert fallback_record.label == fallback.name
        assert fallback_record.timestamp == "2026-01-03T00:00:00"

        cfg = load_run_config(older, results_base=base)
        assert cfg["paramset"] == "GammaSignature"
        cfg["gaba_tau2_ms"] = 99.0
        assert load_run_record(older, results_base=base).config["gaba_tau2_ms"] == 36.0

    print("neuroinfra notebook runs: OK")


if __name__ == "__main__":
    main()
