"""Smoke tests for extracted generic campaign-store helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from neuroinfra.campaigns.store import (
    append_jsonl,
    archive_path,
    batch_artifact_path,
    batch_index_from_name,
    ensure_campaign_dir,
    read_json,
    safe_campaign_slug,
    state_path,
    write_json,
)
from olfactorybulb.hfo_optimizer import ensure_campaign_dir as ensure_hfo_campaign_dir


def main() -> None:
    assert safe_campaign_slug(" Test Campaign / 01 ") == "Test_Campaign___01"
    assert batch_index_from_name("batch_0012") == 12
    assert batch_index_from_name("not_a_batch") is None

    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        campaign_dir = ensure_campaign_dir("My Campaign", base_dir=tmp)
        assert campaign_dir == tmp / "My_Campaign"
        assert (campaign_dir / "batches").exists()
        assert state_path(campaign_dir) == campaign_dir / "state.json"
        assert archive_path(campaign_dir, kind="candidate") == campaign_dir / "candidate_archive.jsonl"
        assert batch_artifact_path(campaign_dir, "batch_0001", "plan") == campaign_dir / "batches" / "batch_0001_plan.json"

        payload = {"a": 1, "b": {"c": 2}}
        write_json(state_path(campaign_dir), payload)
        assert read_json(state_path(campaign_dir), {}) == payload
        assert read_json(campaign_dir / "missing.json", {"fallback": True}) == {"fallback": True}

        append_jsonl(
            archive_path(campaign_dir, kind="candidate"),
            [{"row": 1}, {"row": 2}],
        )
        lines = (campaign_dir / "candidate_archive.jsonl").read_text().splitlines()
        assert json.loads(lines[0]) == {"row": 1}
        assert json.loads(lines[1]) == {"row": 2}

        hfo_campaign_dir = ensure_hfo_campaign_dir("My Campaign", base_dir=tmp)
        assert hfo_campaign_dir == campaign_dir

    print("neuroinfra campaign store smoke test: OK")


if __name__ == "__main__":
    main()
