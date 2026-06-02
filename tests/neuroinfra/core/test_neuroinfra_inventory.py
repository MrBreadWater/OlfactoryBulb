"""Smoke checks for the internal neuroinfra extraction inventory."""

from __future__ import annotations

from pathlib import Path

from neuroinfra.inventory import EXTRACTION_CANDIDATES, REPO_ROOT, inventory_rows, target_module_index


def main() -> None:
    assert EXTRACTION_CANDIDATES, "expected at least one extraction candidate"

    keys = [candidate.key for candidate in EXTRACTION_CANDIDATES]
    assert len(keys) == len(set(keys)), "candidate keys must be unique"

    target_modules = [candidate.target_module for candidate in EXTRACTION_CANDIDATES]
    assert all(target_modules), "every candidate must declare a target module"

    rows = inventory_rows()
    assert len(rows) == len(EXTRACTION_CANDIDATES)

    for candidate, row in zip(EXTRACTION_CANDIDATES, rows, strict=True):
        assert row["key"] == candidate.key
        assert int(row["proposed_phase"]) >= 1
        path_status = row["source_paths_exist"]
        assert isinstance(path_status, dict)
        assert path_status, f"{candidate.key} should expose source paths"
        for rel_path, exists in path_status.items():
            assert exists is True, f"missing source path in inventory: {rel_path}"
            assert (REPO_ROOT / rel_path).exists()

    by_target = target_module_index()
    assert by_target, "expected target-module index to be nonempty"
    for target_module, candidates in by_target.items():
        assert target_module
        assert candidates, f"empty target-module group for {target_module}"

    assert (REPO_ROOT / "neuroinfra" / "README.md").exists()
    assert (REPO_ROOT / "notes" / "REUSABLE_INFRASTRUCTURE_EXTRACTION_MAP_2026-06-01.md").exists()
    print("neuroinfra inventory smoke test: OK")


if __name__ == "__main__":
    main()
