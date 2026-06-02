"""Smoke tests for the curated repo-health audit surface."""

from __future__ import annotations

from olfactorybulb.audit.repo_health import list_repo_health_profiles, repo_health_checks


profiles = list_repo_health_profiles()
assert profiles == ("quick", "maintained", "reference", "full")

quick_checks = repo_health_checks("quick")
assert len(quick_checks) == 1
assert quick_checks[0].check_id == "env_install"
assert "--run-launcher-smoke" in quick_checks[0].command

maintained_checks = repo_health_checks("maintained")
maintained_ids = {check.check_id for check in maintained_checks}
assert "human_review_status" in maintained_ids
assert "hfo_feature_contracts" in maintained_ids
assert "config_helpers" in maintained_ids
assert "reference_validation_engine" in maintained_ids
assert "reference_data_sanity" in maintained_ids

reference_checks = repo_health_checks("reference")
reference_ids = {check.check_id for check in reference_checks}
assert "epl_fsi_reference_data" in reference_ids
assert "gc_reference_data" in reference_ids
assert "verify_epl_fsi_reference_data" in reference_ids
assert "verify_gc_reference_data" in reference_ids

full_checks = repo_health_checks("full")
assert len(full_checks) == len(maintained_checks) + len(reference_checks)

print("repo health profile smoke tests: OK")
