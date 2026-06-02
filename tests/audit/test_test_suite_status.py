"""Smoke tests for the structured test-suite audit registry."""

from __future__ import annotations

from olfactorybulb.audit.test_suite_status import get_test_suite, list_test_suites


suite_ids = list_test_suites()
assert suite_ids == ("maintained_core", "reference_bundles", "audit_surface")

maintained = get_test_suite("maintained_core")
assert maintained.title
assert len(maintained.modules) == 3
assert maintained.modules[0].module_name == "tests.integration.test_config_helpers"

reference = get_test_suite("reference_bundles")
assert any(module.module_name == "tests.reference.test_gc_reference_data" for module in reference.modules)

audit_surface = get_test_suite("audit_surface")
assert any(module.module_name == "tests.audit.test_audit_cli_output" for module in audit_surface.modules)

print("structured test-suite audit smoke tests: OK")
