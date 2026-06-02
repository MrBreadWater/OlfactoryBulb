"""Regression checks for human-facing audit metadata completeness."""

from __future__ import annotations

import argparse

from olfactorybulb.audit.burton_urban_fi import run as run_burton_urban
from olfactorybulb.audit.cli import run_new_sweep
from olfactorybulb.audit.epl_fsi_intrinsic_validation import run as run_epl_fsi_intrinsic_validation
from olfactorybulb.audit.env_install import run as run_env_install
from olfactorybulb.audit.epli_correctness import run as run_epli_correctness
from olfactorybulb.audit.gc_intrinsic_validation import run as run_gc_intrinsic_validation
from olfactorybulb.audit.hfo_feature_contracts import run as run_hfo_feature_contracts
from olfactorybulb.audit.human_review_status import run as run_human_review_status


def _assert_human_metadata(report) -> None:
    assert report.items, report.audit_id
    for item in report.items:
        assert item.title.strip(), item.check_id
        assert item.criterion.strip(), item.check_id
        assert item.description.strip(), item.check_id
        assert item.acceptable.strip(), item.check_id
        assert item.acceptable_basis.strip(), item.check_id


env_report = run_env_install(
    argparse.Namespace(
        skip_neuron=True,
        skip_imports=True,
        require_gpu=False,
        run_launcher_smoke=False,
        import_timeout_seconds=1.0,
        launcher_timeout_seconds=1.0,
    )
)
_assert_human_metadata(env_report)

burton_report = run_burton_urban(
    argparse.Namespace(
        skip_neuron=True,
        cell_count=1,
        cell_types="MC,TC",
        use_coreneuron=False,
        use_gpu=False,
        dt_ms=0.1,
        bias_max_iterations=1,
        jobs=1,
    )
)
_assert_human_metadata(burton_report)

epli_report = run_epli_correctness(
    argparse.Namespace(
        candidate_slice=None,
        skip_neuron=True,
        reference_sigma_multiplier=2.0,
    )
)
_assert_human_metadata(epli_report)

gc_report = run_gc_intrinsic_validation(
    argparse.Namespace(
        skip_neuron=True,
        cell_models="GC1",
        use_coreneuron=False,
        use_gpu=False,
        dt_ms=0.1,
        bias_max_iterations=1,
        jobs=1,
        reference_gc_subtypes="generic_or_unspecified",
        reference_sigma_multiplier=2.0,
    )
)
_assert_human_metadata(gc_report)

epl_fsi_report = run_epl_fsi_intrinsic_validation(
    argparse.Namespace(
        skip_neuron=True,
        cell_models="SyntheticEPL2026.PVCRH_FSI1",
        use_coreneuron=False,
        use_gpu=False,
        dt_ms=0.1,
        bias_max_iterations=1,
        jobs=1,
        reference_sigma_multiplier=2.0,
    )
)
_assert_human_metadata(epl_fsi_report)

hfo_report = run_hfo_feature_contracts(argparse.Namespace())
_assert_human_metadata(hfo_report)

human_review_report = run_human_review_status(argparse.Namespace())
_assert_human_metadata(human_review_report)

new_sweep_report = run_new_sweep(["--skip-neuron", "--skip-imports"])
_assert_human_metadata(new_sweep_report)

prefixed_hfo_item = next(item for item in new_sweep_report.items if item.check_id == "hfo_feature_contracts.hfo_search_space_unique_paths")
assert prefixed_hfo_item.description == next(
    item.description for item in hfo_report.items if item.check_id == "hfo_search_space_unique_paths"
)
assert prefixed_hfo_item.acceptable == next(
    item.acceptable for item in hfo_report.items if item.check_id == "hfo_search_space_unique_paths"
)
assert prefixed_hfo_item.acceptable_basis == next(
    item.acceptable_basis for item in hfo_report.items if item.check_id == "hfo_search_space_unique_paths"
)

print("audit_style_contracts: OK")
