"""Smoke tests for the declarative literature-validation engine."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

from olfactorybulb.audit.reference_validation_config import (
    list_reference_validation_ids,
    load_validation_extensions,
)
from olfactorybulb.audit.reference_validation_document import load_reference_validation_document
from olfactorybulb.audit.reference_validation_engine import run_reference_validation
from olfactorybulb.audit.reference_validation_plan import load_reference_validation_plan
from olfactorybulb.audit.reference_validation_protocols import (
    clear_protocol_execution_cache,
    get_validation_protocol_spec,
    protocol_execution_cache_size,
)
from olfactorybulb.audit.protocol_evidence import ProtocolEvidenceBundle, intrinsic_fi_curve_series_spec
from olfactorybulb.audit.reference_validation_specs import (
    NotePresenceRuleSpec,
    ProtocolExecutedRuleSpec,
)
from olfactorybulb.audit.reference_validation_rules import (
    CustomSingleRuleDispatch,
    GroupedRuleDispatch,
    NotePresenceRuleDispatch,
    ProtocolExecutedRuleDispatch,
)


assert "burton_urban_fi" in list_reference_validation_ids()
assert "gc_intrinsic_validation" in list_reference_validation_ids()
assert "epl_fsi_intrinsic_validation" in list_reference_validation_ids()
assert "epli_correctness" in list_reference_validation_ids()
assert "TEMPLATE" not in list_reference_validation_ids()

burton_document = load_reference_validation_document(validation_id="burton_urban_fi")
load_validation_extensions(burton_document.extension_specs)
assert burton_document.validation_id == "burton_urban_fi"
assert burton_document.title == "Burton & Urban f-I validation audit"
assert burton_document.protocol_runner_id == "burton_urban_mctc_current_clamp"
assert burton_document.design_review_defaults.status == "pending"
assert burton_document.skip_item is not None
assert burton_document.skip_item.check_id == "burton_urban_fi_skipped"
assert burton_document.rules[0]["kind"] == "note_presence"
assert burton_document.extension_specs == ()
assert get_validation_protocol_spec("burton_urban_mctc_current_clamp").title.startswith("Burton and Urban 2014")
burton_plan = load_reference_validation_plan(validation_id="burton_urban_fi")
assert burton_plan.validation_id == "burton_urban_fi"
assert burton_plan.title == "Burton & Urban f-I validation audit"
assert burton_plan.protocol_runner_id == "burton_urban_mctc_current_clamp"
assert burton_plan.protocol_spec.title.startswith("Burton and Urban 2014")
assert burton_plan.skip_neuron_mode == "short_circuit"
assert burton_plan.design_review_defaults.status == "pending"
assert burton_plan.skip_item is not None
assert burton_plan.skip_item.check_id == "burton_urban_fi_skipped"
assert isinstance(burton_plan.rule_dispatches[0], NotePresenceRuleDispatch)
assert any(isinstance(entry, GroupedRuleDispatch) for entry in burton_plan.rule_dispatches)

listed_validations = subprocess.run(
    [sys.executable, "tools/run_reference_validation.py", "--list-validations"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed_validations.returncode == 0, listed_validations
assert "burton_urban_fi" in listed_validations.stdout
assert "gc_intrinsic_validation" in listed_validations.stdout
assert "epl_fsi_intrinsic_validation" in listed_validations.stdout
assert "epli_correctness" in listed_validations.stdout
assert "TEMPLATE" not in listed_validations.stdout

protocol_rule_spec = ProtocolExecutedRuleSpec.from_rule({})
assert protocol_rule_spec.fallback_series_key == "fi_curve_rows"
protocol_rule_spec = ProtocolExecutedRuleSpec.from_rule({"fallback_series_key": "custom_rows"})
assert protocol_rule_spec.fallback_series_key == "custom_rows"

note_rule_spec = NotePresenceRuleSpec.from_rule(
    {
        "scope": "gc",
        "row_contexts": [
            {
                "loader": "csv:synthetic.csv",
                "as_protocol_context": True,
                "property_name": "Synthetic Protocol",
                "filter_field": "protocol_id",
                "filter_value_arg": "protocol_id",
                "filters": [{"field": "sample_scope", "value": "example_cell"}],
            }
        ],
        "synthetic_contexts": [{"protocol_id": "SYNTH", "Property": "Synthetic Protocol"}],
    }
)
assert note_rule_spec.scope == "gc"
assert len(note_rule_spec.row_contexts) == 1
assert note_rule_spec.row_contexts[0].loader == "csv:synthetic.csv"
assert note_rule_spec.row_contexts[0].as_protocol_context is True
assert note_rule_spec.row_contexts[0].property_name == "Synthetic Protocol"
assert note_rule_spec.row_contexts[0].to_filter_spec() == {
    "loader": "csv:synthetic.csv",
    "filter_field": "protocol_id",
    "filter_value_arg": "protocol_id",
    "filters": [{"field": "sample_scope", "value": "example_cell"}],
}
assert note_rule_spec.synthetic_contexts == ({"protocol_id": "SYNTH", "Property": "Synthetic Protocol"},)

listed_protocols = subprocess.run(
    [sys.executable, "tools/run_reference_validation.py", "--validation-id", "burton_urban_fi", "--list-protocols"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed_protocols.returncode == 0, listed_protocols
assert "burton_urban_mctc_current_clamp" in listed_protocols.stdout
assert "gc_intrinsic_current_clamp" in listed_protocols.stdout
assert "epl_fsi_current_clamp" in listed_protocols.stdout
assert "epli_correctness_structural" in listed_protocols.stdout

skip = subprocess.run(
    [sys.executable, "tools/run_reference_validation.py", "--validation-id", "burton_urban_fi", "--skip-neuron", "--jobs", "4", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert skip.returncode == 0, skip
skip_payload = json.loads(skip.stdout)
skip_items = {item["check_id"]: item for item in skip_payload["items"]}
assert skip_payload["audit_id"] == "burton_urban_fi"
assert skip_payload["summary"]["WARN"] == 1
assert skip_items["burton_urban_fi_skipped"]["status"] == "WARN"
assert skip_items["burton_urban_fi_skipped"]["evidence"]["jobs"] == 4
assert skip_items["burton_urban_fi_skipped"]["evidence"]["reference_sigma_multiplier"] == 2.0


with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    extension_path = tmpdir_path / "temp_validation_extension.py"
    extension_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import argparse

            from olfactorybulb.audit import AuditItem
            from olfactorybulb.audit.protocol_evidence import ProtocolEvidenceBundle, intrinsic_fi_curve_series_spec
            from olfactorybulb.audit.reference_validation_protocols import (
                ProtocolRunResult,
                ValidationProtocolSpec,
                register_validation_protocol,
            )
            from olfactorybulb.audit.reference_validation_rules import register_validation_rule

            RUN_COUNT = 0


            def _add_cli_args(parser: argparse.ArgumentParser) -> None:
                parser.add_argument("--custom-score", type=float, default=3.5)


            def _run_protocol(args: argparse.Namespace, protocol_config: dict[str, object]) -> ProtocolRunResult:
                global RUN_COUNT
                RUN_COUNT += 1
                return ProtocolRunResult(
                    metrics=[
                        {
                            "cell_name": "Example1",
                            "cell_type": "Example Cell",
                            "custom_score": float(args.custom_score),
                        }
                    ],
                    protocol_evidence=ProtocolEvidenceBundle(
                        values={
                            "step_duration_ms": protocol_config.get("step_duration_ms", 1000.0),
                            "protocol_label": protocol_config.get("protocol_label", "temporary protocol"),
                            "fi_curve_rows": [
                                {"current_pA": 0.0, "firing_rate_Hz": 0.0},
                                {"current_pA": 50.0, "firing_rate_Hz": 3.0},
                            ],
                        },
                        series_specs=(intrinsic_fi_curve_series_spec(),),
                    ),
                    group_field="cell_type",
                )


            def register() -> None:
                @register_validation_rule("minimum_metric")
                def _minimum_metric(rule, context):
                    metric_key = str(rule["metric_key"])
                    minimum = float(rule["minimum"])
                    observed = float(context.metrics[0][metric_key])
                    status = "PASS" if observed >= minimum else "FAIL"
                    return [
                        AuditItem(
                            check_id=str(rule["check_id"]),
                            status=status,
                            title=str(rule["title"]),
                            criterion=str(rule["criterion"]),
                            description=str(rule["description"]),
                            acceptable=str(rule["acceptable"]),
                            acceptable_basis=str(rule["acceptable_basis"]),
                            evidence={"observed": observed, "minimum": minimum},
                        )
                    ]

                register_validation_protocol(
                    ValidationProtocolSpec(
                        protocol_id="temp_custom_protocol",
                        title="Temporary custom protocol",
                        description="Extension-registered protocol used by the validation-engine smoke test.",
                        add_cli_args=_add_cli_args,
                        run=_run_protocol,
                        cache_enabled=True,
                        cache_arg_names=("custom_score",),
                    )
                )
            """
        )
    )
    config_path = tmpdir_path / "temp.validation.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            validation_id = "temp_validation"
            title = "Temporary validation"
            description = "Temporary extension-driven validation."
            protocol_runner = "temp_custom_protocol"
            extensions = ["temp_validation_extension:register"]
            metric_group_field = "cell_type"

            [defaults]
            reference_sigma_multiplier = 2.0

            [protocol]
            step_duration_ms = 750.0
            protocol_label = "temporary protocol"

            [skip_item]
            check_id = "temp_validation_skipped"
            status = "WARN"
            title = "Temporary validation skipped"
            criterion = "The report should say when the temporary protocol was not run."
            description = "This verifies that config-driven skip items work in the generic validation CLI."
            acceptable = "The report explicitly says that no protocol-backed measurements were produced."
            acceptable_basis = "This item is generated by command-line control flow."
            evidence_arg_keys = ["custom_score", "reference_sigma_multiplier"]

            [[checks]]
            kind = "protocol_executed"
            check_id = "temp_protocol_executed"
            title = "Temporary protocol executed"
            criterion = "The temporary protocol should run and emit at least one metric row."
            description = "This is the top-level execution sanity check for the extension-loaded protocol."
            acceptable = "At least one metric row is produced."
            acceptable_basis = "This is an implementation sanity check."

            [[checks]]
            kind = "minimum_metric"
            check_id = "custom_score_high_enough"
            metric_key = "custom_score"
            minimum = 2.0
            title = "Custom score exceeds the configured lower bound"
            criterion = "The custom score should be at least two."
            description = "This proves that an extension module can define and use a brand-new rule kind."
            acceptable = "The observed custom score is at least two."
            acceptable_basis = "The threshold is declared directly in the validation config."
            """
        )
    )

    sys.path.insert(0, tmpdir)
    try:
        temp_document = load_reference_validation_document(path=config_path)
        load_validation_extensions(temp_document.extension_specs)
        temp_module = importlib.import_module("temp_validation_extension")
        temp_spec = get_validation_protocol_spec("temp_custom_protocol")
        assert temp_spec.title == "Temporary custom protocol"
        assert temp_spec.cache_enabled is True
        assert temp_spec.cache_arg_names == ("custom_score",)
        temp_plan = load_reference_validation_plan(path=config_path)
        assert temp_document.title == "Temporary validation"
        assert temp_document.protocol_runner_id == "temp_custom_protocol"
        assert temp_document.skip_item is not None
        assert temp_document.skip_item.check_id == "temp_validation_skipped"
        assert temp_document.extension_specs == ("temp_validation_extension:register",)
        assert temp_plan.title == "Temporary validation"
        assert temp_plan.protocol_spec.title == "Temporary custom protocol"
        assert isinstance(temp_plan.rule_dispatches[0], ProtocolExecutedRuleDispatch)
        assert isinstance(temp_plan.rule_dispatches[1], CustomSingleRuleDispatch)
        assert temp_plan.skip_item is not None
        first_protocol_result = temp_plan.run_protocol(argparse.Namespace(custom_score=4.5))
        assert isinstance(first_protocol_result.protocol_evidence, ProtocolEvidenceBundle)
        assert first_protocol_result.protocol_evidence.to_dict()["protocol_label"] == "temporary protocol"
        assert first_protocol_result.protocol_evidence.series_specs[0].evidence_key == "fi_curve_rows"
        clear_protocol_execution_cache()
        temp_module.RUN_COUNT = 0
        skip_item = temp_plan.build_skip_item(args=argparse.Namespace(custom_score=4.5, reference_sigma_multiplier=2.0))
        assert skip_item is not None
        assert skip_item.check_id == "temp_validation_skipped"
        assert skip_item.evidence["custom_score"] == 4.5
        assert skip_item.evidence["reference_sigma_multiplier"] == 2.0
        clear_protocol_execution_cache()
        assert protocol_execution_cache_size() == 0
        assert temp_module.RUN_COUNT == 0

        same_process_report = run_reference_validation(
            args=argparse.Namespace(skip_neuron=False, custom_score=4.5, reference_sigma_multiplier=2.0),
            validation=temp_plan,
            audit_id=temp_plan.validation_id,
            title=temp_plan.title,
        )
        same_process_items = {item.check_id: item for item in same_process_report.items}
        assert temp_module.RUN_COUNT == 1
        assert protocol_execution_cache_size() == 1
        assert same_process_items["temp_protocol_executed"].evidence["protocol_cache"]["status"] == "miss"
        assert same_process_items["temp_protocol_executed"].evidence["protocol_cache"]["arg_values"]["custom_score"] == 4.5

        cached_report = run_reference_validation(
            args=argparse.Namespace(skip_neuron=False, custom_score=4.5, reference_sigma_multiplier=2.0),
            validation=temp_plan,
            audit_id=temp_plan.validation_id,
            title=temp_plan.title,
        )
        cached_items = {item.check_id: item for item in cached_report.items}
        assert temp_module.RUN_COUNT == 1
        assert protocol_execution_cache_size() == 1
        assert cached_items["temp_protocol_executed"].evidence["protocol_cache"]["status"] == "hit"

        changed_arg_report = run_reference_validation(
            args=argparse.Namespace(skip_neuron=False, custom_score=6.5, reference_sigma_multiplier=2.0),
            validation=temp_plan,
            audit_id=temp_plan.validation_id,
            title=temp_plan.title,
        )
        changed_arg_items = {item.check_id: item for item in changed_arg_report.items}
        assert temp_module.RUN_COUNT == 2
        assert protocol_execution_cache_size() == 2
        assert changed_arg_items["temp_protocol_executed"].evidence["protocol_cache"]["status"] == "miss"
        assert changed_arg_items["custom_score_high_enough"].evidence["observed"] == 6.5

        env = os.environ.copy()
        env["PYTHONPATH"] = tmpdir if not env.get("PYTHONPATH") else f"{tmpdir}:{env['PYTHONPATH']}"
        temp_run = subprocess.run(
            [
                sys.executable,
                "tools/run_reference_validation.py",
                "--config-path",
                str(config_path),
                "--custom-score",
                "4.5",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert temp_run.returncode == 0, temp_run
        temp_payload = json.loads(temp_run.stdout)
        temp_items = {item["check_id"]: item for item in temp_payload["items"]}
        assert temp_payload["audit_id"] == "temp_validation"
        assert temp_items["temp_protocol_executed"]["status"] == "PASS"
        assert temp_items["temp_protocol_executed"]["series_visuals"][0]["keys"] == ["fi_curve_rows"]
        assert temp_items["custom_score_high_enough"]["status"] == "PASS"
        assert temp_items["custom_score_high_enough"]["evidence"]["observed"] == 4.5

        temp_skip = subprocess.run(
            [
                sys.executable,
                "tools/run_reference_validation.py",
                "--config-path",
                str(config_path),
                "--skip-neuron",
                "--custom-score",
                "4.5",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert temp_skip.returncode == 0, temp_skip
        temp_skip_payload = json.loads(temp_skip.stdout)
        temp_skip_items = {item["check_id"]: item for item in temp_skip_payload["items"]}
        assert temp_skip_payload["summary"]["WARN"] == 1
        assert temp_skip_items["temp_validation_skipped"]["evidence"]["custom_score"] == 4.5
    finally:
        sys.path.pop(0)

print("reference_validation_engine: OK")
