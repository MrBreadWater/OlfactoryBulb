"""Smoke tests for the declarative literature-validation engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

from olfactorybulb.audit.reference_validation_config import (
    list_reference_validation_ids,
    load_reference_validation_config,
    load_validation_extensions,
    validation_protocol_runner_id,
    validation_title,
)
from olfactorybulb.audit.reference_validation_protocols import get_validation_protocol_spec


assert "burton_urban_fi" in list_reference_validation_ids()

burton_config = load_reference_validation_config(validation_id="burton_urban_fi")
load_validation_extensions(burton_config)
assert validation_title(burton_config) == "Burton & Urban f-I validation audit"
assert validation_protocol_runner_id(burton_config) == "burton_urban_mctc_current_clamp"
assert get_validation_protocol_spec("burton_urban_mctc_current_clamp").title.startswith("Burton and Urban 2014")

listed_validations = subprocess.run(
    [sys.executable, "tools/run_reference_validation.py", "--list-validations"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed_validations.returncode == 0, listed_validations
assert "burton_urban_fi" in listed_validations.stdout

listed_protocols = subprocess.run(
    [sys.executable, "tools/run_reference_validation.py", "--validation-id", "burton_urban_fi", "--list-protocols"],
    capture_output=True,
    text=True,
    check=False,
)
assert listed_protocols.returncode == 0, listed_protocols
assert "burton_urban_mctc_current_clamp" in listed_protocols.stdout

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
            from olfactorybulb.audit.reference_validation_protocols import (
                ProtocolRunResult,
                ValidationProtocolSpec,
                register_validation_protocol,
            )
            from olfactorybulb.audit.reference_validation_rules import register_validation_rule


            def _add_cli_args(parser: argparse.ArgumentParser) -> None:
                parser.add_argument("--custom-score", type=float, default=3.5)


            def _run_protocol(args: argparse.Namespace, protocol_config: dict[str, object]) -> ProtocolRunResult:
                return ProtocolRunResult(
                    metrics=[
                        {
                            "cell_name": "Example1",
                            "cell_type": "Example Cell",
                            "custom_score": float(args.custom_score),
                        }
                    ],
                    protocol_evidence={
                        "step_duration_ms": protocol_config.get("step_duration_ms", 1000.0),
                        "protocol_label": protocol_config.get("protocol_label", "temporary protocol"),
                    },
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

    temp_config = load_reference_validation_config(path=config_path)
    sys.path.insert(0, tmpdir)
    try:
        load_validation_extensions(temp_config)
        temp_spec = get_validation_protocol_spec("temp_custom_protocol")
        assert temp_spec.title == "Temporary custom protocol"

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
