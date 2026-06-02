# Reference Validation How-To

This guide explains how to add a simulation-backed, literature-driven
validation audit using the declarative validation layer.

Use this system when you already have normalized reference rows and want to:

- run a specific stimulus protocol against one or more model cells,
- extract model-side measurements from the resulting traces,
- compare those measurements against literature rows, and
- keep protocol caveats visible in the output.

This is the validation-side companion to
[REFERENCE_DATASET_HOWTO.md](/home/michael/OlfactoryBulb/notes/REFERENCE_DATASET_HOWTO.md).

## What is modular now

The validation system is split into four parts:

1. **Reference dataset config**
   - declares and extracts literature rows
   - lives under `research_context/reference_datasets/`

2. **Reference validation config**
   - declares which protocol runner to use
   - declares which rule checks to run
   - lives under `research_context/reference_validations/`

3. **Protocol runners**
   - run the model-side experiment
   - emit metrics and protocol evidence

4. **Rule handlers**
   - consume emitted metrics and normalized reference rows
   - turn comparisons into audit items

The important design point is that `burton_urban_fi` is now just one configured
validation that uses one registered protocol runner:

- validation config:
  [burton_urban_fi.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/burton_urban_fi.validation.toml)
- protocol runner:
  `burton_urban_mctc_current_clamp`

## Where the pieces live

- Validation config template:
  [research_context/reference_validations/TEMPLATE.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/TEMPLATE.validation.toml)
- Current built-in validation config:
  [research_context/reference_validations/burton_urban_fi.validation.toml](/home/michael/OlfactoryBulb/research_context/reference_validations/burton_urban_fi.validation.toml)
- Validation config loader:
  [olfactorybulb/audit/reference_validation_config.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_config.py)
- Validation engine:
  [olfactorybulb/audit/reference_validation_engine.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_engine.py)
- Built-in protocol registry:
  [olfactorybulb/audit/reference_validation_protocols.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_protocols.py)
- Built-in rule registry:
  [olfactorybulb/audit/reference_validation_rules.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_validation_rules.py)
- Generic CLI:
  [tools/run_reference_validation.py](/home/michael/OlfactoryBulb/tools/run_reference_validation.py)

## Quick start

### List available validations

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --list-validations
```

### List registered protocols

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --list-protocols
```

### Run the built-in Burton validation

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id burton_urban_fi
```

### Smoke-test a validation without running NEURON

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_reference_validation.py --validation-id burton_urban_fi --skip-neuron
```

### Run the richer audit wrapper

Use the audit wrapper when you also want Burton-specific slice-context and
registry checks:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py burton_urban_fi
```

## Minimal validation config

Copy the template:

```bash
cp \
  research_context/reference_validations/TEMPLATE.validation.toml \
  research_context/reference_validations/my_validation.validation.toml
```

At minimum, fill in:

- `validation_id`
- `title`
- `description`
- `protocol_runner`
- at least one `[[checks]]`

Optional but useful:

- `extensions`
- `[defaults]`
- `[protocol]`
- `[skip_item]`

## How a validation is evaluated

The flow is:

1. load the validation config
2. load any extension modules named in `extensions`
3. resolve the configured `protocol_runner`
4. add the protocol-specific CLI arguments
5. run the protocol runner
6. collect model-side metrics
7. evaluate the configured rule checks
8. render a styled audit report

The protocol runner is responsible for **measurements**.
The rules are responsible for **judgment**.

That boundary matters.

If you need a new measured quantity, add it to the protocol runner output.
If you need a new decision rule, add a new rule kind.

## Built-in rule kinds

The built-in rule layer already covers common cases:

- `protocol_executed`
- `all_finite_metric`
- `all_exact_metric`
- `group_ordering`
- `group_abs_diff_max`
- `group_positive`
- `reference_band_rows`
- `note_presence`

Use config alone whenever one of these can express the paper cleanly.

If the paper needs a genuinely different comparison rule, register a new rule
kind in an extension module.

## Registering a new protocol runner

If a paper has a stimulus protocol that is not already represented, create an
extension module and point the validation config at it through `extensions`.

For example, add:

```toml
validation_id = "smith2026_intrinsic_validation"
title = "Smith 2026 intrinsic validation"
description = "Validate Example Cells against Smith 2026."
protocol_runner = "smith2026_current_clamp"
extensions = ["olfactorybulb.audit.smith2026_validation_extensions:register"]
metric_group_field = "cell_type"
```

Then create an extension module that registers the protocol:

```python
from __future__ import annotations

import argparse

from olfactorybulb.audit.reference_validation_protocols import (
    ProtocolRunResult,
    ValidationProtocolSpec,
    register_validation_protocol,
)


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell-count", type=int, default=4)
    parser.add_argument("--dt-ms", type=float, default=0.1)


def _run_protocol(args: argparse.Namespace, protocol_config: dict[str, object]) -> ProtocolRunResult:
    metrics = [
        {
            "cell_name": "Example1",
            "cell_type": "Example Cell",
            "input_resistance_MOhm": 123.4,
            "fi_gain_Hz_per_50pA": 9.8,
        }
    ]
    protocol_evidence = {
        "step_duration_ms": protocol_config.get("step_duration_ms", 1000.0),
        "current_start_pA": protocol_config.get("current_start_pA", 0.0),
        "current_stop_pA": protocol_config.get("current_stop_pA", 400.0),
        "current_step_pA": protocol_config.get("current_step_pA", 50.0),
    }
    return ProtocolRunResult(metrics=metrics, protocol_evidence=protocol_evidence, group_field="cell_type")


def register() -> None:
    register_validation_protocol(
        ValidationProtocolSpec(
            protocol_id="smith2026_current_clamp",
            title="Smith 2026 example current clamp",
            description="Example protocol registration for a literature-backed validation.",
            add_cli_args=_add_cli_args,
            run=_run_protocol,
        )
    )
```

Key point:

- the protocol runner can emit any measurement keys you want
- those keys become available to rule checks

## Adding custom measurements

Suppose the paper cares about `first_spike_latency_ms`, but none of the current
protocol runners emit it yet.

The correct place to add it is the protocol runner:

```python
metrics = [
    {
        "cell_name": "Example1",
        "cell_type": "Example Cell",
        "first_spike_latency_ms": 212.0,
    }
]
```

Once the metric exists, you have two options:

1. express the judgment using a built-in rule, if possible
2. register a new rule kind if the logic is new

## Registering a new rule kind

If the built-in checks are not enough, register a new rule kind in the same
extension module:

```python
from olfactorybulb.audit import AuditItem
from olfactorybulb.audit.reference_validation_rules import register_validation_rule


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
```

Then use it in the validation config:

```toml
[[checks]]
kind = "minimum_metric"
check_id = "example_first_spike_latency"
metric_key = "first_spike_latency_ms"
minimum = 150.0
title = "First-spike latency exceeds the lower bound"
criterion = "Explain the literature requirement here."
description = "Explain why this check matters."
acceptable = "The observed first-spike latency is at least 150 milliseconds."
acceptable_basis = "This threshold comes from the cited paper."
```

## When to add a new protocol runner versus a new rule

Add a **new protocol runner** when the paper changes:

- stimulus family
- current-step schedule
- holding potential normalization
- temperature assumptions
- trace-processing pipeline
- measured quantities

Add a **new rule kind** when the paper changes:

- comparison logic
- pass/fail decision style
- grouping semantics
- reference-band construction
- protocol-caveat resolution logic

Do not add a new protocol runner merely because the paper wants a different
ordering check or tolerance band. That belongs in config or a rule.

## Skip behavior

Use `[skip_item]` in the validation config so `--skip-neuron` still produces a
useful report.

Example:

```toml
[skip_item]
check_id = "smith2026_validation_skipped"
status = "WARN"
title = "Smith 2026 validation was skipped"
criterion = "The report should say when the expensive protocol was not run."
description = "This keeps the generic CLI informative during smoke tests."
acceptable = "The report explicitly says that no protocol-backed measurements were produced."
acceptable_basis = "This item is generated by command-line control flow."
evidence_arg_keys = ["reference_sigma_multiplier", "cell_count"]
```

## Notes and protocol caveats

If the validation involves protocol-dependent comparisons, add a `note_presence`
check so caveats remain visible.

That is what keeps differences such as:

- MC/TC protocol versus EPL fast-spiking interneuron protocol
- baseline versus modulated condition
- superficial granule cell versus deep granule cell subtype

from being buried in raw CSV text.

## Recommended workflow for a new paper

1. create or update the normalized reference dataset
2. copy `TEMPLATE.validation.toml`
3. point `protocol_runner` at an existing runner if one already matches
4. if no runner matches, write an extension module and register a new protocol
5. add checks using built-in rule kinds first
6. add a new rule kind only when config cannot express the comparison
7. add a `[skip_item]` so smoke runs stay readable
8. run the generic validation CLI
9. add a dedicated audit wrapper only if you also need repo-specific structural
   or context checks beyond the literature comparison itself

## Practical boundary

The generic validation layer is intended to make new literature-backed tests
cheap to add.

It is not intended to eliminate scientific judgment.

The paper-specific judgment should live in:

- the normalized reference rows,
- the selected protocol runner,
- the configured checks,
- and explicit notes/caveats.

That is the level where new validations stay understandable to other people in
the lab instead of turning back into one-off audit scripts.
