# Tests

The maintained Python test surface now lives under `tests/`, not at repo root.

## Structure

- `tests/audit/`
  - audit CLI/output/meta-audit smoke tests
- `tests/reference/`
  - reference-data engine, downloader, extraction, and sanity tests
- `tests/integration/`
  - broad maintained-facade smoke tests such as `obgpu_experiment_helpers.py`
- `tests/neuroinfra/`
  - extracted reusable infrastructure tests, grouped by domain
- `tests/olfactorybulb/`
  - package-level notebook/analysis tests
- `tests/hfo/`
  - high-frequency-oscillation optimizer and dashboard tests
- `tests/slice/`
  - slice-builder and connectivity tests
- `tests/cell_models/`
  - cell-family/runtime helper tests
- `tests/simulation/`
  - CoreNEURON / shared-gid simulation regression tests

## Execution rules

- Canonical direct execution uses module form:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m tests.reference.test_reference_data_sanity
python -m tests.audit.test_repo_health
```

- Use the audit framework when you want grouped, user-facing presentation:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/run_audit.py test_suite_status --list-suites
python tools/run_audit.py test_suite_status --suite maintained_core
python tools/run_audit.py test_suite_status --suite audit_surface
python tools/run_audit.py test_suite_status --suite reference_bundles --details
```

- The broader maintained web presentation layer lives above the raw tests:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.dashboard new_sweep --output-dir /tmp/full_audit -- --skip-neuron
python -m olfactorybulb.dashboard.control_center
```

- The control-center command should print a reachable URL immediately and keep
  the heavy audit/optimization rendering in the background. Test that real
  serve path when changing dashboard startup behavior.

- `repo_health` should call grouped suite audits for maintained test categories.
  Keep low-level developer tests under `tests/` rather than promoting every
  single file into its own first-class audit.

## Anti-sprawl rules

- Do not add new root-level `test_*.py` files.
- If a new check is:
  - developer-facing and low-level, put it under `tests/`
  - user-facing and operational, add or extend an audit
  - best represented as many low-level checks plus one high-level status
    surface, keep the raw tests under `tests/` and expose them through a
    grouped suite audit
