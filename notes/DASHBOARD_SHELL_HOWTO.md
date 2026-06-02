# Dashboard Shell HOWTO

This repo now has one maintained web shell for the user-facing dashboards:

- audits
- optimization
- docs

The shell is implemented in two layers:

- shared neuroinfra shell renderer:
  - `neuroinfra/dashboard/shell.py`
- repo-specific control center:
  - `olfactorybulb/dashboard/control_center.py`

## What belongs where

`neuroinfra/dashboard/shell.py`
- generic tabbed iframe shell
- shared visual style for maintained dashboard surfaces
- no OlfactoryBulb-specific assumptions

`olfactorybulb/dashboard/control_center.py`
- mounts the current repo modules into one shell
- exports:
  - `/audits/`
  - `/optimization/`
  - `/docs/`
- owns the repo-specific packet-generation and audit-refresh endpoints

`olfactorybulb/audit/dashboard.py`
- renders maintained audit reports to:
  - `index.html`
  - `report.json`
  - `manifest.json`

## Canonical commands

Use the OBGPU environment.

Fastest maintained launch:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center
```

That command:

- defaults to `serve`
- auto-detects the active optimization campaign from the maintained status file
  or optimization results tree
- runs the default maintained audit (`repo_health --profile maintained`)
- opens on the audit tab first
- serves immediately at `http://127.0.0.1:6006/` by default
- prints startup progress to the terminal while the audit and optimization
  views are rendering in the background
- shows loading placeholders in the audit/optimization tabs until those views
  are ready
- if `6006` is already in use, it automatically picks the next available local
  port and prints the actual URL

Open the page directly:

```text
http://127.0.0.1:6006/
```

Open it automatically in a local browser:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center --open-browser
```

Export the unified shell:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center export
```

Serve the unified shell locally:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center serve
```

Serve with a lighter audit target:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center serve \
  --audit-id repo_health
```

Serve a specific campaign explicitly:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center serve \
  results/notebook_runs/optimization/codex_big_hfo_logs
```

Render only the audit dashboard:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.dashboard burton_urban_fi --output-dir /tmp/burton_audit
python -m olfactorybulb.audit.dashboard new_sweep --output-dir /tmp/full_audit -- --skip-neuron
```

## Design rules

- The audit framework is the maintained presentation layer for audit results.
- Low-level tests stay under `tests/`; they may be surfaced through grouped audits.
- The control center should mount maintained modules, not duplicate their logic.
- New dashboard modules should reuse the shared shell instead of creating another standalone visual style.
- If a module needs custom server behavior, keep the generic shell clean and put repo-specific routing in the repo-level control center.

## Notes

- The optimization tab reuses the existing HFO visual dashboard export path.
- The audit tab uses grouped audit JSON from `AuditReport.to_dict()`.
- The docs tab serves the maintained docs portal under `docs/`.
