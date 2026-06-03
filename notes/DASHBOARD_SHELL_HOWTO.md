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
- audit cards render explicit item visuals declared by the audit/rule layer
  rather than inferring them from array-shaped evidence alone
- the standard series backend is Matplotlib; use explicit visual specs when
  registering a new plotted audit item

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
- starts with the audit tab idle; no audit is run until you ask for one
- defaults the audit runner selection to `all`, matching `python tools/run_audit.py`
- opens on the audit tab first
- serves immediately at `http://127.0.0.1:6006/` by default
- prints startup progress to the terminal while the audit and optimization
  views are rendering in the background
- keeps the optimization tab rendering in the background while the shell itself
  is already usable
- exposes an on-page audit runner so you can switch to any registered audit and
  pass explicit arguments without restarting the shell
- does not regenerate missing optimization packets during default startup; use
  explicit controls/flags when you want packet generation work
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

Run the selected audit immediately on startup if you explicitly want that:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center serve \
  --run-audit-on-start \
  --audit-id repo_health -- --profile maintained
```

Once the shell is open:

- use the `Audit id` selector to choose any registered audit
- the selector also exposes:
  - `default` -> maintained repo-health profile
  - `all` -> full registered audit sweep
- use `Audit arguments` for audit-specific flags such as:
  - `--profile maintained`
  - `--suite reference_bundles --details`
  - `--dataset-id granule_cells`
- each audit run stays visible as its own group in the audit tab for the life
  of the current shell session; running another audit should not wipe the
  earlier groups from the page
- the shell state endpoint keeps the audit/optimization/doc badges in sync with
  the real rendered content
- the audit-runner form keeps local edits stable across background state polls
  instead of snapping back to the previously running audit
- when an audit's arguments are only the audit's own implicit defaults, the
  audit-arguments field stays blank rather than force-filling those defaults
- the audit runner lives only inside the audit tab; module status and progress
  live in the shell banner instead of in a duplicated global toolbar
- after a run, the audit runner collapses into a compact summary with
  `Edit selection`; the empty-state `Run selected audit` button posts back to
  the shell so the audit tab can launch a run without another global launcher
- the audit page itself now provides display controls for:
  - search
  - failures/warnings-only filtering
  - hiding fully passing groups
  - hiding detail items
  - expanding/collapsing groups

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
- The served control center exposes maintained live endpoints:
  - `GET /__control_center_state__`
  - `POST /__audit_run__`
  - `POST /__audit_refresh__`
  - use those when diagnosing stale badges, permanent loading states, or rerun
    failures.
