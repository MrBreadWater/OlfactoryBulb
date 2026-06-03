# Suboptimal Review — 2026-06-03

## Scope

- Focus: maintained active surface and extraction seams with highest-confidence reliability risks.
- Objective: document review findings with evidence-backed severity and concrete follow-ups.

## Findings

### 1) High — hard-coded remote defaults in manual allocation config builder

- Location: `olfactorybulb/hfo_optimizer.py:118-122`
- Evidence:
  - `remote_host = str(template.get("remote_host") or "jmpaniag@localhost")`
  - `remote_repo_root = str(template.get("remote_repo_root") or "/home/jmpaniag/OlfactoryBulb")`
- Why suboptimal:
  - These user-specific fallbacks are not portable and can silently target a non-existent host/path in non-JMP sessions.
- Recommendation:
  - Replace user-specific literals with repo-relative defaults (via `_default_repo_root`) and fail fast with clear errors when required remote identity cannot be resolved.

### 2) Medium — malformed batch-plan parsing is silently downgraded to legacy defaults

- Location: `tools/run_hfo_campaign.py:111-121`
- Evidence:
  - `_recent_batch_shape()` catches all JSON errors and continues scanning, returning `(8, "refine")` when nothing parses.
- Why suboptimal:
  - A broken or truncated latest plan can silently force an incorrect proposal schedule without surfacing a hard failure.
- Recommendation:
  - Record the parse error, preserve the last valid batch shape when possible, and make corruption an explicit warning in resume output.

### 3) Medium — silent JSON-fallbacks in remote sweep polling can hide corruption

- Location: `neuroinfra/remote_script_polling.py:150-158` and `neuroinfra/remote_script_sweeps.py:166-169`
- Evidence:
  - `read_json_file()` returns `default` for any read/decode error.
  - `resolve_completed_result_dir()` skips malformed `summary.json` candidates under broad `except Exception`.
- Why suboptimal:
  - Invalid JSON from partially-written files is masked, so status/pickup logic can continue with stale defaults instead of reporting parse health explicitly.
- Recommendation:
  - Include parse-error metadata in returned payloads and log failures so broken artifacts are diagnosable from poll output.

### 4) Medium — broad exception masking in category hooks reduces observability

- Location: `neuroinfra/analysis/catalog.py:29-33`, `47-50`, `99-103`, `139-143`
- Evidence:
  - Multiple categorization/order hooks are wrapped in `except Exception` and force fallback buckets (`"other"`).
- Why suboptimal:
  - Legitimate hook defects become silent data-reclassification events and can degrade report quality in production with no signal.
- Recommendation:
  - Keep fallback behavior but capture exception details (e.g., `warnings` list or `debug` field) so category computation issues remain visible.

### 5) Low — Blender slice-builder has explicit TODO/leftover code duplication

- Location: `olfactorybulb/slicebuilder/blender.py:2064-2067`, `2141-2142`
- Evidence:
  - TODO comment says `position_orient_align_mctc` likely duplicates existing method.
  - `extend_apic` is marked “probably unused, leftover”.
- Why suboptimal:
  - Duplicate/unused code paths can drift from active behavior and increase maintenance cost, especially in high-touch geometry assembly.
- Recommendation:
  - Remove duplicated code or add explicit deprecation/use-path comments and tests showing intentional retention.

### 6) High — campaign path is hard-coded to one developer checkout

- Location: `tools/run_hfo_campaign.py:372`
- Evidence:
  - `campaign_dir = _load_or_init_campaign(Path("/home/alek/OlfactoryBulb/results/notebook_runs/optimization") / campaign_slug, base_config, search_space)`
- Why suboptimal:
  - A hard-coded absolute path makes autonomous runs fail on any machine/worktree not mounted at `/home/alek`, including shared automation and non-default user sessions.
- Recommendation:
  - Resolve base path via the maintained repo root (for example `Path(__file__).resolve().parents[1] / "results"/"notebook_runs"/"optimization"`) and/or require an explicit campaign-root CLI option.

### 7) Medium — simulation-progress write failures are fully swallowed

- Location: `olfactorybulb/model.py:1047-1052`
- Evidence:
  - `write_progress_status()` wraps atomic JSON write + replace in `try/except Exception` and does `pass` on failure.
- Why suboptimal:
  - Polling and resume surfaces that rely on `sim_progress.json` will silently operate with stale status while no operator-visible warning is emitted.
- Recommendation:
  - Keep the progress file optional, but capture failure metadata in a logger or diagnostics channel (e.g., a warning with path and exception text) before continuing.

### 8) Medium — template PSD overlay builder silently drops rendering failures

- Location: `olfactorybulb/analysis_hfo_views.py:27-42`
- Evidence:
  - `build_psd_template_overlays()` catches all resolver exceptions and returns `[]` when template computation fails.
- Why suboptimal:
  - Visualization-only errors are hidden as empty overlays, making it hard to distinguish a failed overlay computation from a legitimately empty series.
- Recommendation:
  - Return a structured warning/failure payload or emit a logged warning; keep rendering robust but observable for debugging and reproducibility checks.

### 9) Medium — docs portal link resolution hard-codes specific user homepaths

- Location: `tools/build_maintained_docs_portal.py:105-107`
- Evidence:
  - `_resolve_repo_target()` checks `raw_path.startswith("/home/michael/OlfactoryBulb/") or raw_path.startswith("/home/alek/OlfactoryBulb/")`.
- Why suboptimal:
  - Absolute developer-machine prefixes are treated as valid repo anchors, which breaks docs-link conversion for any checkout outside those two home directories.
- Recommendation:
  - Normalize docs inputs to repo-relative paths or resolve absolute inputs against `REPO_ROOT` via a path-prefix table derived from environment, then emit an explicit warning for unsupported external prefixes.

### 10) High — broad monitoring exceptions can cancel remote jobs unnecessarily

- Location: `neuroinfra/remote/run_monitor.py:331-333`
- Evidence:
  - The `except Exception` branch in `monitor_remote_run()` unconditionally calls `cancel_remote_job_and_sync("Local notebook error while monitoring remote run")`.
- Why suboptimal:
  - Any non-critical local error (for example, transient poll rendering, status-parsing, or hook side effect failures) can force cancellation of a legitimate running remote job.
- Recommendation:
  - Narrow cancellation triggers to terminal remote-state conditions or explicit user-cancel signals; keep non-terminal local errors as warnings and continue monitoring with backoff/retry.

## Suggested Next Step

- Prioritize the three High findings first (remote defaults, campaign path, and monitor cancellation behavior), then parsing/observability Medium items that affect campaign correctness and live monitoring.
