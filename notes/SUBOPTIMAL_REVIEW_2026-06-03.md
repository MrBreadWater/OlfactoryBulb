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

## Suggested Next Step

- Prioritize High finding first (remote defaults), then the two Medium parsing-failure items that can affect campaign correctness.
