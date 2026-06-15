# Repository Agent Notes

This file is for coding agents operating inside `/home/alek/OlfactoryBulb`.
It is not user-facing product documentation. Treat it as the stable operating
contract for future sessions.

## 0. Maintenance duty and reproducibility

- Treat codebase ergonomicity as a top-priority product surface.
  - The codebase itself is part of the product.
  - Prefer APIs, helpers, config authoring paths, and internal module boundaries
    that are neat, low-ceremony, readable, and easy for a new contributor to
    understand.
  - When a durable default can remove repetitive boilerplate without hiding the
    actual behavior, prefer encoding that default once in a shared helper,
    builder, or contract layer instead of forcing every caller to restate it.
  - Do not confuse ergonomics with implicit drift: keep the real behavior
    inspectable in code, evidence, tests, and docs.

- Maintain `AGENTS.md` as a stable operating contract, not a session journal.
  Update it when a lesson is durable enough to guide future agents across
  tasks; do not record every UI tweak, bug fix, command run, or temporary
  workaround.

- Add or revise this file when the new information is one of:
  - a repo-wide invariant or source-of-truth boundary
  - a maintained workflow or entrypoint future agents should prefer
  - a durable change to how a maintained surface is registered, rendered, or
    launched
  - a recurring failure mode plus its correct recovery path
  - a verification standard that prevents false "done" reports
  - an anti-sprawl rule about where new logic should live
  - a reproducibility rule that must survive across chats
  - a dashboard or validation contract that future agents are expected to keep
    using, such as explicit plot-registration, backend selection, or item-level
    presentation fields

- Do **not** use this file for:
  - dynamic inventories that can be discovered live
  - one-off visual preferences unless they generalize into a UI contract
  - exact test logs, commit summaries, or historical play-by-play
  - current counts, current audit rosters, or current campaign names
  - implementation details already enforced by code/tests/config

- For dynamic inventory, point to live mechanisms instead of copying lists:
  - audits: `python tools/run_audit.py --list`
  - reference validations: `python tools/run_reference_validation.py --list-validations`
  - validation protocols: `python tools/run_reference_validation.py --list-protocols`
  - dataset configs: `research_context/reference_datasets/`
  - validation configs: `research_context/reference_validations/`

- If a behavior matters for reproducibility, encode it in code defaults,
  declarative config, tests, CLI discovery, or HOWTO docs. Use `AGENTS.md` to
  point at those sources of truth, not to replace them.
- When a change alters the day-to-day workflow that future agents will touch
  again, update the relevant HOWTO/docs in the same task and add a short
  contract note here if the rule needs to persist across chats.

- Keep detailed documentation ownership in `notes/DOCS_OWNERSHIP_MAP.md`.
  Keep `AGENTS.md` focused on contract-level rules and anti-rot expectations.

## 1. Core defaults

- Always use the `OBGPU` environment for repo-local Python work.
  - Preferred forms:
    - `source tools/setup/activate_obgpu.sh OBGPU`
    - `/opt/miniconda3/envs/OBGPU/bin/python ...`
  - Do not use base/system Python for:
    - tests
    - notebook-facing scripts
    - simulation helpers
    - audit/reference-data tooling

- `/home/michael/OlfactoryBulb` is the user-facing checkout path and may be a
  symlink to `/home/alek/OlfactoryBulb`.
  - Prefer `/home/michael/OlfactoryBulb` in user-facing paths and notebook
    discussions.
  - Preserve Michael's authenticated notebook/Jupyter session when working on
    live notebook-managed workflows.

- When making user-requested repository changes, create a targeted git commit
  before the final response unless the user explicitly says not to.

- Stage only files that belong to the current task.
  - Leave unrelated dirty files and user changes unstaged.
  - This worktree is often noisy. Do not treat unrelated staged/untracked files
    as yours to clean up.

- For notebooks, avoid committing transient execution output unless the output
  is intentionally part of the deliverable.

## 1a. Active maintained surface and archival boundaries

- Prefer the actively maintained OBGPU workflow over historical entrypoints.
  - For source-of-truth boundaries, check:
    - `notes/CODEBASE_CLEANUP_AUDIT.md`
    - `notes/DOCS_OWNERSHIP_MAP.md`
  - The maintained surface is organized by category, not by every file name:
    - setup/activation: `tools/setup/`
    - benchmark/smoke entrypoints: `tools/benchmarks/`
    - notebook/local/remote orchestration facade: `obgpu_experiment_helpers.py`
    - extracted reusable infrastructure: `neuroinfra/`
    - core runtime/model/config surface: `olfactorybulb/`, `single_cell_utils.py`, `fi_curve_utils.py`
    - active notebooks: `notebooks/obgpu-working-experiment.ipynb`, `notebooks/fi_curve_analysis.ipynb`

- Treat compatibility and archival paths cautiously.
  - `prev_ob_models/` contains important references, but most trees are
    archival rather than active runtime surfaces.
  - Do not treat a historical model tree as active just because it exists.
    Use the current runtime registry/configuration path as the source of truth.

- Keep generated and local-only noise out of normal commits.
  - In particular:
    - `results/notebook_runs/`
    - `results/sweeps/`
    - `results/benchmarks/`
    - `results/comparisons/`
    - `results/profiles/`
    - `results/debug_*/`
    - `results/tmp_*/`
    - `.jupyter-ai-state/`
    - `.codex/`
    - `.ipynb_checkpoints/`
    - architecture build outputs such as `aarch64/`, `x86_64/`,
      `corenrn_data/`, generated mechanism C/object/shared-library files

- Keep `external/` treated as a resettable upstream/dependency cache, not a
  hand-edited source of truth.

- Use the central cell-model registry as the source of truth for family, role,
  aliases, and `network_ready` status.
  - Do not bypass it with ad hoc direct imports when the registry path exists.

- Current biological/runtime boundaries:
  - `Birgiolas2020` MC/TC/GC models are the maintained current-network family.
  - The canonical maintained slice path is MC/TC/GC-first.
  - The synthetic `SyntheticEPL2026.PVCRH_FSI1` external plexiform layer
    interneuron surrogate is provisional and not network-ready by default.
  - Published candidate/proxy families in the registry are useful references,
    but should not be silently treated as maintained runtime defaults.

## 1b. OBGPU build and setup path

- The maintained build/setup path is the modern OBGPU path.
  - Start with:
    - `install-obgpu.sh`
    - `tools/setup/setup_ob_modern.sh`
    - `tools/setup/activate_obgpu.sh`
    - `tools/setup/activate_sol_obgpu.sh`
    - `tools/setup/verify_obgpu_python_imports.py`
  - Prefer current setup docs under:
    - `tools/README.md`
    - `notes/porting/SOL_REMOTE_WORKFLOW.md`
    - `notes/porting/NEURON_UPGRADE_WORKFLOW.md`

- Do not resurrect older build/setup branches when the modern path is the
  maintained one.

- The generic Linux/GPU OBGPU path is the primary build path.
  - Jetson-specific helpers remain secondary and should not become the main
    source of truth for the general workflow.

- Any NEURON/CoreNEURON portability or compatibility fix must land in:
  - `third_party_patches/nrn/`
  - and the pinned patch-stack workflow
  - not as an ad hoc local edit inside `external/nrn-*`

- Upstream NEURON bumps are deliberate and gated.
  - Use:
    - `python tools/setup/check_nrn_upgrade.py --candidate-ref <ref> ...`
  - The supported upstream ref changes only after:
    - patch stack replay succeeds
    - OBGPU rebuild succeeds
    - smoke/parity checks pass
  - Do not silently retarget the pinned upstream dependency.

- If the environment shows NVHPC stale temp-object loader failures such as
  `dlopen failed ... /tmp/pgcudafat...`:
  - use `tools/setup/fix_nvhpc_libnrnmech.sh`
  - and verify with:
    - `python tools/run_audit.py env_install`
  - Do not treat that warning as harmless without checking the actual mechanism
    libraries.

- After changing environment/bootstrap/import/runtime-surface code, run the
  maintained environment audit:
  - `python tools/run_audit.py env_install`
  - Use that instead of relying on ad hoc import spot-checks alone.
- On branches that actively migrate the SciUnit / NeuronUnit scientific core,
  also run:
  - `python tools/setup/verify_neuronunit_overhaul_imports.py`
  - Keep this as a narrower fork-readiness gate until the overhaul is mature
    enough to justify folding those imports into the main maintained env audit.

## 2. Verification standard

- Do not claim a fix without checking the actual result path the user will rely
  on.
- For CLI/reporting work, run the real command.
- For dashboard/runtime work, hit the served page or live artifact, not just
  the generator.
  - For interactive dashboard changes, verify the actual control behavior
    (draft preservation, reruns, loading/progress states, and tab content) in
    the served page, not only exported HTML or static snapshots.
  - For visual/layout dashboard changes, verify at least one rendered
    post-CSS state that exercises the changed layout. DOM presence/order tests
    are not enough when the task is spacing, overflow, collapse behavior,
    z-order, or responsive fit.
  - For math-criterion dashboard changes, verify the served browser path
    renders both the main equation and any symbol-definition rows from the
    exported HTML itself; raw TeX in definition text is a regression even if
    some separate math runtime appears elsewhere on the card.
- For notebook-facing defaults, verify the actual notebook/runtime paths that
  consume them.
  - If a user reports that a just-delivered change failed, reproduce that exact
    command or launch path before changing code again, then add or tighten a
    regression for the missed path when practical.

- When changing a notebook-facing default, verify the actual user paths that
  should observe it:
  - single run
  - remote run
  - sweep batch
  - live sync
  - final sync
  - load/animation helpers
  - Add or update tests for the default behavior, not only the helper
    implementation.

- Prefer proving behavior with the maintained live path when practical.
  - In this repo, helper-only validation is often insufficient.

- When changing result loading, remote sync, or notebook monitoring behavior,
  preserve the maintained performance/reliability semantics:
  - selected-file sync is an optimization, not the only success path
  - deferred artifact loading and lazy local/remote loaders are intentional
  - compact sync and partial-result recovery are intentional
  - heartbeat refresh/poll paths must stay bounded rather than inheriting
    unbounded remote waits
  - file/JSON-backed simulation progress is preferred over fragile stdout
    scraping

## 3. Live HFO optimizer and notebook-managed remote workflows

- If a live HFO workflow is blocked, recover the running campaign first and
  harden the failure path second.

- The live HFO campaign status file is:
  - `results/notebook_runs/optimization/codex_big_hfo_logs/latest_big_hfo_optimizer_status.json`

- The standard repair entrypoint for notebook-managed HFO recovery is:
  - `tools/analysis/resume_live_hfo_optimizer.py`

- For notebook-managed remote HFO reconnect issues:
  - inspect the latest campaign status first
  - prefer resuming the pending planned batch before proposing new work
  - treat fresh Paramiko handshake/banner-read failures as retryable transport
    errors, not only dead cached-session reuse

- If notebook kernel auto-discovery drifts, use the explicit connection-file
  path sooner instead of relying on ambient discovery.

## 3a. Remote execution architecture

- The supported remote-workflow model is:
  - local notebook/kernel as the control surface
  - remote cluster as the execution host
  - remote clone of this repo plus an `OBGPU` environment
  - notebook-managed submit/poll/sync lifecycle
  - local analysis on synced results

- The remote cluster should stay headless in the maintained workflow.
  - Do not redesign the supported path toward “run Jupyter on the cluster”.

- The maintained remote notebook/backend path is Paramiko-only.
  - Do not reintroduce or fork behavior around:
    - OpenSSH control-master transport
    - `ssh_multiplex`
    - `ssh_control_path`
    - `ssh_control_persist_s`
    - rsync result-sync paths
    - `rsync_binary`
    - `rsync_options`
    - `ssh_transport=\"openssh\"`

- The supported concepts are:
  - Paramiko persistent sessions
  - streamed compressed result sync
  - selected-file sync for sweeps and deferred artifacts
  - reusable Slurm allocations and manual `slurm_allocation_job_id`
  - `ssh_options` for port/jump-host behavior

- Keep remote config builders minimal.
  - If a value can be inferred from execution mode or Slurm resources, avoid
    adding a second independent knob.

- For remote execution, use committed code rather than dirty notebook-local
  state.
  - The maintained workflow expects notebook-managed git publication of the
    current local commit when needed.
  - Prefer the recorded git-ref path over ad hoc source copying.

- Preserve the remote result contract.
  - Remote runs should sync back into the same local result layout expected by
    local analysis helpers and notebooks, rather than inventing a second result
    format for remote execution.

- Selected-file sync and deferred-artifact fetches are maintained optimization
  layers.
  - Keep the fallback path to full/safer sync behavior intact.
  - Do not convert an optimization failure into silent data loss or a false
    success report.

- Preserve the distinction between local execution knobs and remote resource
  requests.
  - Local notebook runs still use settings such as `use_corenrn` and `use_gpu`.
  - Remote Slurm runs infer execution mode from the resource request unless
    explicitly overridden.
  - Do not collapse those two layers into one ambiguous configuration surface.

- On Sol, do not run heavy jobs on the login node.
  - Allocate first, then activate:
    - `salloc ...`
    - `source tools/setup/activate_sol_obgpu.sh`
  - When inside an allocation, use `$OB_MPIEXEC` instead of guessing the MPI
    launcher.

- The Sol module autoload convenience path is intentionally opt-in.
  - Preserve that behavior; do not make cluster-module side effects surprise
    generic Linux hosts on plain environment activation.

## 4. Dashboard/runtime expectations

- Maintained dashboard layers:
  - shell: `neuroinfra.dashboard.shell`
  - audit renderer: `olfactorybulb.audit.dashboard`
  - unified control center: `olfactorybulb.dashboard.control_center`
  - optimization/HFO module: `tools/analysis/hfo_visual_dashboard.py`

- Prefer the unified control center for end-user dashboard behavior:
  - `python -m olfactorybulb.dashboard.control_center`
  - The zero-argument path should serve promptly, print the reachable URL,
    leave audits idle until requested, and fall forward if port `6006` is busy.

- Verify dashboard changes against the served UI, not only generated files.
  At minimum, check the real launch path, relevant HTTP endpoints, and any
  changed interactive behavior with a DOM-capable browser test when JavaScript
  or layout state is involved.

- Dashboard UI contracts:
  - keep one shared shell rather than creating standalone dashboard stacks
  - mount new surfaces as tabs or modules in the unified shell
  - keep tab panels as full working surfaces, not card-in-card layouts
  - keep global status/progress compact and truthful
  - preserve local form drafts across background polling
  - the optimization tab should not auto-select a campaign on startup; it should expose an explicit selector backed by live discovery under `results/notebook_runs/optimization/`
  - keep the audit runner compact after a run and reopen it through the
    dedicated `Edit selection` action rather than a second launcher
  - when a dashboard iframe needs to trigger a shell-side action such as rerunning an audit from an empty state, use a small postMessage bridge to the shell rather than duplicating a second control surface inside the iframe
  - make audit reports glanceable by default, with collapsed groups/items and
    explicit expansion controls
  - audit-group sidebar rows are navigation only; clicking a group should
    scroll to it, but the sidebar must not render a selected/current-group
    state or `aria-current` highlight
  - keep warnings visible, specific, and non-duplicated
  - keep core maintained dashboard rendering self-contained; do not depend on
    third-party CDN runtime assets for features such as math rendering
  - use the bundled local KaTeX runtime to typeset criterion math in the
    browser, and verify the served page actually renders `.katex` output
  - keep rendered math backgrounds transparent; do not rely on opaque wrapper
    backgrounds around math fragments
  - when audit evidence contains an explicit series or curve, such as
    f-I/current-clamp arrays or `fi_curve_rows`, render the curve first only
    when the item explicitly opts into a `series_visuals` declaration; do not
    infer a line graph from array-shaped evidence alone
  - for math-rendered criteria, prefer one headline inequality or equality in
    `criterion_latex`, then place any supporting bound-construction formulas in
    `criterion_formulae` rather than hiding the actual mathematics inside prose
  - use `olfactorybulb.audit.criterion_math` for recurring criterion-math
    patterns rather than handcoding equivalent TeX strings in each audit rule,
    test fixture, or generated artifact; add or extend a builder before
    repeating a notation pattern
  - do not let audit/dashboard presentation rules become a web of hand-edited
    fixtures, generated artifacts, and hardcoded one-off strings. When a
    presentation convention changes, first move the convention into a
    high-level helper, declarative visual/math spec, renderer contract, or
    generator default, then regenerate the artifacts and update tests through
    shared fixtures or builders where practical. Configurability is the default
    expectation; repeated manual edits across many files are a signal to add or
    improve an abstraction before continuing
  - when a reference-band check is centered around a mean, prefer a compact
    absolute-residual criterion rather than endpoint notation or z-score
    notation. Use `|\bar{x} - \mu_{\mathrm{ref}}| \leq k\sigma_{\mathrm{ref}}`
    for symmetric arithmetic bands and
    `|\ln(\bar{x}) - \mu_{\log}| \leq k\sigma_{\log}` for lognormal
    reconstructions, with the selected multiplier rendered numerically. Use a
    metric-specific observed symbol when the quantity has a common notation such as
    `\bar{R}_{\mathrm{in}}`, `\bar{\tau}_m`, or `\bar{I}_{\mathrm{rh}}`, and
    keep `\bar{x}` as the fallback only when no clearer symbol exists. Treat
    `criterion_definitions` as a Definitions section for functions as well as
    variables, but do not introduce helper functions when the residual
    inequality is already readable. For lognormal reconstructions, keep
    `\mu_{\log} = \ln(\mu_{\mathrm{ref}}) - \sigma_{\log}^2/2` and
    `\sigma_{\log} = \sqrt{\ln(1+(\sigma_{\mathrm{ref}}/\mu_{\mathrm{ref}})^2)}`
    in supporting formulae. Use `\sigma_{\log}`, not the uploaded arithmetic
    `\sigma_{\mathrm{ref}}`, on the right-hand side of the log-space
    inequality. Avoid endpoint notation or auxiliary lognormal symbols like
    `\mu_\ell`, `\sigma_\ell`, `L`, or `U` unless the paper actually defines
    those quantities
  - make the plot declaration explicit at registration time:
    - use the helper builders in `olfactorybulb.audit` rather than hand-built
      dicts when possible
    - choose the backend per visual (`matplotlib` for the standard plots,
      `svg` only for compact bespoke renderers that really need it)
    - keep plot style knobs in the visual spec so the dashboard does not need
      to infer them from raw data shape
  - render explicit series/curve graphics in the persistent card body so they
    stay visible when the item is collapsed; do not hide them only inside the
    expanded detail body
  - f-I curve graphics should include explicit axis names plus sparse tick
    marks and numeric labels so the graph communicates scale without becoming
    cluttered
  - scalar numeric evidence may use a compact shared-scale dot strip, and
    short unpaired numeric sequences may use a sparkline, but only when the
    item explicitly opts into those companion visuals via `companion_visuals`
  - do not force scalar numeric evidence into a bar chart; use the compact
    dot strip or sparkline when a light visual improves the item, and keep the
    raw values available in the detail grid
  - keep persistent audit-card body regions visually coherent in collapsed and
    expanded states; numeric summaries, notes/caveats, and warning blocks
    should share deliberate spacing/order instead of relying on each block's
    incidental first-child margins
  - persistent interval summaries should stay compact, but they must still
    expose the key numeric values in-view; keep the metric grid for the
    expanded interval block, and use unlabeled value annotations aligned to
    the interval marks in the collapsed summary instead of verbose bound/mean
    callout text
  - when changing audit-card body layout, check representative combinations:
    no persistent body, numeric-only, notes-only, warning-only, notes plus
    warning, and numeric plus notes/warning when applicable
  - prefer responsive spacing such as `clamp()` over fixed desktop-only values

- Audit dashboard behavior that should stay stable:
  - sequential audit runs in one session remain visible instead of replacing
    earlier groups
  - `all` / new-sweep output preserves constituent audit groups
  - progress indicators reflect live audit progress, not just a spinner
  - numeric interval checks use structured visual summaries when possible

- The docs tab serves rendered maintained markdown under `docs/maintained/`.
  After changing maintained markdown or the render template, run:
  - `python tools/build_maintained_docs_portal.py`
  - `python tools/run_audit.py maintained_docs_integrity`

- Do not assume dashboard runtimes are healthy from status files alone. Verify
  the listener and rendered page. Shut down local servers you start during
  validation unless the user explicitly asks to leave them running.

- If `6006` is unexpectedly reoccupied after killing a dashboard child process,
  check for an older HFO dashboard watchdog/runtime that is respawning it and
  stop the runtime through the maintained HFO stop path.

## 5. Audit system: current architecture

- Keep the layer distinction clear:
  - `tools/run_audit.py` is the main human-facing surface for live system/model
    audits.
  - `tools/run_reference_validation.py` is the generic declarative
    literature-validation runner.
  - Low-level developer tests live under `tests/` and should usually be run as
    modules, for example `python -m tests.reference.test_reference_data_sanity`.
  - `python tools/run_audit.py test_suite_status --suite <suite_id>` is the
    maintained grouped presentation layer for those low-level tests.
  - `python tools/run_audit.py reference_dataset_status --dataset-id <id>` /
    `python tools/run_audit.py reference_dataset_contracts --dataset-id <id>`
    validate extracted literature bundles and provenance; they are not, by
    themselves, simulation-backed model-vs-literature audits.
  - Do not stop at reference-data extraction tests if the user asked for an
    actual literature-validation audit.

- Human-facing audit CLI:
  - `python tools/run_audit.py --list`
  - `python tools/run_audit.py <audit_id>`
  - stable aliases:
    - `python tools/run_audit.py default` -> maintained repo-health profile
    - `python tools/run_audit.py all` -> full registered audit sweep
  - `python tools/run_audit.py test_suite_status --list-suites`
  - For large grouped output, the default text mode may collapse fully passing
    groups.
  - Use:
    - `python tools/run_audit.py --expand`
    - `python tools/run_audit.py --failures-only`
    when you need the full or narrowed text view.

- Generic literature-validation CLI:
  - `python tools/run_reference_validation.py --list-validations`
  - `python tools/run_reference_validation.py --list-protocols`
  - `python tools/run_reference_validation.py --validation-id <id>`

- Audit web presentation:
  - `python -m olfactorybulb.audit.dashboard <audit_id> --output-dir <dir> [-- <audit args>]`
  - The audit dashboard should be driven from `AuditReport.to_dict()` rather
    than a separate ad hoc report schema.
  - Numeric interval-style validation items should render as structured
    reference-interval visuals rather than raw evidence dumps when the evidence
    includes accepted bounds plus the observed/reference mean values.
  - In the control-center shell, sequential audit runs should accumulate as
    separate audit groups for the current session.

- Do not hardcode the current audit roster here.
  - Audit IDs change over time.
  - The live source of truth is `python tools/run_audit.py --list` plus the
    audit registry in code.

- Do not add new root-level `test_*.py` files.
  - Low-level tests belong under `tests/`.
  - If a new check needs a user-facing maintained surface, prefer:
    - grouped suite audit for a family of tests
    - or a real operational audit if it is not fundamentally a developer test
  - Do not create one first-class audit per tiny helper regression.

- Keep the CLI output standards intact:
  - explicit `Description`
  - `Acceptable result`
  - `How Acceptable Result Was Determined`
  - explicit, concise `Warning` text for `WARN` items, especially when the
    warning comes from pending/provisional validation-design review state or
    an intentionally surfaced caveat; avoid repeating boilerplate phrasing like
    "why this is a warning" when a shorter warning label plus a specific
    reason is clearer
  - evidence blocks readable in plain text
  - grouped summaries should collapse cleanly in large multi-audit reports
    without hiding warning/failure detail

## 5a. Contract and registry discipline

- Keep feature/validation/dashboard surfaces mechanically linked to their
  registries or contracts.
  - Do not reintroduce hand-maintained duplicate lists when a contract or
    registry already exists.

- Keep user-facing parameter surfaces centralized.
  - When adding or changing notebook/run/network/cell parameters, prefer one
    canonical config or contract path rather than parallel helper-only knobs.
  - If human-facing parameter catalog docs are still relevant to the task, keep
    them aligned with the real runtime/config surface:
    - `build_run_config_parameters.md`
    - `notes/porting/NETWORK_AND_CELL_PARAMETER_CATALOG.md`
  - Do not let helper wrappers or notebooks become the only place a new
    user-facing knob exists.
  - When wrappers clone or prefix `AuditItem` instances, preserve
    `series_visuals`, `companion_visuals`, `criterion_latex`,
    `criterion_definitions`, and the other item-level presentation fields; do
    not rebuild audit items from a partial field subset.
- When a validation criterion has a compact mathematical form, keep the
  plain-language `criterion` as the fallback and opt into math rendering with
  explicit `criterion_latex` plus `criterion_definitions`; do not infer math
  from prose at render time.
  - Prefer `olfactorybulb.audit.criterion_math` builders for repeated
    comparison, tolerance, and reference-band forms.
  - For simple comparison rules such as `summary_metric_min`,
    `summary_metric_max`, `summary_metric_range`, `group_ordering`,
    `group_abs_diff_max`, `group_positive`, and `all_exact_metric`, emit the
    inequality or equality directly in `criterion_latex` instead of hiding it
    in prose.
  - For `summary_metric_range`, keep finite closed intervals in the shared
    absolute-residual form `|x - c| \leq r` and collapse semi-bounded ranges
    to one-sided inequalities instead of rendering `-\infty \leq x \leq U`
    or hand-writing endpoint notation in each validation.
  - The dashboard should still surface `criterion_formulae` when a rule only
    supplies supporting math rows.

- For HFO-facing parameter and visualization surfaces:
  - use the current contract/registry path rather than ad hoc whitelists
  - keep dashboard, packets, controls, and optimizer/search-space views derived
    from the same contract layer
  - run the contract audit after changes:
    - `python tools/run_audit.py hfo_feature_contracts`
  - if a new maintained dashboard surface is added, wire it through the shared
    shell and keep its visual style aligned with the other maintained tabs

- For literature validation:
  - dataset membership belongs in dataset configs
  - validation behavior belongs in validation configs, protocol runners, and
    rule kinds
  - human review status belongs in validation metadata
  - default behavior should preserve separation across incompatible targets
    rather than silently pooling them
  - if a maintained validation or audit still reads a concrete
    `research_context/*.csv` file directly, that file must be committed in the
    branch or the loader must be redirected to a committed dataset output.
    Clean worktrees must not depend on borrowing untracked local reference CSVs
    from some other checkout.
  - `reference_curve_match` now has an explicit unit/transform contract.
    Do not infer series-comparison compatibility from field names alone.
    Maintained validations should declare reference/model axis units,
    comparison units, and any needed axis transform explicitly.
    When a protocol runner already knows the shape of a graphable row bundle,
    register that once through the typed
    `olfactorybulb.audit.protocol_evidence` layer instead of repeating the
    model-side x/y keys and units at each consumer. `protocol_executed`
    should consume that contract instead of special-casing `fi_curve_rows`,
    and `reference_curve_match` may use it for ergonomic model-side defaults
    only. Keep reference-side and comparison-space units explicit in the
    validation config.
  - `reference_curve_match` should also declare `alignment_policy`,
    `distribution_kind`, and `score_family` explicitly; do not let the
    maintained path silently fall back to hidden series-comparison semantics.
    If the alignment policy uses tolerance-based matching or shared tolerance
    clustering, declare
    `x_match_tolerance` explicitly too.
    Secondary statistical knobs may use documented ergonomic defaults when the
    default is stable and the emitted evidence records the resolved choice.
    The same applies to secondary resampling knobs: for
    `alignment_policy = "resampled_grid"`, the maintained path may default
    `resampling_grid_source` to `union_observed_x`,
    `resampling_domain_policy` to `allow_partial_support`, and
    `interpolation_method` to `linear`, but the emitted evidence must record
    the resolved choice. Keep the resampling contract explicit and small:
    current interpolation options are `linear`, `nearest`, `pchip`, and
    `step_hold`, and current domain policies are `allow_partial_support`,
    `intersection`, `reference`, and `model`.
    Series-comparison evidence should keep the chosen score family plus compact
    reference/model provenance summaries visible in the emitted item payload.
    Prefer unit-neutral residual/statistical evidence keys such as
    `mean_absolute_error`, `root_mean_square_error`, `reference_sd_values`,
    and `equivalence_margin`, with unit metadata carried separately in
    `comparison_y_unit_text` / `error_unit_text`; do not keep teaching the
    NeuronUnit-side core that every series comparison is measured in hertz.
    When multiple contiguous `reference_curve_match` checks appear in one
    validation, batch them into one shared series-comparison suite overview
    plus the per-check detail items rather than emitting one one-case overview
    per rule.
    The emitted suite-case `score_text` and `norm_score` should reflect the
    declared score family rather than collapsing everything to a raw residual:
    residual families show residual diagnostics, equivalence families show
    equivalence p-values, hybrid families show both, and the suite overview
    should carry an aggregate norm-score summary when case-level normalized
    scores are available.
    Prefer explicit transform objects such as `affine`, `affine_lookup`, or
    `piecewise_linear` over one-off hardcoded x-axis conversion logic in
    protocol runners or rule handlers. Use `affine_lookup` when the scale or
    offset belongs to explicit row/protocol metadata rather than to a
    validation-local constant; keep the lookup key declarative instead of
    burying the conversion in Python.
    Keep reusable provenance structure out of ad hoc evidence-dict assembly:
    when a series-comparison change affects reference/model provenance payloads
    or compact protocol-context summaries, extend the shared typed layer in
    `olfactorybulb.neuronunit.provenance` and then adapt it into evidence,
    rather than growing more one-off dict-building helpers inside
    `series_validation_suite.py`.
    Keep bound series rows, transforms, comparison units, and provenance
    context on the shared `SeriesObservedDataset` layer instead of repeatedly
    pairing raw `rows`, `SeriesDataSpec`, and `context` dicts late inside the
    score path. `SeriesDistributionObservation` should produce those typed
    datasets for the reference and model sides before bins, paths, or
    provenance are computed.
    Likewise, keep protocol evidence bundled through the typed
    `olfactorybulb.audit.protocol_evidence` layer instead of threading a raw
    `protocol_evidence` dict plus a separate `evidence_series_specs` tuple
    through protocol results and rule consumers.
    Likewise, when a maintained literature bundle still arrives as a committed
    local summary CSV, move it under `research_context/source_data/<dataset>/`
    and route it through a declarative reference-dataset config rather than
    leaving a one-off repo-side loader as the long-term source of truth.
    Inside the series-comparison core itself, prefer the typed
    `SeriesDataSpec` / `SeriesVisualContract` layer over repeating raw
    x-key/y-key/unit/transform parameter bundles across bins, paths,
    provenance, and renderer-facing evidence assembly.
    In the declarative `reference_curve_match` rule layer, build those typed
    series specs directly from TOML/config fields before constructing the
    observation object; do not keep spelling the same flat series-field bundle
    inline at the rule-builder callsite. Keep the declarative parsing itself
    centralized in the typed `SeriesComparisonRuleSpec` path rather than
    scattering axis/policy/visual parsing across several unrelated helpers.
    Apply the same rule to protocol evidence presentation: the dashboard
    should consume an explicit row-source series contract from
    `olfactorybulb.audit.protocol_evidence` rather than guessing graph
    semantics from a magic evidence key.
  - When a maintained validation rule family compiles into a SciUnit-backed
    suite and also emits per-case `AuditItem`s, add one shared suite-overview
    item through `olfactorybulb.neuronunit.suite_presentation` rather than
    hand-building one-off rollup cards in each adapter. Keep the per-suite
    adapter focused on case-specific evidence/item construction and push the
    shared overview/matrix contract into that helper layer. Keep suite
    aggregation semantics in the typed `olfactorybulb.neuronunit.suite_scores`
    layer so the aggregate status/norm policy is not reimplemented as loose
    dict math inside each adapter or in the dashboard renderer. Use the typed
    `SuiteDescriptor` bundle there for suite id, suite kind label,
    candidate/model ids, and aggregate policy instead of threading four
    separate parameters through each adapter. Mark the overview item
    `summary_rollup_exempt` so report/group PASS/WARN/FAIL counts stay tied to
    the detailed cases rather than double-counting the overview. When a suite
    overview carries `suite_aggregate_score` or
    `suite_norm_score_summary`, surface that in the dashboard matrix header
    instead of hiding it only in raw evidence. If the matrix has a compact
    score or norm label that helps scanning, emit it in the suite-case payload
    rather than teaching the dashboard to reverse-engineer it from raw
    detailed evidence. `reference_band_rows` is not an exception here: it also
    compiles through a `SuiteDescriptor` and accepts the same
    `suite_aggregate_policy` contract as the grouped summary/comparison/series
    suite families. When a grouped suite exposes real case-level statistical
    diagnostics, carry the suite-level policy through the same descriptor path
    with `suite_statistical_policy` rather than baking an aggregation choice
    into the dashboard or one suite adapter. The maintained default remains
    category-specific auto selection, but explicit rollups such as `median`
    should stay declarative and surface their resolved source in emitted
    evidence. When a suite-level statistical summary should only count as
    supported if enough cases contributed real statistical diagnostics, use
    declarative `minimum_available_case_count` and/or
    `minimum_available_case_fraction` on `suite_statistical_policy` instead
    of inferring that support requirement from the rolled-up p-value alone.
    Keep the resulting `available_case_count`, `available_case_fraction`,
    `support_gate_passed`, `threshold_gate_passed`, and combined
    `gate_passed` visible in `suite_statistical_summary`. Keep the migrated
    NeuronUnit-to-`AuditItem` adapter
    centralized in `olfactorybulb.neuronunit.suite_presentation`; use the
    shared `AuditItemAdapterSpec` path instead of hand-constructing the same
    criterion/review/visual field bundle in every suite adapter, and add or
    extend explicit adapter tests when a new user-facing audit field must
    survive the bridge into the maintained shell. When a suite case has a real
    numeric/statistical score behind the compact label, emit the typed
    `case_score` payload in the suite-case evidence rather than forcing the
    dashboard or downstream tools to recover semantics from `score_text`
    alone. Scalar SciUnit-backed rule families should also resolve one typed
    metric-quantity contract instead of threading raw `metric_key` strings and
    one-off unit labels through the spec, case, score, and presentation
    layers. Use the shared `olfactorybulb.neuronunit.metric_quantities`
    helpers there, let common suffixes such as `_mV`, `_ms`, `_pA`, `_pF`,
    `_MOhm`, `_Hz`, and `_um` provide ergonomic defaults, let the shared
    observed-symbol map provide common scientific notation such as
    `\bar{V}_{\mathrm{th}}`, `\bar{\tau}_m`, or `\bar{I}_{\mathrm{rh}}`
    automatically, and use explicit `metric_unit_text`,
    `metric_quantity_name`, or `metric_observed_symbol` overrides only when
    the metric key is ambiguous or the displayed scientific presentation needs
    refinement. For
    scalar summary/comparison suites, keep the internal prediction/observation
    path on the shared typed scalar layer in
    `olfactorybulb.neuronunit.scalar_observations` instead of rebuilding raw
    dict bundles for metric maps, group pairs, or grouped scalar sets in each
    rule family. When a scalar rule needs status-coded discrete values such as
    `summary_metric_status_map`, route that through one typed
    `ScalarStatusMapPolicy` instead of threading parallel `pass_values` /
    `warn_values` / `fail_values` lists across the declarative parser, suite
    case, score builder, and adapted evidence. Keep protocol metric rows and
    grouped numeric summaries on the
    shared typed metric-table layer in
    `olfactorybulb.neuronunit.metric_tables` rather than passing fresh
    `list[dict]` / `dict[group][metric]` bundles through the protocol result,
    validation context, and NeuronUnit suite/model seams. When a suite
    aggregate policy uses `norm_rollup = "weighted_mean"`, only do so
    for a suite family that emits a principled per-case weight through the
    shared suite-case contract. In the maintained branch that currently means
    the series-comparison suite family, which uses matched-point count as its
    aggregate weight source. Keep the case weight visible in `suite_cases`
    rather than hiding it only inside the aggregate rollup. When case scores
    expose real statistical diagnostics such as equivalence or Welch p-values,
    preserve them at the suite-overview layer through the typed
    `suite_statistical_summary` payload instead of flattening the overview to
    norm scores only.
  - Keep shell-side meta rules such as `protocol_executed` and `note_presence`
    in the audit layer, but do not leave their declarative parsing as raw
    handler-local dict plumbing. Route their rule-specific config through typed
    parser objects in `olfactorybulb.audit.reference_validation_specs` so the
    declarative surface stays uniform even when a rule intentionally remains
    outside the NeuronUnit scientific core.

## 5b. Reusable infrastructure extraction rules

- `neuroinfra/` is the internal extraction target for reusable infrastructure.
  - Use:
    - `neuroinfra/README.md`
    - `python -m neuroinfra`
  - to inspect the current extraction inventory and source-of-truth locations.

- Do not grow new mixed-responsibility monoliths when a reusable layer already
  exists under `neuroinfra`.
  - Prefer extending the extracted generic modules and keeping repo-specific
    wiring in the `olfactorybulb.*` adapters or notebook-facing glue.

- `obgpu_experiment_helpers.py` is still a major notebook-facing facade, but it
  should keep shrinking toward orchestration glue rather than regaining
  ownership of:
  - remote transport internals
  - result-artifact schemas
  - generic analysis primitives
  - dashboard runtime supervision
  - generic config/run catalog logic

- Before adding new helper logic, check whether the reusable home already
  exists in:
  - `neuroinfra.notebooks`
  - `neuroinfra.remote`
  - `neuroinfra.analysis`
  - `neuroinfra.artifacts`
  - `neuroinfra.dashboard`
  - `neuroinfra.contracts`
  - `neuroinfra.campaigns`
  - `neuroinfra.models`

- The `tools/remote/*.py` entrypoints now often act as compatibility bootstrap
  wrappers around `neuroinfra.remote_*` modules.
  - Keep user-facing CLI wrappers working, but put new generic logic in the
    extracted module rather than the wrapper script.

- `tools/debug/` scripts are one-off diagnostics, not maintained public
  workflow entrypoints.
  - Do not infer supported workflow contracts from a debug helper unless the
    same rule is anchored in the maintained setup/notebook/audit surfaces.

- The maintained import/runtime audit scope is the OBGPU workflow surface.
  - Blender-only paths and the older `neuronunit` stack are not part of the
    maintained import-verification contract for current OBGPU work.

## 6. Declarative literature-validation rules

- Use the declarative validation framework instead of bespoke one-off audit code
  when adding literature-backed validation:
  - configs live in `research_context/reference_validations/`
  - template:
    - `research_context/reference_validations/TEMPLATE.validation.toml`
  - guide:
    - `notes/REFERENCE_VALIDATION_HOWTO.md`
  - system overview:
    - `notes/REFERENCE_VALIDATION_SYSTEM_OVERVIEW.md`

- `burton_urban_fi` is one registered protocol-backed validation in this
  system, not the architecture itself.

- If a paper needs a new protocol:
  - add a registered protocol runner
  - keep custom measurements in the protocol runner output

- If a paper needs a new judgment rule:
  - add a new validation rule kind
  - keep comparison logic in the rule layer, not buried inside the protocol
    runner

- If a validation needs cheap static checks plus optional NEURON-backed checks,
  prefer the declarative path and use:
  - `skip_neuron_mode = "protocol_handles_skip"`
  - This is how `epli_correctness` now works.

- The reference-validation tutorial/docs surface is a maintained permanent
  contract, not optional background reading.
  - When changing validation-design review statuses, accepted top-level
    validation TOML sections, built-in protocol runners, built-in rule kinds,
    warning/caveat semantics, or validation visualization/math behavior,
    update these in the same task:
    - `notes/REFERENCE_VALIDATION_TUTORIAL.md`
    - `notes/REFERENCE_VALIDATION_HOWTO.md`
    - `research_context/reference_validations/TEMPLATE.validation.toml`
  - Keep definitions explicit there rather than expecting future agents to
    rediscover semantics by reading code.

## 7. Validation-design review metadata is mandatory

- Every declarative validation item should resolve to a validation-design
  review state.

- Supported review statuses:
  - `approved`
  - `provisional`
  - `pending`
  - `not_applicable`

- Use `[validation_design_review]` in validation configs.
  - The semantic meaning is manual review by an appropriately qualified expert
    of the validation-design choice itself.
  - This repo intentionally allows LLM-assisted drafting as a first-class
    workflow, but LLM output does not count as validation-design review.
  - The purpose of this metadata is to mark where a scientifically qualified
    person has actually checked the design choice behind the validation item.
  - At minimum:
    - `[validation_design_review]`
    - `default_status = "pending"`

- For `reference_band_rows`, per-property review state is preferred when not
  all metrics are at the same maturity:
  - `property_validation_design_review_statuses`
  - `property_validation_design_review_notes`
  - `property_validation_design_review_reviewers`

- Run the coverage audit after changing validation configs:
  - `python tools/run_audit.py validation_design_review_status`

- The `validation_design_review_status` audit should:
  - fail on missing review status coverage
  - fail on unknown status strings
  - warn on `pending`
  - warn on `provisional`

- Keep the semantics clear in human-facing renderers.
  - Validation-design review metadata describes the review state of the
    underlying validation rule or reference-band choice, not whether one
    specific observed audit result row was manually checked after the fact.
  - Good examples:
    - whether a reference-band distribution shape is acceptable
    - whether pooling/separation choices are acceptable
    - whether a protocol-equivalence assumption is acceptable
    - whether a manual extraction or mapping choice is acceptable
  - Non-examples:
    - someone has verified the full code stack used by the audit
    - someone has manually confirmed this specific PASS/WARN/FAIL outcome
    - someone has re-reviewed every upstream source document in full for this run
  - Do not render validation-design review metadata inline on individual audit
    result items in the CLI or HTML dashboards unless the wording is explicitly
    reframed to avoid that confusion.

## 8. Reference-band selection rules

- `reference_band_rows` requires an explicit band mode for every property in
  `property_metric_map`.
  - No silent fallback is allowed.
  - Keep the declarative parsing centralized in the typed
    `ReferenceBandRuleSpec` / `ReferenceBandPropertyRuleSpec` path rather than
    scattering per-property bounds, quantile-field selection, and review-state
    parsing across the handler body. Those typed compiler specs now live in
    `olfactorybulb.audit.reference_validation_specs`; keep the rule handler
    focused on orchestration and suite compilation rather than raw per-row case
    construction.

- For the grouped summary/comparison rule families, keep the declarative
  parsing centralized in the typed `SummaryRuleSpec` / `ComparisonRuleSpec`
  path before the SciUnit-backed suite cases are built. Those parser objects
  also live in `olfactorybulb.audit.reference_validation_specs`. Do not let
  the handler body regress into a second pile of raw rule-dict lookups.

- Apply the same rule to `reference_curve_match`: keep both its declarative
  parsing and its `SeriesComparisonCase` construction in the typed
  `SeriesComparisonRuleSpec` path. The shell-side handler should load rows,
  resolve protocol evidence specs, and hand the typed case to the SciUnit
  bridge rather than rebuilding the flat observation bundle inline.

- Keep the top-level runtime path on the same side of that boundary.
  - `olfactorybulb.audit.reference_validation_config` should stay the boring
    raw-I/O layer: load raw TOML, list validation ids, and register declared
    extension modules.
  - `olfactorybulb.audit.reference_validation_protocol_core` is the typed
    protocol contract layer: keep protocol specs, registry lookup, execution
    cache policy, and cache-key normalization there instead of scattering
    those concerns across the concrete protocol implementations or CLI glue.
  - The first structured layer above that is the typed
    `ReferenceValidationDocument` in
    `olfactorybulb.audit.reference_validation_document`.
    That document should carry typed `ValidationRuleRecord` entries rather
    than a raw `checks` list once the config has crossed the raw-I/O layer.
  - The maintained runtime should compile that raw config into the typed
    `ReferenceValidationPlan` in `olfactorybulb.audit.reference_validation_plan`
    before the engine, CLI, or maintained audit wrappers consume it.
  - That plan should also own the precompiled rule-dispatch sequence for the
    validation. Do not make the runtime regroup contiguous summary/comparison/
    series rule families from raw rule dicts on every execution once the plan
    has already been built. The raw rule list belongs to the typed document
    layer; do not keep duplicating it inside the runtime plan unless a real
    runtime consumer actually needs it.
  - The runtime rule layer should consume a typed `ValidationRuleContext`
    carrying explicit fields such as `validation_id`, `notes_path`,
    `default_group`, and typed review defaults; do not reintroduce a generic
    `context.config` dict just to tunnel those values through the rule engine.
    Keep any non-cyclic interface helpers for that seam in
    `olfactorybulb.audit.reference_validation_contracts` rather than falling
    back to `context: Any` inside the typed spec/parser layer.
  - Keep the protocol-result boundary typed too: the plan, engine, and rule
    context should trade `ProtocolRunResult`, not `Any`, once protocol
    execution has crossed the registry/cache layer.
    Apply the same rule to adapter helpers: use small structural protocols for
    protocol-evidence carriers and suite-case adapters instead of leaving
    those seams typed as raw `Any`.
  - Inside `olfactorybulb.audit.reference_validation_rules`, built-in
    maintained rule kinds should compile into typed dispatch records/specs
    before runtime execution instead of carrying loose raw rule dicts all the
    way to the handler edge. Keep the raw `register_validation_rule(...)`
    hook as the compatibility surface for extension-defined custom rule kinds,
    not as the internal execution model for the maintained built-in rules.
  - Static config-inspection surfaces such as validation-design-review audits
    should consume the typed document and its typed rule records rather than
    hand-walking loose config dicts and parallel accessor calls.
  - Do not let `reference_validation_engine.py` or the maintained wrapper
    modules drift back toward passing loose config dicts and parallel accessor
    calls around at runtime.
  - Keep protocol-result caching explicit and inspectable:
    - concrete maintained protocol implementations belong in
      `olfactorybulb.audit.reference_validation_protocols`
    - cacheable protocols should declare semantic `cache_arg_names`
      explicitly on `ValidationProtocolSpec` rather than inheriting an
      implicit "all CLI args" cache surface
    - once protocol evidence has crossed
      `coerce_protocol_evidence_bundle(...)`, keep the public runtime/model
      seam on `ProtocolEvidenceBundle` instead of advertising raw dicts as a
      first-class protocol-evidence payload type
    - keep the public protocol-evidence payload maps typed too: store
      `ProtocolEvidenceBundle.values` and `ProtocolEvidenceSeriesSpec.style`
      on shared frozen mapping wrappers, but keep plain mapping coercion at
      the boundary so maintained protocol builders and tests can stay concise
    - keep the top-level validation config/cache tables on the same typed
      wrapper pattern too: `ReferenceValidationDocument.defaults`,
      `ReferenceValidationDocument.protocol_defaults`, skip-item evidence, and
      `ProtocolExecutionCacheInfo` payload maps should not drift back to loose
      mutable dicts once they have crossed the load/cache boundary
    - keep declarative validation-rule payloads on the same frozen-wrapper
      pattern too: `ValidationRuleRecord.raw_rule` should stay on a typed
      frozen mapping payload, and the rule/spec parsing layer should accept
      mapping-like payloads instead of advertising mutable raw dicts as the
      primary maintained contract
    - once declarative payloads are frozen, do not let parser helpers regress
      to mutable-sequence assumptions; rule/spec parsers should accept the
      tuple-backed sequence values produced by the frozen payload layer rather
      than requiring raw `list` inputs
    - if a migrated SciUnit suite needs the full protocol-evidence payload,
      expose that through an explicit capability such as
      `ProvidesProtocolEvidenceBundle` and consume the bundle directly rather
      than reconstructing it from a looser map/row pair inside the suite
    - keep the series-suite prediction seam typed too: do not split typed
      protocol evidence back into ad hoc `rows + context` payloads when the
      prediction bundle can carry `ProtocolEvidenceBundle` directly
    - when series-comparison evidence needs both reference and model
      provenance, keep that on one nested typed payload rather than
      scattering sibling keys like `reference_*` / `model_*` across the score
      evidence map unless a renderer or CLI surface truly needs flattened
      aliases
    - keep the series-comparison score/evidence seam typed too: the computed
      series diagnostics should live on one explicit payload object that owns
      score text, suite-case statistical summaries, and evidence export,
      instead of rebuilding a large ad hoc dict in `compute_score()`
    - when a maintained series comparison needs more than a raw matched-bin
      count, keep the overlap contract explicit too: use typed minimum
      reference/model coverage-fraction fields and surface the resulting
      support-gate evidence/norms, rather than letting a thin aligned overlap
      masquerade as a fully supported residual/statistical pass
    - the maintained resampled-grid interpolation family currently includes
      `linear`, `nearest`, `pchip`, and `step_hold`; extend that explicit
      family deliberately rather than smuggling a new interpolation behavior
      in behind the same label
    - keep resampled-grid domain handling explicit too: the maintained domain
      policies are `allow_partial_support`, `intersection`, `reference`, and
      `model`, and the emitted evidence should expose both the resolved policy
      and any filtered-out grid points instead of leaving domain clipping
      implicit in missing-support counts
    - when a maintained SciUnit wrapper already has a typed case or
      observation object, make that object own the SciUnit observation payload
      via a small `observation_payload()` helper instead of rebuilding a
      parallel raw dict inline in each wrapper `__init__`
    - when suite-level statistical rollups need case-level p-value metadata,
      carry that on an explicit typed statistical payload attached to
      `SuiteCaseScorePayload` rather than reverse-engineering it from generic
      `observation` / `prediction` dict fragments
    - keep the non-statistical `observation` / `prediction` /
      `normalization` sections on `SuiteCaseScorePayload` typed too, but
      preserve low-ceremony callers by coercing plain mappings into one shared
      frozen payload wrapper at the score-boundary instead of forcing every
      suite builder to instantiate verbose adapter classes by hand
    - keep grouped-suite default ids family-scoped (`summary_rules`,
      `comparison_rules`, `series_rules`, etc.) instead of reusing the bare
      validation id, so overview items stay mechanically distinct on the
      maintained report path
    - cache hits/misses should stay visible in emitted protocol evidence under
      `protocol_cache`; do not hide them behind silent control flow

- Supported band modes currently include:
  - `symmetric_sd`
  - `lognormal_sd`
  - `beta_sd`
  - `quantile_interval`
  - `binary_indicator`

- Choose the mode manually per metric. Do not assume every positive-valued
  metric should use a lognormal reconstruction.

- Keep a hard distinction between:
  - what the paper reported
  - what the audit reconstructed as an acceptance band

- If a band is an audit-side reconstruction or other temporary stopgap, mark it
  explicitly with:
  - review status `provisional`
  - visible note/caveat in the relevant notes CSV when appropriate

## 9. Reference-data pipeline: current architecture

- Use the generic declarative reference-data system for literature bundle work:
  - dataset configs:
    - `research_context/reference_datasets/`
  - template:
    - `research_context/reference_datasets/TEMPLATE.dataset.toml`
  - guide:
    - `notes/REFERENCE_DATASET_HOWTO.md`
  - manual intake templates:
    - `research_context/manual_reference_templates/`

- Main generic commands:
  - `python tools/download_reference_dataset_sources.py --dataset-id <id>`
  - `python tools/extract_reference_dataset.py --dataset-id <id>`

- Human-readable bundle summaries:
  - `python tools/run_audit.py reference_dataset_status --dataset-id <id>`
  - `python tools/run_audit.py reference_dataset_contracts --dataset-id <id>`
  - Do not assume dataset-specific helper wrappers exist.

- Treat normalized reference bundle outputs in `research_context/` as generated
  artifacts unless a file is explicitly a manual-intake template or a raw
  downloaded source file.
  - Prefer changing:
    - dataset config
    - extraction engine
    - source downloader/manifest
    - manual intake artifact
  - Then regenerate the normalized CSV/README outputs.
  - Do not hand-edit generated canonical outputs just to “fix the data” unless
    the task is explicitly about a one-off manual correction and the provenance
    implications are understood.

- Preserve stable provenance URLs.
  - When download URLs redirect through signed/object-store links, keep the
    stable publisher URL in `source_url` and treat the redirected fetch target
    as transport detail rather than canonical provenance.

## 10. Reference-data boundaries

- The pipeline directly supports:
  - downloaded supplemental tables
  - local manual tables
  - local manually digitized CSVs

- The pipeline does **not** do built-in screenshot or figure digitization.
- If actual numeric values cannot be extracted reliably:
    - do not guess
    - use `needs_manual_extraction.csv`
    - use the manual reference templates where applicable

- Manual intake templates live under:
  - `research_context/manual_reference_templates/`
  - They are raw human-curated intake artifacts, not canonical validation CSVs.

- Manual intake standards:
  - leave unknown fields blank
  - do not guess
  - keep `source_location` specific
  - preserve exact wording in `reported_text`
  - keep subtype and condition separated
  - do not pool different cell classes or protocols in one row
  - for f-I point templates, do not back-project points from summary metrics
    such as max rate or gain alone
  - keep one row per metric for summary tables, one row per current step for
    f-I tables, and one row per protocol variant for protocol templates

- Default literature-validation discipline:
  - do not silently pool different cell classes
  - do not silently pool subtype-specific and generic targets
  - do not silently pool different protocols
  - do not silently pool baseline with pharmacology/modulation conditions
  - do not silently pool across species, age, or maturity when those
    distinctions are part of the reference meaning
  - if incompatible targets appear together, keep the relevant notes/caveats
    visible in downstream outputs

- Preserve provenance in all extracted rows:
  - `source_file`
  - `source_url`
  - `source_location`
  - `reported_value_raw`

## 11. Mixed-unit table warning

- Be careful with `formatted_summary_rules`.

- If a formatted table mixes properties with different unit semantics, do **not**
  use one blanket `transform_scale` unless every mapped property truly needs the
  same scaling.

- Preferred solution:
  - `property_transform_scales = { ... }`

- The engine now rejects risky mixed-property blanket scaling by default unless
  explicitly overridden with:
  - `allow_blanket_transform_scale = true`

- This rule exists because it previously caused real GC extraction corruption:
  gain conversion was correct, but the same blanket scale also inflated
  rheobase, latency, and peak-rate rows by a factor of one thousand.

## 12. Sanity tests for extracted bundles

- Run the reference-data sanity test when changing extraction logic:
  - `python -m tests.reference.test_reference_data_sanity`

- Also rerun the relevant dataset extractor and verifier, not just unit tests.
  - Generic rebuild:
    - `python tools/extract_reference_dataset.py --dataset-id <id>`
  - Generic/source checks:
    - `python tools/download_reference_dataset_sources.py --dataset-id <id>`
  - Human-readable verification:
    - `python tools/run_audit.py test_suite_status --suite reference_bundles`
    - `python tools/run_audit.py reference_dataset_status --dataset-id <id>`
    - `python tools/run_audit.py reference_dataset_contracts --dataset-id <id>`

- That test is heuristic, not exhaustive. It is intended to catch obvious
  mistakes such as:
  - absurd magnitudes for standard units
  - negative standard deviations
  - reversed quantiles
  - impossible signs for CV, resistance, capacitance, rheobase, and similar
    metrics

- Still inspect suspicious rows against source files manually. Passing sanity
  checks does not prove semantic correctness.

## 13. Notes/caveats must travel with data

- Validation caveats belong in notes tables and must show up downstream.

- Use the notes system rather than burying important warnings inside free-text
  comments:
  - protocol differences
  - subtype separation
  - modulation-state separation
  - provisional or audit-side reconstruction caveats

- For outputs that compare incompatible protocols or populations, the notes
  section should remain visible in CLI/report/HTML outputs.

## 14. Good commands to know

- Audit/discovery:
  - `python tools/run_audit.py --list`
  - `python tools/run_audit.py`
  - `python tools/run_audit.py default`
  - `python tools/run_audit.py test_suite_status --list-suites`

- Dashboards/docs:
  - `python -m olfactorybulb.dashboard.control_center`
  - `python -m olfactorybulb.audit.dashboard new_sweep --output-dir /tmp/full_audit -- --skip-neuron`
  - `python tools/build_maintained_docs_portal.py`
  - `docs/index.html`

- Reference workflows:
  - `python tools/run_reference_validation.py --list-validations`
  - `python tools/run_reference_validation.py --list-protocols`
  - `python tools/extract_reference_dataset.py --dataset-id <id>`

## 15. Commit hygiene reminder

- Before final response:
  - review the diff
  - stage only task files
  - commit once with a targeted message
  - leave unrelated modifications alone

- If you accidentally stage unrelated files:
  - fix the index/commit scope cleanly
  - do not rewrite or discard user worktree changes
