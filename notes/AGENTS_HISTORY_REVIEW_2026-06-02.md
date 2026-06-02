# AGENTS History Review - 2026-06-02

Purpose:
- review the full git patch history after `AGENTS.md` was created and, per user
  request, continue beyond recent-memory bias
- capture durable agent-facing rules, traps, source-of-truth locations, and
  anti-rot guidance that belong in `AGENTS.md`
- avoid polluting `AGENTS.md` with transient state

Method:
- inspect the full git diff history in chronological order
- keep notes by commit range/topic
- promote only stable operational guidance into `AGENTS.md`

Status:
- in progress
- full patch stream exported and processed
  - `/tmp/olfactorybulb_full_history.patch`
  - 592 commit boundaries indexed
  - 7,667 candidate guidance lines mined from maintained-surface files

## Range Notes

### Whole-history themes identified so far

- The repo has two very different eras:
  - 2020 import/docs/archive setup
  - 2026 OBGPU rebuild, remote notebook workflow, audits, reference-data
    system, and `neuroinfra` extraction

- Stable themes already promoted or queued for promotion into `AGENTS.md`:
  - always use `OBGPU` for repo-local Python/import-driven work
  - `/home/michael/OlfactoryBulb` is the user-facing checkout path
  - the maintained workflow is the modern OBGPU path, not historical setup
    branches
  - live notebook remote execution is Paramiko-only and headless on the remote
    cluster
  - remote runs should publish committed git state, not ad hoc dirty code
  - remote/local runs should preserve one result contract
  - the cell-model registry is the source of truth for network-ready status
  - `Birgiolas2020` is the maintained current-network family
  - `SyntheticEPL2026.PVCRH_FSI1` is provisional, not network-ready by default
  - literature validation should default to separation, not silent pooling
  - `neuroinfra` is the reusable extraction target; compatibility wrappers
    should stay wrappers

### Initial/history-root pass

- Read the initial repository patch enough to confirm:
  - the repo was always science-artifact heavy
  - large morphology/media/docs assets are historical context, not a reason to
    treat everything in-tree as an active runtime surface

### OBGPU / remote workflow era

- Confirmed durable setup/remote themes from full-history commits and the
  maintained docs:
  - OBGPU setup became pinned-ref plus patch-stack based
  - Sol/Phoenix remote workflow became notebook-local + remote-headless
  - OpenSSH control-master and rsync paths were retired in favor of Paramiko
  - remote results were intentionally normalized back into the local notebook
    artifact contract
  - reusable allocations, deferred artifact sync, compact sync, and heartbeat
    monitoring are part of the maintained remote model, not incidental hacks

### Validation / reference-data era

- Confirmed durable validation-system themes:
  - declarative dataset configs and declarative validation configs are the
    maintained pattern
  - manual intake templates are raw human-curated inputs, not canonical outputs
  - notes/caveats must travel with the data
  - explicit human review and explicit band-mode selection are architectural
    requirements, not one-off Burton choices
  - normalized reference CSV/README outputs under `research_context/` are
    generated artifacts; agents should prefer changing configs/engines and
    regenerating rather than hand-editing canonical outputs
  - stable publisher/source URLs matter for reproducibility; redirected object
    storage URLs should stay transport-only, not provenance

### Still to review more carefully

- More of the mid-2026 notebook/remote helper extraction history for any stable
  anti-sprawl rules not yet promoted
- Whether older benchmark/debug/helpers encode additional “supported vs
  unsupported” boundaries worth making explicit in `AGENTS.md`

### Full-history scan hotspots

- Highest-signal files in the mined guidance-candidate scan:
  - `obgpu_experiment_helpers.py`
  - `tools/remote/submit_sol_run.py`
  - `olfactorybulb/audit/burton_urban_fi.py`
  - `olfactorybulb/audit/epli_correctness.py`
  - `neuroinfra/README.md`
  - `notes/REUSABLE_INFRASTRUCTURE_EXTRACTION_MAP_2026-06-01.md`
  - `notes/REFERENCE_VALIDATION_HOWTO.md`
  - `tools/setup/setup_ob_modern.sh`

- Additional durable rules promoted after the full-history scan:
  - environment/bootstrap/import/runtime-surface changes should be verified
    with `python tools/run_audit.py env_install`
  - normalized reference bundles should be regenerated from config/engine
    changes rather than patched by hand
  - stable publisher URLs, not transient redirect targets, are the right
    provenance identity in extracted reference rows
  - extraction changes should be followed by extractor + verifier reruns, not
    just unit tests
  - keep the layer distinction explicit between:
    - reference-data extraction tests / verifiers
    - declarative literature-validation runs
    - broader live system/model audits
  - parameter-surface sprawl should be resisted by keeping user-facing knobs
    centralized in real config/contract paths, with catalog docs following the
    runtime source of truth rather than replacing it
  - a curated maintained-surface health pass belongs in the official audit
    system, not as a separate ad hoc top-level script surface
  - `research_context/` needs an explicit boundary doc because raw sources,
    manual intake, configs, and generated outputs live side by side
