# Local External State Registry

This note tracks non-obvious local paths and other easy-to-forget operational
artifacts outside the repo root that affect the maintained OBGPU workflow.

Use this registry when a future agent creates or depends on:

- a sibling git worktree
- an alternate checkout path
- a temporary external workspace tied to repo work
- another local path outside `/home/alek/OlfactoryBulb` that materially affects
  how the repo is edited, launched, or cleaned up

For each entry, record:

- path
- kind
- status
- purpose
- linked branch or note if relevant
- removal trigger for temporary entries
- cleanup commands or other retirement instructions

Do not use this file for normal repo-internal generated outputs such as
`results/` artifacts, caches, or transient scratch files that are already
covered by `.gitignore`, setup docs, or cleanup conventions.

## Current entries

### User-facing checkout path

- Path: `/home/michael/OlfactoryBulb`
- Kind: alternate user-facing checkout path
- Status: durable
- Purpose: preferred user-facing path for commands, links, and notebook-facing
  discussion. It may be a symlink to `/home/alek/OlfactoryBulb`.
- Notes: this is already part of the maintained workflow contract in
  `AGENTS.md`; this registry keeps it discoverable alongside other external
  paths.

### NeuronUnit overhaul worktree

- Path: `/home/michael/worktrees/OlfactoryBulb-neuronunit-overhaul`
- Kind: temporary git worktree
- Status: temporary
- Branch: `neuronunit-overhaul`
- Purpose: isolated implementation workspace for the planned NeuronUnit/SciUnit
  overhaul so work can continue there without contaminating the main checkout.
- Linked note: `notes/TEMPORARY_NEURONUNIT_OVERHAUL_WORKTREE.md` inside that
  worktree
- Remove when: the overhaul PR is merged, abandoned, or otherwise no longer
  needs an isolated worktree
- Cleanup:
  ```bash
  git worktree remove /home/michael/worktrees/OlfactoryBulb-neuronunit-overhaul
  git branch -d neuronunit-overhaul
  ```
  If the local branch must be removed before merge resolution is clean, use
  `git branch -D neuronunit-overhaul` deliberately instead of `-d`.
