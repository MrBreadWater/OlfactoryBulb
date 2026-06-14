# Temporary NeuronUnit Overhaul Worktree

This note documents a **temporary local worktree** created so NeuronUnit
overhaul work can proceed without interfering with unrelated ongoing work in
the main checkout.

## Temporary status

This file is intentionally temporary.

It should be removed when **both** of the following are true:

1. the `neuronunit-overhaul` branch is no longer the active local workspace for
   this overhaul effort
2. the corresponding local worktree has been deleted

Do **not** keep this file as permanent project documentation after the worktree
is no longer needed.

## Current temporary worktree

- branch: `neuronunit-overhaul`
- local path:
  [`/home/michael/worktrees/OlfactoryBulb-neuronunit-overhaul`](/home/michael/worktrees/OlfactoryBulb-neuronunit-overhaul)
- purpose:
  isolate the NeuronUnit/SciUnit overhaul and PR-preparation work from the main
  active checkout at
  [`/home/alek/OlfactoryBulb`](/home/alek/OlfactoryBulb)

## Why it exists

The main checkout currently carries unrelated local modifications and generated
artifact noise. This separate worktree exists so the overhaul can be developed,
committed, and reviewed on its own branch without dragging those unrelated
changes into the same branch history.

## Related local state

- SciUnit creates a local user config directory at
  `/home/michael/.sciunit/` on first import.
- Current observed file:
  `/home/michael/.sciunit/config.json`
- Why it matters:
  this branch now uses SciUnit-backed import and validation helpers, so future
  agents should expect that config path to exist while working on the overhaul.
- Cleanup:
  delete `/home/michael/.sciunit/` only if you intentionally want to reset the
  local SciUnit user state for this machine.

## When to remove the worktree itself

Remove the worktree when one of these is true:

1. the overhaul branch has been merged and local branch-isolated work is no
   longer needed
2. the overhaul effort is abandoned or restarted elsewhere
3. the branch is converted into a different long-lived workflow and this
   specific temporary path is no longer the active location

## How to remove the worktree

From any checkout of this repository:

```bash
git worktree remove /home/michael/worktrees/OlfactoryBulb-neuronunit-overhaul
```

If the branch has already been merged or is no longer needed locally, remove
the local branch reference too:

```bash
git branch -d neuronunit-overhaul
```

If the branch must be force-removed because it is intentionally abandoned and
unmerged:

```bash
git branch -D neuronunit-overhaul
```

## When to remove this note

Remove this file in the same change or cleanup session that:

- deletes the local worktree above, or
- formally replaces it with a different documented temporary worktree path

If this file is ever merged into another branch by accident after the worktree
is gone, delete it immediately as stale operational documentation.
