# Phoenix CoreNEURON GID Reuse - 2026-05-27

## Symptom

Phoenix CoreNEURON runs with native LFP enabled could fail during MPI setup with
errors of the form:

- `Can't associate gid 1503000008. PreSyn already associated with gid 43367062.`
- `ParallelContext.cell(...)` on a synthetic `1.5e9+` gid

This showed up in an EPLI provisional run on Phoenix, but the failure mode is
not EPLI-specific.

## Root cause

The CoreNEURON native LFP path in `olfactorybulb/model.py` minted a fresh
`_next_lfp_report_gid` for every cell and then called `pc.cell(gid, nc)` on the
cell soma.

Some cells already had an existing NEURON gid bound to that cell's spike source
through the standard network/synapse setup path. In those cases, the LFP path
was trying to associate the same source variable with a second gid, which NEURON
rejects.

So this was **not** a numeric gid collision. It was a **duplicate source
registration** problem.

## Fix

`OlfactoryBulb.get_cell_report_gid(...)` now:

1. checks `_native_lfp_gid_source` for an already-known gid for that cell
2. reuses it when available
3. only allocates a fresh `1.5e9+` report gid for cells with no existing gid

This keeps the CoreNEURON native LFP mapping compatible with cells that already
own a gid through the normal network setup.

## Regression coverage

Added:

- `test_corenrn_native_lfp_gid_reuse.py`

This test verifies:

- existing cell gid is reused
- no synthetic report gid is consumed in that case
- fresh gids are still allocated for cells without a known source gid
