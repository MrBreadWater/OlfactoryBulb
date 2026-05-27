# Phoenix/CoreNEURON shared-node gid fix

## Failure signature

CoreNEURON-enabled Phoenix runs of `GammaSignature_EPLI_Provisional_TCOnly` failed during
parallel synapse setup with errors of the form:

- `Can't associate gid 801091747. PreSyn already associated with gid 633119557.`

This was a different failure mode from the earlier native-LFP report-gid collision.

## Root cause

BlenderNEURON hashes synapse source gids from the exported section name and segment index.
That is usually fine, but it breaks when two exported addresses land on the same NEURON
voltage source after the cell is instantiated.

The important case here is a shared parent/child node, for example:

- `parent_section(1.0)._ref_v`
- `child_section(0.0)._ref_v`

Those can be the same underlying NEURON source handle even though the exported section
strings hash to different gids. When both appear in reciprocal synapse sets, the second
`ParallelContext.cell(gid, nc)` crashes because the source is already bound to another gid.

## Fix

Before creating each synapse set, the runtime now:

1. scans the local entries that this rank can resolve to real NEURON sections
2. groups requested gids by actual NEURON source handle equality
3. chooses a canonical gid for each shared handle (`min(gids)`)
4. gathers and broadcasts the alias map to every rank
5. forces both source registration and `gid_connect(...)` to use the canonical gid

This preserves cross-rank consistency. A source-owning rank and a destination-only rank now
agree on the same canonical gid even when the original exported addresses alias to one handle.

## Verification

Focused tests:

- `python -m py_compile olfactorybulb/model.py test_shared_source_gid_alias.py test_corenrn_native_lfp_gid_reuse.py`
- `MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_shared_source_gid_alias.py`
- `MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/python test_corenrn_native_lfp_gid_reuse.py`

MPI setup smoke:

- `MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/envs/OBGPU/bin/mpiexec --oversubscribe -n 16 /opt/miniconda3/envs/OBGPU/bin/nrniv -mpi -python -c ...`

Observed result:

- `NRN MPI init OK 16`

That local 16-rank `nrniv -mpi` initialization completed without the old `PreSyn already associated`
crash while loading `GammaSignature_EPLI_Provisional_TCOnly`.
