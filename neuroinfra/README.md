# neuroinfra

`neuroinfra` is the internal extraction target for the reusable infrastructure
that has accumulated inside this repository.

It is **not** yet the runtime owner of the main simulation, notebook, or remote
execution paths. Right now it serves two purposes:

1. provide a stable place to define what should become reusable framework code
2. force the extraction plan to become structured, inspectable, and testable

## Current scope

The first implemented artifact is the component inventory:

- `neuroinfra.inventory`

The first live internal extraction is now also present:

- `neuroinfra.artifacts.output_paths`
- `neuroinfra.artifacts.result_artifacts`

The next standardized seam is also in place:

- `neuroinfra.remote.config`
- `neuroinfra.remote.helper_bundle`
- `neuroinfra.remote.command_launch`
- `neuroinfra.models.registry`
- `neuroinfra.campaigns.store`
- `neuroinfra.contracts.parameters`
- `neuroinfra.contracts.visuals`
- `neuroinfra.dashboard.packets`
- `neuroinfra.dashboard.runtime`

It captures:

- candidate reusable subsystems
- current source file locations
- generic capabilities
- repo-specific couplings
- extraction confidence
- proposed extraction phase

## Usage

Text summary:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m neuroinfra
```

JSON inventory:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m neuroinfra --json
```

## Near-term intent

The expected first-wave extractions are:

1. audit core / CLI / registry
2. shared Slurm helper layer

The result-artifact and output-path helpers have already been mirrored into
`neuroinfra.artifacts` and left behind compatibility wrappers under
`olfactorybulb.*`.

The remote Slurm layer is not extracted yet, but the helper-bundle
manifest/signature protocol that packages those scripts for remote upload, plus
the local command builders that launch uploaded or inline helpers, now have a
standardized home under `neuroinfra.remote`.

The remote endpoint parsing, timeout normalization, retry policy, and generic
Paramiko-backed Slurm config builder that the notebook layer uses now also live
under `neuroinfra.remote.config`, while the notebook-facing wrappers remain in
`obgpu_experiment_helpers.py`.

The generic parameter-space and contract helpers that back the HFO optimizer's
search-space registry now live under `neuroinfra.contracts.parameters`, while
the HFO-specific parameter catalog remains in `olfactorybulb.hfo_features`.

The generic visualization-contract metadata types and snapshot builder that
back the HFO packet/dashboard schema now live under
`neuroinfra.contracts.visuals`, while the concrete HFO plot families and
render helpers remain in `olfactorybulb.hfo_visuals`.

The generic manifest-backed packet discovery and stale-packet cleanup helpers
that the HFO dashboard uses now live under `neuroinfra.dashboard.packets`,
while the HFO-specific packet freshness rules and HTML rendering remain in
`tools.analysis.hfo_visual_dashboard`.

The generic sidecar/runtime process primitives that the HFO dashboard uses now
live under `neuroinfra.dashboard.runtime`, while the HFO-specific command
assembly and freshness policy still remain in
`tools.analysis.hfo_visual_dashboard`.

The main file blocking deeper extraction is still:

- `obgpu_experiment_helpers.py`

That file needs to be split by responsibility before a clean reusable package
boundary can exist.
