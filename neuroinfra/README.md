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

The main file blocking deeper extraction is still:

- `obgpu_experiment_helpers.py`

That file needs to be split by responsibility before a clean reusable package
boundary can exist.
