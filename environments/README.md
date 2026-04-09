# Environments

Environment specs and exports used during the port:

- `environment-modern.yml`
  Maintainable `OBGPU` environment spec used by the source-built NEURON/CoreNEURON workflow.
- `environment.yml`
  Maintainable legacy `OB` environment spec.
- `environment-lock.yml`
  Frozen export of the working legacy environment.
- `environment-linux-aarch64-explicit.txt`
  Explicit package export for this Jetson/Linux `aarch64` host.

`OBGPU` is no longer tied to a hand-edited vendored NEURON tree. The supported
source build now comes from:

- pinned upstream ref: [third_party_patches/nrn/manifest.json](/home/alek/OlfactoryBulb/third_party_patches/nrn/manifest.json)
- repo patch stack: [third_party_patches/nrn](/home/alek/OlfactoryBulb/third_party_patches/nrn)
- bootstrap script: [setup_ob_modern.sh](/home/alek/OlfactoryBulb/tools/setup/setup_ob_modern.sh)

See also:

- [MODERN_NEURON_PORT_NOTES.md](/home/alek/OlfactoryBulb/notes/porting/MODERN_NEURON_PORT_NOTES.md)
- [SOL_REMOTE_WORKFLOW.md](/home/alek/OlfactoryBulb/notes/porting/SOL_REMOTE_WORKFLOW.md)
