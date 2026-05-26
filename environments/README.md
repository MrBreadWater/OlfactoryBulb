# Environments

The maintained environment spec is:

- `environment-modern.yml`

It is used by `install-obgpu.sh` and `tools/setup/setup_ob_modern.sh` to create
or update the `OBGPU` conda environment for the source-built
NEURON/CoreNEURON workflow.

Historical specs remain for reference:

- `environment.yml`: legacy `OB` environment used during the first NEURON port.
- `environment-lock.yml`: frozen export of that legacy environment.
- `environment-linux-aarch64-explicit.txt`: explicit package export from an
  older Linux `aarch64` host.

Do not update the legacy specs unless you are intentionally preserving or
reproducing the old `OB` environment. New work should change
`environment-modern.yml`.

The `OBGPU` build comes from:

- pinned upstream ref: `third_party_patches/nrn/manifest.json`
- repo patch stack: `third_party_patches/nrn/`
- bootstrap script: `tools/setup/setup_ob_modern.sh`
- generic activation helper: `tools/setup/activate_obgpu.sh`
- Sol activation helper: `tools/setup/activate_sol_obgpu.sh`

The remote notebook backend uses Paramiko, so `paramiko` is part of the modern
spec. The old OpenSSH/pexpect multiplex path is retired.

See also:

- `notes/porting/MODERN_NEURON_PORT_NOTES.md`
- `notes/porting/SOL_REMOTE_WORKFLOW.md`
