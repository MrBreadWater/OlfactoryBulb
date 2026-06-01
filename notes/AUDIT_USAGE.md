# Audit Usage

Use the audit entrypoint from the repository root in the `OBGPU` environment:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli --list
```

Run one audit:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli hfo_feature_contracts
```

Run everything:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli all
```

Emit JSON for scripts:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli env_install --json
python -m olfactorybulb.audit.cli all --json
```

## Current audits

- `env_install`: environment, activation hooks, mechanism outputs, loader issues, imports, optional launcher smoke
- `burton_urban_fi`: FI-style cell-behavior audit
- `epli_correctness`: external plexiform layer interneuron integration/correctness checks
- `hfo_feature_contracts`: optimizer/search-space/dashboard/packet contract checks

## Useful options

`env_install`:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli env_install --run-launcher-smoke
python -m olfactorybulb.audit.cli env_install --require-gpu
```

`epli_correctness`:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli epli_correctness --candidate-slice 0
```

`burton_urban_fi`:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli burton_urban_fi --cell-count 8 --cell-types MC TC --dt-ms 0.025
python -m olfactorybulb.audit.cli burton_urban_fi --cell-count 5 --cell-types MC,TC --jobs 0
```

`--jobs 0` means "use all local CPU cores" for the Burton/Urban audit unless
`--use-gpu` is set, in which case the audit is forced back to a single worker.

## Interpreting results

- `PASS`: the check met its criterion
- `WARN`: the check did not prove the condition, but the state is not treated as fatal
- `FAIL`: actionable problem; the CLI exits with code `1`

## Troubleshooting

### `dlopen failed - /tmp/pgcudafat...`

This means a compiled mechanism library was built with a stale NVHPC temporary
object path baked into its NEEDED entries. The `env_install` audit now checks
for this directly under `nvhpc_transient_dependencies`.

Inspect it:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli env_install --run-launcher-smoke
```

Repair the active local library:

```bash
source tools/setup/activate_obgpu.sh OBGPU
bash tools/setup/fix_nvhpc_libnrnmech.sh aarch64/libnrnmech.so
```

Then re-run:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.audit.cli env_install --run-launcher-smoke
```

The warning should disappear once the stale `/tmp/pgcudafat*` dependency is removed.
