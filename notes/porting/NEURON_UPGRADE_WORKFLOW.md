# NEURON Upgrade Workflow

## Supported Model

The supported OBGPU build is:

- one pinned upstream `nrn` ref
- one repo-managed patch manifest
- one explicit acceptance workflow for future bumps

The pinned metadata lives in:

- [third_party_patches/nrn/manifest.json](/home/alek/OlfactoryBulb/third_party_patches/nrn/manifest.json)

The patch files live in:

- [third_party_patches/nrn](/home/alek/OlfactoryBulb/third_party_patches/nrn)

## Why This Exists

This avoids another “custom fork frozen forever” situation.

The repo no longer depends on “whatever edits currently exist under `external/nrn-9.0.1`.” Instead, that tree can be rebuilt from:

1. the pinned upstream ref
2. the ordered patch stack

## Upgrade Gate

Use the upgrade helper:

```bash
python tools/setup/check_nrn_upgrade.py --candidate-ref <tag-or-commit> --skip-build
```

That checks:

1. clean candidate checkout
2. patch stack replay

For a real upgrade test, run build + smokes:

```bash
python tools/setup/check_nrn_upgrade.py --candidate-ref <tag-or-commit> --enable-gpu
```

By default, that runs:

- import sanity
- a short `OneMsTest` benchmark

You can add extra smoke commands with repeated `--smoke-command`.

## Acceptance Rule

A candidate upstream ref is accepted only if:

1. the patch stack applies cleanly
2. OBGPU rebuilds successfully
3. the smoke matrix passes
4. any required parity spot checks pass

If any step fails:

- the candidate is rejected
- the current pinned ref remains supported

## After A Successful Upgrade

Only then should you:

1. update `upstream_ref` in [manifest.json](/home/alek/OlfactoryBulb/third_party_patches/nrn/manifest.json)
2. regenerate patch files if the patch stack changed
3. rerun the documented smoke/parity checks
4. commit the manifest/patch update as one explicit upgrade change
