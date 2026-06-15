# Research Context Layout

`research_context/` mixes four different kinds of artifacts. Treat them
differently.

## 1. Raw/source files

Examples:

- local paper PDFs
- downloaded supplemental spreadsheets
- `source_data/`

These are provenance inputs. Do not silently rewrite or normalize them in
place.

## 2. Manual intake artifacts

Examples:

- `manual_reference_templates/`
- manually curated intake CSVs created from literature review

These are raw human-curated inputs. They are not the same thing as the
canonical normalized validation bundles.

## 3. Declarative configs

Examples:

- `reference_datasets/`
- `reference_validations/`

These are the real source of truth for the generic literature-data and
literature-validation systems.

If a generated output looks wrong, prefer changing one of:

- the dataset config
- the validation config
- the extractor/validation engine
- the raw source file
- the manual intake artifact

and then regenerate.

## 4. Generated canonical outputs

Examples:

- `GC_*.csv`
- `GC_*_README.md`
- `PV_CRH_EPL_FSI_*.csv`
- `PV_CRH_EPL_FSI_*_README.md`
- `validation_notes.csv`
- `needs_manual_extraction.csv`
- `GC_validation_notes.csv`
- `GC_needs_manual_extraction.csv`

These are generated artifacts produced by the reference-data pipeline.

They should usually be treated as:

- reviewable
- diffable
- regenerable

but **not** as the first place to hand-edit data.

## Editing rules

- Do not guess missing numeric values.
- Do not hand-edit generated canonical outputs unless the task explicitly calls
  for a deliberate manual correction and the provenance implications are
  understood.
- Keep stable publisher URLs in `source_url`; redirected object-store fetch
  targets are transport details, not canonical provenance.
- If a source-only gap remains unresolved, record it in the appropriate
  `needs_manual_extraction.csv` file instead of fabricating values.
- If a maintained reference source is already a small committed summary CSV,
  put that file under `source_data/<dataset>/` and route it through the
  dataset config with `legacy_summary_csv_rules` rather than keeping a
  repo-side one-off loader.

## Useful commands

Rebuild a dataset:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/extract_reference_dataset.py --dataset-id <id>
```

Verify generated outputs:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m tests.reference.test_reference_data_sanity
python tools/run_audit.py test_suite_status --suite reference_bundles
python tools/run_audit.py reference_dataset_contracts --dataset-id burton_mc_tc_principal_cells
python tools/run_audit.py reference_dataset_status --dataset-id burton_mc_tc_principal_cells
python tools/run_audit.py reference_dataset_contracts --dataset-id granule_cells
python tools/run_audit.py reference_dataset_status --dataset-id granule_cells
python tools/run_audit.py reference_dataset_contracts --dataset-id pv_crh_epl_fsi
python tools/run_audit.py reference_dataset_status --dataset-id pv_crh_epl_fsi
```

View the resulting audits and optimization outputs together in the maintained
web shell:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python -m olfactorybulb.dashboard.control_center
```

That serves immediately at `http://127.0.0.1:6006/` by default, falls forward
to the next available local port if `6006` is already busy, and fills the
audit/optimization tabs as their background renders complete.
