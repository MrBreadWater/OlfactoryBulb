# Research Context Layout

`research_context/` mixes four different kinds of artifacts. Treat them
differently.

## 1. Raw downloaded or manually added source material

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
- Do not hand-fix generated canonical CSVs unless the task explicitly calls for
  a deliberate manual correction and the provenance implications are understood.
- Keep stable publisher URLs in `source_url`; redirected object-store fetch
  targets are transport details, not canonical provenance.
- If a source-only gap remains unresolved, record it in the appropriate
  `needs_manual_extraction.csv` file instead of fabricating values.

## Useful commands

Rebuild a dataset:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/extract_reference_dataset.py --dataset-id <id>
```

Verify generated outputs:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python test_reference_data_sanity.py
python tools/run_audit.py reference_dataset_status --dataset-id granule_cells
python tools/run_audit.py reference_dataset_status --dataset-id pv_crh_epl_fsi
```
