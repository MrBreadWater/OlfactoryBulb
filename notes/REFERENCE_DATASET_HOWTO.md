# Reference Dataset How-To

This guide explains how to add and maintain a declarative reference-data
dataset in this repository.

If you also need to run model-side literature validation against those
normalized rows, use the companion guide:
[REFERENCE_VALIDATION_HOWTO.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_HOWTO.md).

The system is meant for structured, protocol-aware validation targets such as:

- intrinsic electrophysiology summary tables
- current-versus-firing-rate point sets
- protocol metadata
- cell-identity and morphology constraints
- validation notes and protocol caveats
- explicit manual-extraction backlog items

It is not a general PDF-mining system. It works best when your source data are
already available as:

- supplemental `csv`
- supplemental `xls` / `xlsx`
- local manually curated tables
- local manually digitized CSVs

It does **not** perform built-in figure digitization or screenshot ingestion.
If a paper only exposes a plotted curve, keep that gap in
`needs_manual_extraction.csv` until a human creates a provenance-preserving CSV.

## Where the pieces live

- Dataset config template:
  [research_context/reference_datasets/TEMPLATE.dataset.toml](/home/michael/OlfactoryBulb/research_context/reference_datasets/TEMPLATE.dataset.toml)
- Current example dataset:
  [research_context/reference_datasets/pv_crh_epl_fsi.dataset.toml](/home/michael/OlfactoryBulb/research_context/reference_datasets/pv_crh_epl_fsi.dataset.toml)
- Generic downloader:
  [tools/download_reference_dataset_sources.py](/home/michael/OlfactoryBulb/tools/download_reference_dataset_sources.py)
- Generic extractor:
  [tools/extract_reference_dataset.py](/home/michael/OlfactoryBulb/tools/extract_reference_dataset.py)
- Generic config loader:
  [olfactorybulb/audit/reference_dataset_config.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_dataset_config.py)
- Generic extraction engine:
  [olfactorybulb/audit/reference_dataset_engine.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_dataset_engine.py)
- Generic source acquisition:
  [olfactorybulb/audit/reference_sources.py](/home/michael/OlfactoryBulb/olfactorybulb/audit/reference_sources.py)

## What a dataset produces

Every dataset config declares output filenames for:

- `ephys`
- `fi_curve`
- `protocols`
- `identity`
- `notes`
- `manual`
- `readme`

Datasets can also declare additional outputs when one project needs multiple
files with the same row shape. The current granule-cell dataset uses:

- `subtype_ephys`
- `subtype_fi_curve`
- `synaptic_latency`
- `modulation`

If an output key does not match a built-in schema name, add an
`[output_schemas]` section that maps output keys to schema presets such as:

- `ephys`
- `fi_curve`
- `protocols`
- `identity`
- `notes`
- `manual`
- dataset-specific presets such as `gc_ephys`, `gc_fi_curve`, `gc_protocols`,
  and `gc_identity`

For the current EPL-FSI dataset those become:

- `PV_CRH_EPL_FSI_ephys.csv`
- `PV_CRH_EPL_FSI_fI_curve.csv`
- `PV_CRH_EPL_FSI_protocols.csv`
- `PV_CRH_EPL_FSI_identity.csv`
- `validation_notes.csv`
- `needs_manual_extraction.csv`
- `PV_CRH_EPL_FSI_extraction_README.md`

For the current GC dataset those become:

- `GC_ephys.csv`
- `GC_fI_curve.csv`
- `GC_sGC_dGC_ephys.csv`
- `GC_sGC_dGC_fI_curve.csv`
- `GC_protocols.csv`
- `GC_identity_morphology.csv`
- `GC_synaptic_latency_references.csv`
- `GC_modulation_references.csv`
- `GC_validation_notes.csv`
- `GC_needs_manual_extraction.csv`
- `GC_extraction_README.md`

## Quick start

### 1. Copy the template

Create a new dataset config under `research_context/reference_datasets/`.

Example:

```bash
cp \
  research_context/reference_datasets/TEMPLATE.dataset.toml \
  research_context/reference_datasets/my_new_dataset.dataset.toml
```

### 2. Fill in the required sections

The template already leaves required sections uncommented. At minimum you must
set:

- `dataset_id`
- `dataset_name`
- `source_data_subdir`
- `[outputs]`
- `[output_schemas]` if you use nonstandard output keys
- `[readme]`
- at least one `[[sources]]`
- at least one `[[static_protocol_rows]]`
- at least one `[[static_note_rows]]` if protocol caveats must survive into
  downstream validation
- at least one extraction rule such as `[[summary_rules]]`,
  `[[formatted_summary_rules]]`, or `[[point_rules]]`

### 3. Download sources

Always use the `OBGPU` environment in this repo.

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/download_reference_dataset_sources.py --dataset-id my_new_dataset
```

If you want to target a config file directly instead of a dataset id:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/download_reference_dataset_sources.py \
  --config-path research_context/reference_datasets/my_new_dataset.dataset.toml
```

### 4. Extract normalized outputs

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/extract_reference_dataset.py --dataset-id my_new_dataset
```

This will:

- load the dataset config
- ensure required downloadable sources exist locally
- apply the declared row rules
- write the normalized CSV outputs
- write the dataset README

### 5. Verify the outputs

At minimum, check:

- all expected files were written
- `fi_curve` is non-empty only when you truly have current-rate points
- `protocol_id` is present on all f-I rows
- `source_url` preserves the stable source URL, not a temporary redirect target
- `needs_manual_extraction.csv` captures anything unresolved instead of guessing

## Minimal working dataset

The smallest realistic dataset usually contains:

1. one machine-readable source table
2. one protocol row
3. one protocol-caveat note
4. one `summary_rules` entry

For example:

```toml
dataset_id = "example_dataset"
dataset_name = "Example reference dataset"
source_data_subdir = "example_dataset"

[outputs]
ephys = "EXAMPLE_ephys.csv"
fi_curve = "EXAMPLE_fI_curve.csv"
protocols = "EXAMPLE_protocols.csv"
identity = "EXAMPLE_identity.csv"
notes = "EXAMPLE_validation_notes.csv"
manual = "EXAMPLE_needs_manual_extraction.csv"
readme = "EXAMPLE_extraction_README.md"

[readme]
title = "Example reference-data extraction"
summary = "One-line summary."
source_summary = ["Primary supplemental table."]
suitable_now = ["Intrinsic-property rows from the primary table."]
caveats = ["Protocol caveats must remain visible in downstream validation."]

[[sources]]
source_id = "example_primary_table"
source = "Example et al. (2026)"
label = "Example supplemental table"
filename = "example_table.csv"
source_url = "https://example.org/example_table.csv"
downloadable = true
required = true
expected_extension = ".csv"

# Optional: use download_url when the stable citation URL differs from the
# direct download endpoint.
# download_url = "https://example.org/download/example_table.csv"

[[static_protocol_rows]]
protocol_id = "EXAMPLE_PROTOCOL"
source = "Example et al. (2026)"
cell_type = "Example Cell"
marker_profile = "Example Marker"
stimulus_type = "somatic depolarizing current step"
step_duration_ms = 1000.0
current_start_pA = 0.0
current_stop_pA = 400.0
current_step_pA = 50.0
current_values_pA = "0;50;100;150;200;250;300;350;400"
rate_definition = "Mean firing rate over the full current step."
spike_detection_rule = "Describe the rule if known."
baseline_or_holding_vm_mV = -65.0
synaptic_blockers = ""
temperature_C = ""
compatible_group = "example_fI"
notes = "Protocol metadata row."

[[static_note_rows]]
note_id = "N_EXAMPLE_PROTOCOL_CAVEAT"
severity = "warning"
scope = "fI_validation"
target_type = "protocol"
target = "EXAMPLE_PROTOCOL"
message = "Explain the protocol caveat here."
display_order = 10
source = "Example et al. (2026)"
source_location = "Protocol metadata rows"

[[summary_rules]]
output = "ephys"
source_id = "example_primary_table"
sheet_name = "Sheet1"
row_filter_column = "cell_type"
row_filter_equals = "Example Cell"
column = "rest_mV"
property_name = "Membrane Resting Voltage"
reported_definition = "resting potential"
unit = "mV"
data_kind = "intrinsic_property"
notes = "Computed directly from source rows."
cell_type = "Example Cell"
marker_profile = "Example Marker"
```

## Supported source types

### Downloadable sources

Use `downloadable = true` when the engine should fetch the file.

Example:

```toml
[[sources]]
source_id = "paper_supplement"
filename = "paper_supplement.xls"
source_url = "https://example.org/paper_supplement.xls"
downloadable = true
required = true
expected_extension = ".xls"
```

The downloader:

- follows redirects with `allow_redirects=True`
- stores the file under
  `research_context/source_data/<source_data_subdir>/`
- preserves the stable `source_url` from the config in downstream outputs
- optionally uses `download_url` for the transfer while still preserving the
  stable `source_url` in the normalized rows
- falls back to `curl` with a browser-like user agent when a site returns an
  HTML block page or challenge page instead of the requested file

### Local-only sources

Use `downloadable = false` when the file is local and should not be fetched.

Example:

```toml
[[sources]]
source_id = "local_manual_csv"
filename = "manually_curated_points.csv"
source_url = "local://manually_curated_points.csv"
downloadable = false
required = false
expected_extension = ".csv"
```

This is the right path for:

- local manually curated tables
- local manually digitized point CSVs
- local PDFs you want to cite but not auto-download

## How the engine builds rows

The engine supports several row families.

### 1. `static_*_rows`

Use static rows when the values are already known and you want to write them
directly into the normalized outputs.

Available families:

- `[[static_ephys_rows]]`
- `[[static_identity_rows]]`
- `[[static_protocol_rows]]`
- `[[static_note_rows]]`
- `[[static_manual_rows]]`

Use these for:

- values reported directly in text
- hand-curated summary values
- protocol metadata
- notes that must travel with the dataset
- manual-extraction backlog entries

### 2. `summary_rules`

Use `[[summary_rules]]` when a source sheet contains per-cell or per-row values
that should be summarized into a mean/SD/SEM row.

The engine will:

- load the source table
- optionally filter rows
- pull the requested numeric column
- compute `mean`, `sd`, `sem`, and `n`
- emit a normalized row into either `ephys` or `identity`

Required fields for a typical rule:

- `output`
- `source_id`
- `sheet_name` if the file is multi-sheet
- `column`
- `property_name`
- `unit`
- `data_kind`
- `cell_type`
- `marker_profile`

Optional filter styles:

```toml
row_filter_column = "cell_type"
row_filter_equals = "FSI"
```

or:

```toml
row_filters = [
  { column = "cell_type", equals = "FSI" },
  { column = "condition", equals = "control" },
]
```

Optional numeric transform:

```toml
transform_scale = 1000.0
```

This is useful for unit conversions such as:

- `Hz/pA -> Hz/nA`

### 3. `formatted_summary_rules`

Use `[[formatted_summary_rules]]` when a source table already stores values in
formatted cells such as:

- `74.9 ± 25.9 (19)`
- `52.7 +/- 7.6`

This is common in:

- DOCX source-data tables
- supplemental tables exported from journal sites
- HTML tables where each cohort is a separate column

The engine will:

- read the table as text
- normalize property labels
- parse mean/spread/sample-count from the formatted cell text
- emit canonical rows into the requested output

Typical fields:

- `output`
- `source_id`
- `table_index`
- `value_column`
- `property_map`
- optional `unit_map`
- `source_location`

Use `property_map` to normalize paper-facing labels into canonical property
names, and `unit_map` when units should be overridden for specific labels.

### 4. `point_rules`

Use `[[point_rules]]` when a source table contains actual current-rate points.

This is the only rule family that should populate `fi_curve`.

Do **not** use it for:

- max firing rate summary values
- FI gain summary values
- rheobase-only values

Those belong in `ephys`, not `fi_curve`.

Typical required fields:

- `source_id`
- `sheet_name`
- `current_column`
- `value_column`
- `cell_type`
- `marker_profile`
- `protocol_id`
- `rate_definition`

Useful optional fields:

- `cell_id`
- `sample_scope`
- `current_min_pA`
- `current_max_pA`
- `step_duration_ms`
- `baseline_or_holding_vm_mV`
- `note_ids`

### 5. Conditional rows

Use:

- `[[conditional_note_rows]]`
- `[[conditional_manual_rows]]`

when a note or backlog item should only appear if something goes wrong or if a
particular output ended up empty.

The current engine supports these conditions:

- `condition = "missing_source"`
- `condition = "output_empty"`

Examples:

- add a warning note if a required supplement could not be downloaded
- add a manual-extraction row if no valid `fi_curve` rows were produced

When the note/manual row belongs in one output file but the emptiness check
should watch another output, set:

```toml
condition = "output_empty"
condition_output = "fi_curve"
```

That lets a dataset write a note into `notes` when `fi_curve` is empty without
pretending that the `notes` output itself is empty.

## Notes and protocol caveats

Do not hide protocol incompatibilities in free text.

Use explicit note rows and attach them with `note_ids` where needed.

Typical pattern:

1. create a protocol note in `[[static_note_rows]]`
2. attach its `note_id` to:
   - `summary_rules` that produce f-I summary metrics
   - `point_rules` that produce `fi_curve` rows
   - any downstream validation output that compares across protocols

This is how the repo currently keeps
`N_FI_PROTOCOL_DIFFERENCE` visible when MC/TC Burton 2014 data and EPL-FSI
Burton 2024 data appear together.

## Source provenance rules

Every normalized row should preserve:

- `source`
- `source_file`
- `source_location`
- `source_url`
- `extraction_method`

Do not replace the stable source URL with a transient redirect target.

For downloadable sources:

- `source_url` in the dataset config should always be the stable article or
  supplement URL
- the downloader may follow redirects during transfer
- the CSV rows should still keep the stable URL

## Choosing the right output file

Use:

- `ephys` for summary intrinsic/electrophysiology metrics
- `fi_curve` for actual current-vs-rate points only
- `protocols` for stimulation protocol definitions
- `identity` for marker/morphology/axonless/population constraints
- `notes` for visible validation caveats
- `manual` for unresolved extraction tasks
- dataset-specific extra outputs such as `subtype_ephys`, `modulation`, or
  `synaptic_latency` when those rows should stay separate from baseline
  intrinsic validation

If you are unsure whether something belongs in `fi_curve`, use this rule:

If it is not a real pair or series of current values and firing-rate values, it
does **not** belong in `fi_curve`.

## Manual data is still allowed

The engine does not force every source to be remote and machine-readable.

Good uses of local manual sources:

- a manually curated CSV copied from a supplemental table
- a manually digitized CSV with provenance
- a local PDF that you cite in static rows

The point is not to eliminate human judgment. The point is to make the final
normalized outputs reproducible once the human-curated input exists.

## Common mistakes

### Mistake 1: putting summary values into `fi_curve`

Wrong:

- max firing rate only
- FI slope only
- rheobase only

Right:

- keep those in `ephys`
- reserve `fi_curve` for actual current-rate pairs

### Mistake 2: silently pooling incompatible populations

Do not collapse:

- PV+
- CRH+
- PV/CRH-overlap

unless the source and protocol truly describe the same population.

Keep them separate using:

- `cell_type`
- `marker_profile`
- `protocol_id`

### Mistake 3: losing protocol caveats

If two cell classes use different stimulation protocols, that caveat must stay
explicit in:

- `validation_notes.csv`
- `note_ids`
- downstream rendered outputs

### Mistake 4: guessing missing values

If a value cannot be extracted reliably:

- do not invent it
- do not back-project it from a summary metric
- put the missing work in `needs_manual_extraction.csv`

### Mistake 5: relying on PDFs when a spreadsheet exists

Prefer:

1. `csv/xls/xlsx`
2. structured tables
3. local manual CSVs
4. PDF text only when needed

## Recommended workflow for a new paper

1. Decide whether the paper contributes:
   - `ephys`
   - `fi_curve`
   - `identity`
   - `protocols`
   - `notes`
2. Add a `[[sources]]` entry for every source file you will cite.
3. Add protocol rows first.
4. Add note rows for protocol caveats before adding f-I rows.
5. Add `summary_rules` for machine-readable summary data.
6. Add `point_rules` only for real current-rate points.
7. Add `static_*_rows` for direct text-derived or hand-curated values.
8. Add `conditional_manual_rows` for any source that might fail acquisition.
9. If a source uses cohort-style formatted tables, prefer
   `formatted_summary_rules` over handwritten parsing code.
10. Run the downloader.
11. Run the extractor.
12. Inspect the generated CSVs and README.
13. Run the dataset-specific tests.

## Current example commands

Using the current EPL-FSI dataset:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/download_reference_dataset_sources.py --dataset-id pv_crh_epl_fsi
python tools/extract_reference_dataset.py --dataset-id pv_crh_epl_fsi
python test_reference_dataset_engine.py
python test_download_epl_fsi_reference_sources.py
python test_pv_crh_epl_fsi_reference_data.py
python tools/verify_pv_crh_epl_fsi_reference_data.py
```

Using the current GC dataset:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/download_reference_dataset_sources.py --dataset-id granule_cells
python tools/extract_reference_dataset.py --dataset-id granule_cells
python test_reference_dataset_engine.py
python test_download_gc_reference_sources.py
python test_gc_reference_data.py
python tools/verify_gc_reference_data.py
```

Backward-compatible wrappers still exist:

```bash
source tools/setup/activate_obgpu.sh OBGPU
python tools/download_epl_fsi_reference_sources.py
python tools/extract_pv_crh_epl_fsi_reference_data.py
python tools/download_gc_reference_sources.py
python tools/extract_gc_reference_data.py
```

## When to add code versus when to add config

Add config when:

- you are adding a new source
- you are mapping a new source column
- you are declaring a new protocol
- you are adding a new note
- you are adding a new backlog/manual-extraction entry

Add code only when:

- the dataset engine cannot express the source structure
- the file format is unsupported
- a new kind of conditional behavior is needed
- a new reusable rule family is justified

That is the core discipline here: prefer extending the config schema over
writing another one-off extractor script.
