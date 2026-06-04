# Reference Validation System Overview

Start here if you want the simplest explanation of what this system is and why
it exists.

If you later need the implementation details:

- full paper-to-audit tutorial and field reference: [REFERENCE_VALIDATION_TUTORIAL.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_TUTORIAL.md)
- dataset ingestion: [REFERENCE_DATASET_HOWTO.md](/home/michael/OlfactoryBulb/notes/REFERENCE_DATASET_HOWTO.md)
- simulation-backed validation: [REFERENCE_VALIDATION_HOWTO.md](/home/michael/OlfactoryBulb/notes/REFERENCE_VALIDATION_HOWTO.md)
- manual intake templates: [manual_reference_templates/README.md](/home/michael/OlfactoryBulb/research_context/manual_reference_templates/README.md)

## What This System Is

This repository has a structured reference-validation system for literature-backed
model checks.

It does three separate jobs:

1. acquire and normalize reference data from papers and supplements
2. preserve protocol caveats and source traceability
3. run model-side validation against those normalized references

The key idea is that the literature data are not just pasted into ad hoc CSVs
or notebooks. They are converted into explicit, reusable validation targets.

## Why We Needed It

Without this system, the same recurring problems kept showing up:

- values from different papers were easy to mix without noticing
- cell types and subtypes could get silently pooled
- protocol differences were easy to forget
- it was hard to tell where a given number came from
- updating or extending a validation target required bespoke code

That makes the validation layer fragile and difficult to trust.

## Why It Is Worth Using

The system is worth using because it makes the literature layer:

- explicit
- reproducible
- reviewable
- protocol-aware
- extensible

In practice that means:

- every row can carry source identity and source location
- protocol caveats travel with the data
- different cell classes can stay separate by default
- validation rules can be upgraded without rewriting the whole pipeline

## Why "Expert Review" Exists

This repository intentionally treats LLM-assisted authoring as a first-class
workflow option for drafting datasets, validation configs, rule choices,
protocol mappings, and documentation.

That is useful for speed, but it is not enough for scientific trust.

Large language models can help assemble a candidate validation design, but they
are not reliable scientific reviewers. They can misread papers, flatten
protocol caveats, overstate comparability, or quietly normalize a questionable
assumption into something that looks clean in config.

That is why the validation-design review metadata exists.

When these docs say `expert` or `qualified_domain_expert`, they mean a person
with the relevant scientific background to judge the validation-design choice
itself, such as protocol equivalence, reference-band reconstruction, pooling
strategy, or manual extraction/mapping decisions.

It does not mean:

- the LLM drafted the item and therefore it is reviewed
- the current PASS/WARN/FAIL result is scientifically endorsed
- every line of implementation code was audited manually

## Simplest Explanation Of How To Use It

### Fast path

If you already have a paper with usable tables or source data:

1. make sure the source files are available locally or downloadable
2. define or update the dataset config under `research_context/reference_datasets/`
3. run the dataset extractor
4. run the validation config that consumes those normalized rows

That is the whole idea.

### Optional details

If the paper is messy:

- use a manual intake table instead of guessing
- keep the raw extracted values and provenance separate from the canonical files
- add protocol notes if the paper is not directly comparable to an existing protocol

## How It Is Structured Internally

At a high level there are five layers.

### 1. Source files

These are the raw inputs:

- PDFs
- supplemental XLS/XLSX/CSV
- local manually curated tables

### 2. Dataset configs

These live under:

- `research_context/reference_datasets/`

They declare:

- which sources belong to a dataset
- which outputs should be produced
- how summary tables and point tables are mapped
- which notes and manual-extraction gaps should exist

### 3. Normalized reference outputs

These are the canonical CSVs used downstream, for example:

- `*_ephys.csv`
- `*_fI_curve.csv`
- `*_protocols.csv`
- `*_identity.csv`
- `validation_notes.csv`
- `needs_manual_extraction.csv`

### 4. Validation configs

These live under:

- `research_context/reference_validations/`

They declare:

- which protocol runner to execute on the model side
- which normalized reference rows to compare against
- which rule checks should judge the result

### 5. Validation engine

This is the runtime layer that:

- runs the model-side protocol
- extracts measurements
- applies rule logic
- renders the final audit items

## What “Provenance” Means Here

In this project, provenance means the traceable origin of a value.

A good row answers:

- which paper?
- which file?
- which figure, table, sheet, or paragraph?
- how was the value obtained?
- what transformation, if any, was applied?

That matters because otherwise the validation layer becomes a pile of numbers
that cannot be audited later.

## What The Manual Templates Are For

Some papers do not expose clean machine-readable tables for everything.

When that happens, the system should not guess.

Instead, we keep unresolved items in:

- `needs_manual_extraction.csv`

and use the manual intake templates under:

- `research_context/manual_reference_templates/`

Those templates are for raw human-curated intake. They are not the final
canonical validation outputs.

## When To Use The Optional Layers

Use the full detailed HOWTOs only if you need to:

- add a new dataset config
- add a new validation config
- register a new protocol runner
- register a new validation rule

If you only want to contribute missing literature values, the manual templates
are the shortest path.
