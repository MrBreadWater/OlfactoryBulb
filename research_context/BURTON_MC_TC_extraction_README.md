# Burton 2014 MC/TC reference-data extraction

This dataset converts the committed Burton & Urban 2014 mitral-cell and tufted-cell summary CSVs into one canonical principal-cell electrophysiology bundle.

## Source summary

- The maintained inputs are the committed mitral-cell and tufted-cell legacy summary CSVs derived from Burton & Urban 2014.

## Suitable now

- Canonical MC/TC intrinsic-property and firing-rate summary rows for declarative validation.
- Canonical protocol metadata for the shared two-second current-clamp family used by those summary rows.

## Caveats

- These source CSVs are already curated summary tables, not raw per-cell recordings.
- The maintained bundle contains summary metrics only; no machine-readable pointwise MC/TC f-I curve source is included here.
- Current-clamp metrics remain tagged to BU2014_MC_TC_2s_0_300pA_50pA so downstream validations can keep protocol caveats explicit.

## Extraction status

- `ephys` rows: 60
- `protocols` rows: 1
- `manual` rows: 1
- `readme` rows: 0
- Missing required sources after acquisition: none
