# OlfactoryBulb Parameter Surface Reference

_Generated on 2026-03-23 from repository state._

Status as of 2026-05-26: keep this as a generated reference snapshot, not as the
canonical runtime schema. The live notebook-facing controls are documented by
`obgpu_experiment_helpers.control_help()` and the maintained config builders.
Core biological parameter surfaces below are still useful, but newer OBGPU
options such as remote Slurm, artifact formats, and lazy soma loading are not
fully represented here.

This document is organized as a practical reference for all adjustable parameters currently exposed by this project for:

- Full network runs (`runbatch.py` / `initslice.py` -> `olfactorybulb.model.OlfactoryBulb`)
- Slice asset JSONs (`olfactorybulb/slices/DorsalColumnSlice`)
- Intrinsic single-cell model knobs used by the network (`prev_ob_models/Birgiolas2020/isolated_cells.py`)

## Contents

1. [Parameter Flow (What Actually Drives a Run)](#1-parameter-flow-what-actually-drives-a-run)
2. [Canonical Network Parameter Schema (`params`)](#2-canonical-network-parameter-schema-params)
3. [Paramset Library and Sweep Families](#3-paramset-library-and-sweep-families)
4. [Slice JSON Parameter Surfaces (DorsalColumnSlice)](#4-slice-json-parameter-surfaces-dorsalcolumnslice)
5. [Odor Database-Constrained Domains](#5-odor-database-constrained-domains)
6. [Cell Intrinsic Parameter Surfaces (Birgiolas2020)](#6-cell-intrinsic-parameter-surfaces-birgiolas2020)
7. [Appendix: Exact `param_values` Vectors for Used Models](#7-appendix-exact-param_values-vectors-for-used-models)

## 1. Parameter Flow (What Actually Drives a Run)

1. `runbatch.py` chooses one or more paramset class names (for example `GammaSignature`).
2. `initslice.py -paramset <ClassName>` passes that class name to `OlfactoryBulb(params=...)`.
3. `OlfactoryBulb.__init__` reads those fields and applies them to:
   - Simulation runtime (`h.dt`, `tstop`, recording cadence)
   - Gap-junction conductances
   - Global synapse mechanism attributes (`AmpaNmdaSyn`, `GabaSyn`)
   - Odor input schedule and strengths
   - LFP electrode position and soma recordings
4. Geometry and fixed synapse topology come from slice JSON files (`MCs/TCs/GCs`, `GCs__MCs`, `GCs__TCs`, `glom_cells`).
5. Intrinsic membrane/channel parameters come from the selected cell classes (`MC4`, `MC5`, `TC3`, etc.) and their `param_values` vectors.

## 2. Canonical Network Parameter Schema (`params`)

The following fields are read by `olfactorybulb/model.py` from the paramset object:

`gap_juction_gmax`, `inhale_duration`, `input_odors`, `input_syn_tau1`, `input_syn_tau2`, `lfp_electrode_location`, `max_firing_rate`, `mc_input_delay`, `mc_input_weight`, `name`, `record_from_somas`, `recording_period`, `rnd_seed`, `sim_dt`, `slice_dir`, `slice_name`, `synapse_properties`, `tc_input_delay`, `tc_input_weight`, `tstop`

### 2.1 Runtime and recording fields

| Field | Type | Project domain | Used in methods |
|---|---|---|---|
| `rnd_seed` | `int` | `[0]` | `__init__` |
| `slice_dir` | `string` | `["olfactorybulb/slices"]` | `__init__` |
| `slice_name` | `string` | `["DorsalColumnSlice"]` | `__init__` |
| `sim_dt` | `float` | `[0.1]` | `run` |
| `recording_period` | `float` | `[0.1]` | `__init__, record_from_somas` |
| `tstop` | `number` | `[1, 1800, 400, 800.1]` | `__init__` |
| `record_from_somas` | `list` | `[('MC', 'TC', 'GC')]` | `__init__` |
| `lfp_electrode_location` | `list` | `[(116, 1078, -61)]` | `__init__` |

### 2.2 Odor drive and principal-cell input fields

| Field | Type | Project domain | Used in methods |
|---|---|---|---|
| `inhale_duration` | `int` | `[125]` | `stim_glom_segments` |
| `max_firing_rate` | `int` | `[150]` | `stim_glom_segments` |
| `input_syn_tau1` | `int` | `[6]` | `stim_glom_segments` |
| `input_syn_tau2` | `int` | `[12]` | `stim_glom_segments` |
| `mc_input_delay` | `int` | `[0, 50]` | `stim_glom_segments` |
| `mc_input_weight` | `number` | `[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1]` | `stim_glom_segments` |
| `tc_input_delay` | `int` | `[0]` | `stim_glom_segments` |
| `tc_input_weight` | `number` | `[0, 0.2, 0.4, 0.6, 0.8, 1]` | `stim_glom_segments` |

### 2.3 Connectivity override fields

- `gap_juction_gmax`
  - `MC`: `[0, 1, 2, 4, 8, 16, 32, 64, 128]`
  - `TC`: `[0, 1, 2, 4, 8, 16, 32, 64, 128]`
- `synapse_properties`
  - `AmpaNmdaSyn.gmax`: `[0, 1, 2, 4, 8, 16, 32, 64, 128, 256]`
  - `AmpaNmdaSyn.ltdinvl`: `[0]`
  - `AmpaNmdaSyn.ltpinvl`: `[0]`
  - `GabaSyn.gmax`: `[0, 1, 2, 4, 8]`
  - `GabaSyn.ltdinvl`: `[0]`
  - `GabaSyn.ltpinvl`: `[0]`
  - `GabaSyn.tau2`: `[16, 36, 100]`

### 2.4 Odor schedule field (`input_odors`)

- Schema per onset key: `{ "name": <string>, "rel_conc": <number> }`
- Onset keys used by project paramsets: `[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]` ms
- `name` values used by project paramsets: `['Apple', 'Coffee', 'Mint']`
- `rel_conc` values used by project paramsets: `[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]`

## 3. Paramset Library and Sweep Families

Total `SilentNetwork` subclasses discoverable: **57**

### 3.1 Class inventory by file

#### base.py

`ParameterSetBase`, `SilentNetwork`

#### case_studies.py

`GammaSignature`, `GammaSignature_DifferentOdor`, `GammaSignature_DifferentOdorConc`, `GammaSignature_EqualTCMCInputs`, `GammaSignature_NoInhibition`, `GammaSignature_NoMCGJs`, `GammaSignature_NoTCGJs`, `MC_TC_Combined_Base`, `MCsWithGJsGCs`, `OneMsTest`, `PureMCs`, `PureMCsWithGJs`, `PureTCs`, `PureTCsWithGJs`, `TCsWithGJsGCs`

#### sensitivity.py

`GammaSignature_AMPANMDA_0`, `GammaSignature_AMPANMDA_1`, `GammaSignature_AMPANMDA_128`, `GammaSignature_AMPANMDA_16`, `GammaSignature_AMPANMDA_2`, `GammaSignature_AMPANMDA_256`, `GammaSignature_AMPANMDA_32`, `GammaSignature_AMPANMDA_4`, `GammaSignature_AMPANMDA_64`, `GammaSignature_AMPANMDA_8`, `GammaSignature_GABA_0`, `GammaSignature_GABA_1`, `GammaSignature_GABA_2`, `GammaSignature_GABA_4`, `GammaSignature_GABA_8`, `GammaSignature_GJ_0`, `GammaSignature_GJ_1`, `GammaSignature_GJ_128`, `GammaSignature_GJ_16`, `GammaSignature_GJ_2`, `GammaSignature_GJ_32`, `GammaSignature_GJ_4`, `GammaSignature_GJ_64`, `GammaSignature_GJ_8`, `GammaSignature_MCWGHT_00`, `GammaSignature_MCWGHT_01`, `GammaSignature_MCWGHT_015`, `GammaSignature_MCWGHT_02`, `GammaSignature_MCWGHT_025`, `GammaSignature_MCWGHT_03`, `GammaSignature_MCWGHT_04`, `GammaSignature_MCWGHT_06`, `GammaSignature_MCWGHT_08`, `GammaSignature_MCWGHT_10`, `GammaSignature_TCWGHT_00`, `GammaSignature_TCWGHT_02`, `GammaSignature_TCWGHT_04`, `GammaSignature_TCWGHT_06`, `GammaSignature_TCWGHT_08`, `GammaSignature_TCWGHT_10`

### 3.2 Sensitivity sweep families (organized by controlled axis)

| Family | Controlled parameter(s) | Value set | Class count |
|---|---|---|---|
| Gap Junction Sweep | `gap_juction_gmax.MC and gap_juction_gmax.TC` | `[0, 1, 2, 4, 8, 16, 32, 64, 128]` | `9` |
| AMPA/NMDA Conductance Sweep | `synapse_properties.AmpaNmdaSyn.gmax` | `[0, 1, 2, 4, 8, 16, 32, 64, 128, 256]` | `10` |
| GABA Conductance Sweep | `synapse_properties.GabaSyn.gmax` | `[0, 1, 2, 4, 8]` | `5` |
| TC Input Weight Sweep | `tc_input_weight` | `[0, 0.2, 0.4, 0.6, 0.8, 1]` | `6` |
| MC Input Weight Sweep | `mc_input_weight` | `[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1]` | `10` |

### 3.3 Paramsets currently used by `runbatch.py`

- `GammaSignature`
- `GammaSignature_NoInhibition`
- `GammaSignature_NoTCGJs`
- `GammaSignature_NoMCGJs`
- `GammaSignature_EqualTCMCInputs`

## 4. Slice JSON Parameter Surfaces (DorsalColumnSlice)

### 4.1 Cell-tree files (`MCs.json`, `TCs.json`, `GCs.json`)

| File | Roots | Model classes | Model counts | `nseg` range |
|---|---|---|---|---|
| `MCs.json` | `10` | `['MC4', 'MC5']` | `{'MC5': 8, 'MC4': 2}` | `1..23` |
| `TCs.json` | `24` | `['TC3', 'TC4', 'TC5']` | `{'TC5': 11, 'TC4': 9, 'TC3': 4}` | `1..13` |
| `GCs.json` | `159` | `['GC1', 'GC2', 'GC3', 'GC4', 'GC5']` | `{'GC3': 87, 'GC5': 21, 'GC1': 29, 'GC4': 17, 'GC2': 5}` | `1..7` |

Common top-level keys in these files:

`import_synapses`, `interaction_granularity`, `name`, `record_activity`, `record_variable`, `recording_granularity`, `recording_period`, `recording_time_start`, `recording_time_end`, `roots`

Current metadata values (same across all three except `name`):

- `import_synapses`: `[False]`
- `interaction_granularity`: `['Section']`
- `record_activity`: `[False]`
- `record_variable`: `['v']`
- `recording_granularity`: `['Cell']`
- `recording_period`: `[1]`
- `recording_time_start`: `[0]`
- `recording_time_end`: `[0]`

Recursive `roots` segment-node schema:

- `name: str`
- `nseg: int`
- `point_count: int`
- `coords: list[float]` (flattened xyz triplets)
- `radii: list[float]`
- `connection_end: number`
- `parent_connection_loc: number`
- `children: list[segment_node]`

### 4.2 Synapse-set files (`GCs__MCs.json`, `GCs__TCs.json`)

#### GCs__MCs.json

- Entries: `2238`
- Entry keys: `['create_spine', 'delay', 'dest_section', 'dest_seg_i', 'dest_syn', 'dest_syn_params', 'dest_x', 'is_reciprocal', 'source_section', 'source_seg_i', 'source_syn', 'source_syn_params', 'source_x', 'threshold', 'weight']`
- Categorical domains (truncated where long):
  - `create_spine`: `[False]`
  - `dest_section`: `['MC4[0].dend[0]', 'MC4[0].dend[10]', 'MC4[0].dend[11]', 'MC4[0].dend[12]', 'MC4[0].dend[13]', 'MC4[0].dend[14]', 'MC4[0].dend[15]', 'MC4[0].dend[16]', '...']`
  - `dest_syn`: `['GabaSyn']`
  - `dest_syn_params`: `["{'gmax': 0.005, 'tau1': 1, 'tau2': 100}"]`
  - `is_reciprocal`: `[True]`
  - `source_section`: `['GC1[100].apic[5]', 'GC1[12].apic[5]', 'GC1[12].apic[8]', 'GC1[16].apic[2]', 'GC1[16].apic[3]', 'GC1[16].apic[4]', 'GC1[16].apic[5]', 'GC1[16].apic[8]', '...']`
  - `source_syn`: `['AmpaNmdaSyn']`
  - `source_syn_params`: `["{'gmax': 0.1}"]`
- Numeric ranges:
  - `delay`: `0.49885996598005294..0.5049983288645744`
  - `dest_seg_i`: `0..13`
  - `dest_x`: `0..1`
  - `source_seg_i`: `0..6`
  - `source_x`: `0..1`
  - `threshold`: `0..0`
  - `weight`: `1..1`

#### GCs__TCs.json

- Entries: `1751`
- Entry keys: `['create_spine', 'delay', 'dest_section', 'dest_seg_i', 'dest_syn', 'dest_syn_params', 'dest_x', 'is_reciprocal', 'source_section', 'source_seg_i', 'source_syn', 'source_syn_params', 'source_x', 'threshold', 'weight']`
- Categorical domains (truncated where long):
  - `create_spine`: `[False]`
  - `dest_section`: `['TC3[0].dend[2]', 'TC3[0].dend[5]', 'TC3[0].dend[7]', 'TC3[2].dend[4]', 'TC3[2].dend[7]', 'TC3[4].dend[4]', 'TC3[4].dend[7]', 'TC4[0].dend[0]', '...']`
  - `dest_syn`: `['GabaSyn']`
  - `dest_syn_params`: `["{'gmax': 0.005, 'tau1': 1, 'tau2': 100}"]`
  - `is_reciprocal`: `[True]`
  - `source_section`: `['GC1[16].apic[2]', 'GC1[16].apic[4]', 'GC1[16].apic[5]', 'GC1[16].apic[8]', 'GC1[18].apic[10]', 'GC1[18].apic[2]', 'GC1[18].apic[4]', 'GC1[18].apic[5]', '...']`
  - `source_syn`: `['AmpaNmdaSyn']`
  - `source_syn_params`: `["{'gmax': 0.1}"]`
- Numeric ranges:
  - `delay`: `0.49964423647522926..0.5049983040392398`
  - `dest_seg_i`: `0..10`
  - `dest_x`: `0..1`
  - `source_seg_i`: `0..4`
  - `source_x`: `0..1`
  - `threshold`: `0..0`
  - `weight`: `1..1`

### 4.3 Glomerulus mapping file (`glom_cells.json`)

- Schema: `dict[str(glom_id) -> list[str(cell_ref)]]`
- Glomeruli present in current slice map: `[1474, 1614]`
- Cell ref pattern examples: `MC4[0]`, `MC5[14]`, `TC5[20]`

## 5. Odor Database-Constrained Domains

Backed by `olfactorybulb/model-data.sqlite` (`odor` + `odor_glom` tables):

- Valid odor names (`odor.name`): `['Apple', 'Banana', 'Basil', 'Black_Pepper', 'Cheese', 'Chocolate', 'Cinnamon', 'Cloves', 'Coffee', 'Garlic', 'Ginger', 'Kiwi', 'Lemongrass', 'Mint', 'Onion', 'Oregano', 'Pear', 'Pineapple']`
- Global odor intensity range (`odor_glom.intensity`): `0.203719..1.0`
- Distinct glomeruli with odor entries: `127` (ID range `33..1912`)

## 6. Cell Intrinsic Parameter Surfaces (Birgiolas2020)

Source: `prev_ob_models/Birgiolas2020/isolated_cells.py`

Slice-used models:

- `['GC1', 'GC2', 'GC3', 'GC4', 'GC5', 'MC4', 'MC5', 'TC3', 'TC4', 'TC5']`

### 6.1 MC parameter surface

| Idx | Attribute | Applied to section list(s) | Fit bounds (`low..high`) | Used range in current slice (`min..max`) |
|---|---|---|---|---|
| 0 | `diam` | `['apical', 'basal', 'axonal']` | `0.1..5.0` | `1.1889955393643883..1.4659052116669653` |
| 1 | `L` | `['somatic']` | `6.58..13.37` | `6.713316586000511..11.272989236805506` |
| 2 | `Ra` | `['all']` | `1.0..150.0` | `45.81580432923558..113.0647154118838` |
| 3 | `cm` | `['all']` | `0.1..2.0` | `0.6423077604986799..1.0195884959924766` |
| 4 | `ena` | `['all']` | `20.0..80.0` | `24.189605615199625..27.047779923908816` |
| 5 | `ek` | `['all']` | `-100.0..-50.0` | `-70.21445104015474..-61.22263984742484` |
| 6 | `e_pas` | `['all']` | `-90.0..-50.0` | `-77.76500118588712..-58.44571000276015` |
| 7 | `g_pas` | `['all']` | `0..0.0002` | `2.41179589261979e-05..2.799474934847125e-05` |
| 8 | `sh_Na` | `['all']` | `0..10` | `2.1234251230802768..7.473006638962397` |
| 9 | `tau_CaPool` | `['all']` | `1..300` | `21.769108516252658..116.66617044665243` |
| 10 | `gbar_Na` | `['all']` | `0..0.2` | `0.08030263281497557..0.08938182311564233` |
| 11 | `gbar_Kd` | `['all']` | `0..0.1` | `0.028305145075594434..0.045378073990493345` |
| 12 | `gbar_Kslow` | `['all']` | `0..0.002` | `1.940515013623855e-05..0.0013717137679573465` |
| 13 | `gbar_KA` | `['all']` | `0..0.02` | `0.003367433087058607..0.008235801667492574` |
| 14 | `gbar_KCa` | `['all']` | `0..0.016` | `0.006295591114313158..0.008749055435310703` |
| 15 | `gbar_LCa` | `['all']` | `0..0.0005` | `6.059225606189184e-05..0.00011617477848639343` |
| 16 | `eh` | `['apical']` | `-40.0..-10.0` | `-32.61943576903494..-16.431775954888618` |
| 17 | `gbar_Ih` | `['apical']` | `0..6e-06` | `5.730004860036104e-07..2.5545247291881558e-06` |
| 18 | `gbar_CaT` | `['apical']` | `0..0.02` | `0.013625818845992767..0.018106878080415785` |

### 6.2 TC parameter surface

| Idx | Attribute | Applied to section list(s) | Fit bounds (`low..high`) | Used range in current slice (`min..max`) |
|---|---|---|---|---|
| 0 | `diam` | `['apical', 'basal', 'axonal']` | `0.1..2.0` | `0.45739561744658364..1.1150775732429254` |
| 1 | `L` | `['somatic']` | `3.5..11.6` | `4.43210198915518..5.427464927311558` |
| 2 | `Ra` | `['all']` | `1.0..150.0` | `38.306676426103074..140.0603567489126` |
| 3 | `cm` | `['all']` | `0.1..5.0` | `0.338208858836475..2.5647876077309393` |
| 4 | `ena` | `['all']` | `20.0..80.0` | `25.78453453122405..48.07430525546685` |
| 5 | `ek` | `['all']` | `-100.0..-50.0` | `-72.19350074054312..-64.10144100660276` |
| 6 | `e_pas` | `['all']` | `-90.0..-50.0` | `-72.04410633972584..-59.438072156710426` |
| 7 | `g_pas` | `['all']` | `0..0.0004` | `4.465709904148039e-05..0.00035375861033552975` |
| 8 | `sh_Na` | `['all']` | `0..10` | `0.0153674585604624..0.9986730975896996` |
| 9 | `tau_CaPool` | `['all']` | `1..300` | `88.40262670945955..265.1951143786057` |
| 10 | `gbar_Na` | `['all']` | `0..0.1` | `0.03672195458892829..0.09858081240423741` |
| 11 | `gbar_Kd` | `['all']` | `0..0.2` | `0.1049422156274652..0.17075238612729338` |
| 12 | `gbar_Kslow` | `['all']` | `0..0.002` | `0.0005348071482369292..0.00134280190081124` |
| 13 | `gbar_KA` | `['all']` | `0..0.02` | `0.001918455378510126..0.014430506220670415` |
| 14 | `gbar_KCa` | `['all']` | `0..0.016` | `5.983382586310648e-06..0.001540229017858929` |
| 15 | `gbar_LCa` | `['all']` | `0..0.001` | `6.938460042954509e-07..0.0005467916756341761` |
| 16 | `eh` | `['apical']` | `-40.0..-10.0` | `-38.57314310427289..-12.23003917041239` |
| 17 | `gbar_Ih` | `['apical']` | `0..6e-05` | `1.464842906610013e-05..4.860988596316896e-05` |
| 18 | `gbar_CaT` | `['apical']` | `0..0.02` | `0.010787608496049355..0.019510389835192148` |

### 6.3 GC parameter surface

| Idx | Attribute | Applied to section list(s) | Fit bounds (`low..high`) | Used range in current slice (`min..max`) |
|---|---|---|---|---|
| 0 | `diam` | `['apical']` | `0.1..3.0` | `0.2541071033868456..0.8779752631390383` |
| 1 | `L` | `['somatic']` | `0.89..5.04` | `0.9018686615121411..2.7418437634170245` |
| 2 | `Ra` | `['all']` | `5.0..150.0` | `5.840902030011179..135.48423907699538` |
| 3 | `cm` | `['all']` | `0.1..10.0` | `3.9732889416523793..9.915742710613806` |
| 4 | `ena` | `['all']` | `10.0..90.0` | `25.2960798724282..44.457228147075426` |
| 5 | `ek` | `['all']` | `-100.0..-30.0` | `-76.70759599060172..-47.61720974003267` |
| 6 | `e_pas` | `['all']` | `-100.0..-50.0` | `-91.92491812845267..-79.25006914322489` |
| 7 | `g_pas` | `['all']` | `0..0.004` | `0.00011634241105991428..0.0002875798281665264` |
| 8 | `sh_Na` | `['all']` | `0..10` | `0.6794102899039003..8.994121763868439` |
| 9 | `gbar_Na` | `['apical']` | `0..0.4` | `0.14280039829386304..0.2329630644884414` |
| 10 | `gbar_Kd` | `['apical']` | `0..1.6` | `0.056796599167294994..1.1277751441096302` |
| 11 | `gbar_Na` | `['somatic']` | `0..5.0` | `0.3960861517225469..1.989811028917615` |
| 12 | `gbar_Kd` | `['somatic']` | `0..5.0` | `0.5050189668908865..2.8410280349325325` |
| 13 | `gbar_KA` | `['somatic']` | `0..0.8` | `0.3619221096015237..0.7995603568770102` |
| 14 | `eh` | `['somatic']` | `-60.0..-10.0` | `-58.427640754751636..-10.019130794184761` |
| 15 | `gbar_Ih` | `['somatic']` | `0..0.0002` | `1.5062876653742342e-08..3.236141764651116e-05` |
| 16 | `gbar_KM` | `['somatic']` | `0..0.13` | `0.00020697623727848618..0.08699171893837351` |

## 7. Appendix: Exact `param_values` Vectors for Used Models

Order of each vector follows the corresponding cell type `params` list order from `isolated_cells.py`.

- `GC1`: `[0.6354331329119002, 1.2938842675019608, 86.60228776188752, 6.097757804039867, 25.2960798724282, -76.70759599060172, -89.3885367109972, 0.00011634241105991428, 2.6678571405236333, 0.19655222427059713, 0.20216712905593315, 0.3960861517225469, 1.3914563069980788, 0.506414168406638, -10.019130794184761, 1.5062876653742342e-08, 0.00036501509920944155]`
- `GC2`: `[0.5919240225726485, 0.9377868051004455, 135.48423907699538, 9.868734888180338, 44.457228147075426, -69.849717759686, -87.44367826094535, 0.0002532025410260323, 0.6794102899039003, 0.14280039829386304, 0.6754823006187463, 1.5101328902421631, 0.5050189668908865, 0.5196791927599024, -10.798383152114752, 4.855896020236519e-06, 0.08699171893837351]`
- `GC3`: `[0.5207279006127108, 2.7418437634170245, 29.86526990214049, 9.915742710613806, 34.811323621837964, -47.61720974003267, -79.25006914322489, 0.00026346830363808586, 8.994121763868439, 0.2060895658072, 0.056796599167294994, 1.298689426862954, 2.8410280349325325, 0.7995603568770102, -58.427640754751636, 7.935912189256338e-07, 0.012519341898903455]`
- `GC4`: `[0.2541071033868456, 0.9018686615121411, 5.840902030011179, 8.982427157672912, 33.26154903312686, -72.49039418565542, -87.00745282633099, 0.0002875798281665264, 1.3482155620981673, 0.2329630644884414, 0.7991879969967408, 1.1557906139672314, 1.079595577699155, 0.7422718950734719, -15.525477880615483, 2.407280315875134e-05, 0.0029172228059790853]`
- `GC5`: `[0.8779752631390383, 1.858976049156634, 110.66782930144701, 3.9732889416523793, 27.681984895293596, -58.08391616048308, -91.92491812845267, 0.00015430675359767468, 4.8308294713153055, 0.1805187512240916, 1.1277751441096302, 1.989811028917615, 1.155885940569044, 0.3619221096015237, -45.459550085137906, 3.236141764651116e-05, 0.00020697623727848618]`
- `MC4`: `[1.1889955393643883, 6.713316586000511, 113.0647154118838, 1.0195884959924766, 27.047779923908816, -70.21445104015474, -58.44571000276015, 2.41179589261979e-05, 2.1234251230802768, 116.66617044665243, 0.08938182311564233, 0.028305145075594434, 0.0013717137679573465, 0.008235801667492574, 0.006295591114313158, 6.059225606189184e-05, -16.431775954888618, 2.5545247291881558e-06, 0.013625818845992767]`
- `MC5`: `[1.4659052116669653, 11.272989236805506, 45.81580432923558, 0.6423077604986799, 24.189605615199625, -61.22263984742484, -77.76500118588712, 2.799474934847125e-05, 7.473006638962397, 21.769108516252658, 0.08030263281497557, 0.045378073990493345, 1.940515013623855e-05, 0.003367433087058607, 0.008749055435310703, 0.00011617477848639343, -32.61943576903494, 5.730004860036104e-07, 0.018106878080415785]`
- `TC3`: `[1.1150775732429254, 4.4342321443458665, 140.0603567489126, 2.5647876077309393, 36.98887466006239, -64.10144100660276, -72.04410633972584, 0.00018889462798377362, 0.9986730975896996, 265.1951143786057, 0.09497953219449135, 0.17075238612729338, 0.0006995167792363354, 0.01258253721755341, 5.983382586310648e-06, 0.0005467916756341761, -12.23003917041239, 1.464842906610013e-05, 0.019510389835192148]`
- `TC4`: `[0.880892886671345, 5.427464927311558, 57.73964108732055, 1.2553640380097877, 48.07430525546685, -69.8309330807106, -59.438072156710426, 4.465709904148039e-05, 0.0153674585604624, 118.9260692659365, 0.03672195458892829, 0.1049422156274652, 0.0005348071482369292, 0.001918455378510126, 0.001540229017858929, 0.00034950147735730446, -38.57314310427289, 3.303294139699703e-05, 0.011898977012691088]`
- `TC5`: `[0.45739561744658364, 4.43210198915518, 38.306676426103074, 0.338208858836475, 25.78453453122405, -72.19350074054312, -62.7232566025413, 0.00035375861033552975, 0.024296848024877304, 88.40262670945955, 0.09858081240423741, 0.11389061162000566, 0.00134280190081124, 0.014430506220670415, 0.0005401346693437823, 6.938460042954509e-07, -31.55910335355583, 4.860988596316896e-05, 0.010787608496049355]`

## 8. Optional Single-JSON Run Spec Skeleton

If you want one explicit JSON file per run, this skeleton matches the current runtime surface:

```json
{
  "paramset_name": "GammaSignature",
  "runtime": {
    "rnd_seed": 0,
    "slice_dir": "olfactorybulb/slices",
    "slice_name": "DorsalColumnSlice",
    "sim_dt": 0.1,
    "recording_period": 0.1,
    "tstop": 1800,
    "record_from_somas": ["MC", "TC", "GC"],
    "lfp_electrode_location": [116, 1078, -61]
  },
  "connectivity": {
    "gap_juction_gmax": {"MC": 32, "TC": 32},
    "synapse_properties": {
      "AmpaNmdaSyn": {"gmax": 64, "ltpinvl": 0, "ltdinvl": 0},
      "GabaSyn": {"gmax": 2, "tau2": 36, "ltpinvl": 0, "ltdinvl": 0}
    }
  },
  "odor_input": {
    "inhale_duration": 125,
    "max_firing_rate": 150,
    "input_syn_tau1": 6,
    "input_syn_tau2": 12,
    "mc_input_delay": 0,
    "mc_input_weight": 0.2,
    "tc_input_delay": 0,
    "tc_input_weight": 0.8,
    "input_odors": {
      "0": {"name": "Apple", "rel_conc": 0.1},
      "200": {"name": "Apple", "rel_conc": 0.2}
    }
  }
}
```
