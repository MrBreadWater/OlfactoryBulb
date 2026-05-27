# EPLI Placement Debug - 2026-05-26

This note records the placement and slice-export debugging work that followed
the offline connectivity optimizer.

## Objective

Determine why opt-in `EPLI` slice exports were producing:

- only `1` exported `EPLI` root, and
- zero plausible `EPLI -> MC/TC` dendritic contacts.

## Step 1: verify candidate availability

The first suspicion was that the scene simply did not provide enough soma
candidate points for `EPLI`.

Headless Blender inspection showed:

- particle clouds present in `ob-gloms-fast.blend`:
  - `0 GL Particles`
  - `1 OPL Particles`
  - `2 ML Particles`
  - `4 GRL Particles`
- there is **no dedicated EPL particle cloud**
- the current `EPLI` fallback source is therefore `1 OPL Particles`

Counting `1 OPL Particles` inside `DorsalColumnSlice` gave:

- `492` OPL particles inside the slice
- `464` inside the default `0.2-0.8` EPL depth band
- `440` inside the tested `0.25-0.75` depth band

Conclusion:

- `EPLI` underpopulation was **not** caused by lack of raw candidate points.

## Step 2: verify builder-side selection

A minimal headless export with:

- `max_mcs=0`
- `max_tcs=0`
- `max_gcs=0`
- `max_eplis=5`

reported:

- `Selecting 5/433 1 OPL Particles EPLI locations inside slice`
- `EPLIs: 5`
- `Adding EPLI 0..4`

But the saved output still showed:

- `Saving cell group 1 EPLIs`
- `EPLIs.json` contained only one root

Conclusion:

- the loss happened **after** location selection and `add_epli(...)`
- the failure lived in group/root bookkeeping or serialization

## Step 3: isolate the bookkeeping bug

The `import_instance(...)` path imports one cell at a time with:

- `group.include_roots_by_name([instance_name], exclude_others=True)`

That means the group can temporarily collapse to the most recently imported
root. This is only safe if the final `select_roots(...)` call correctly
re-adds every `EPLI` root from the global BlenderNEURON root index.

The final selection code was:

```python
self.node.groups[EPLI_GROUP_NAME].select_roots('All', 'PVCRH*')
```

But BlenderNEURON internally matches with:

```python
fnmatch(root.name.lower(), pattern)
```

So the actual comparison was effectively:

- root name: `pvcrh_fsi1[0].soma`
- pattern: `PVCRH*`

which never matches on a case-sensitive `fnmatch`.

That leaves the group containing only the last imported `EPLI`, which is
exactly what the exported JSON showed.

## Fix

Added a shared helper:

- `olfactorybulb.epli.epli_root_name_pattern()`

which resolves the configured `EPLI` model and returns the lower-case root
prefix pattern, currently:

- `pvcrh_fsi1*`

Then updated the slice builder to use that pattern instead of the hard-coded
uppercase string.

## Expected next validation

After the fix, a minimal headless export with:

- `max_mcs=0`
- `max_tcs=0`
- `max_gcs=0`
- `max_eplis=5`

was re-run and produced:

- `Selecting 5/433 1 OPL Particles EPLI locations inside slice`
- `EPLIs: 5`
- `Adding EPLI 0..4`
- `Saving cell group 5 EPLIs`

and `EPLIs.json` now contains all `5` roots.

That confirms the export-count bug was fixed.

## Next step after count recovery

Now that `EPLI` export counts are no longer collapsing to one root, the next
job is to inspect connectivity geometry:

1. generate small mixed `MC/TC/GC/EPLI` smoke slices
2. evaluate `EPLI -> MC/TC` candidate rules with the offline optimizer
3. replace the current placeholder `*dend* -> *soma*` default only if the
   recovered geometry supports something better

## Working interpretation

At this stage the main placement/export blocker is:

- **root-selection bookkeeping**, not
- candidate-particle scarcity

If the export count recovers after the pattern fix, the next bottleneck will
likely be the biological synapse geometry:

- current defaults still bias `EPLI -> principal cell` inhibition toward
  `*dend* -> *soma*`
- the offline optimizer already showed that soma-driven proximity can create
  misleading false positives
- so the next pass should optimize for plausible dendritic or perisomatic
  contact structure, not raw proximity alone

## Step 4: check whether smoke-slice candidate ordering is biased

The next suspicion was that the small mixed slices were choosing poor `EPLI`
locations simply because they truncated the filtered candidate list in raw
particle order.

Using the `DorsalColumnSlice` smoke-slice configuration with:

- `max_mcs=2`
- `max_tcs=4`
- `max_gcs=20`
- `max_eplis=5`
- `epli_depth_min_fraction=0.25`
- `epli_depth_max_fraction=0.75`

the filtered candidate pool contained `437` valid `EPLI` soma locations.

For the first five locations chosen by plain slice order, nearest selected
principal-cell soma distances were about:

- `54–116 µm`

But the same pool contained candidates with nearest principal distances of:

- `9.7–23.5 µm`

Conclusion:

- the smoke slices were indeed suffering from **candidate-order bias**
- choosing the first `N` filtered candidates is a poor strategy for debugging
  local EPL inhibitory geometry

## Step 5: add opt-in principal-proximity selection

An opt-in `epli_selection_strategy='principal_proximity'` path was added.

Behavior:

- keep the default `slice_order` behavior unchanged
- when enabled, rank filtered `EPLI` soma candidates by:
  1. nearest selected `MC/TC` soma distance
  2. mean selected `MC/TC` soma distance
  3. particle id as a stable tie-break

This does not claim biological optimality. It is a deterministic debugging and
smoke-test strategy for finding whether the geometry can produce local contacts
at all.

## Step 6: inspect resulting contact geometry

With `principal_proximity` enabled on the same small mixed slice:

- exported default synapse files were still all zero
- but the offline optimizer now found robust nonzero `EPLI -> TC` local
  dendritic overlaps

Best recovered `EPLI -> TC` candidate family on the smoke slice:

- `source_pattern = *dend*`
- `target_pattern = *dend*`
- `max_distance_um = 10`
- `use_radius = True`
- `max_syns_per_pt = 2`

Representative metrics:

- `entries = 27`
- `source_coverage = 1.0`
- `target_coverage = 0.75`
- `median_distance_um ≈ 6.1`

Interpretation:

- the current synthetic `EPLI` geometry can support **local dendritic overlap
  with TCs**
- the current default exported rule (`*dend* -> *soma*`, `20 µm`) is too
  restrictive / mismatched for these smoke slices

## Step 7: inspect MC failure mode

Even after:

- fixing root selection,
- enabling principal-proximity candidate ranking, and
- widening `EPLI` dendrite confinement to the full EPL corridor,

the same small smoke slices still produced:

- **no nonzero `EPLI -> MC` contacts** under a broad search across
  `*dend*`, `*apic*`, `*soma*`, and `*axon*` targets, with distances up to
  `120 µm`

That is useful negative evidence.

Current interpretation:

- `MC` emptiness is **not** caused by the original root-selection bug
- `MC` emptiness is **not** fixed simply by better `EPLI` soma candidate
  ranking
- `MC` emptiness is **not** fixed simply by letting `EPLI` dendrites span the
  whole EPL corridor

So the next likely bottleneck is deeper:

- either the synthetic `EPLI` morphology is too compact or too shallow for
  `MC` reach,
- or the placement objective needs to target `MC` dendritic / proximal-apical
  geometry rather than principal-cell soma proximity,
- or both

## Current state after this round

The evidence now supports these narrower conclusions:

1. The `EPLI` exporter bug is fixed.
2. The offline optimizer is working and recovering meaningful smoke-slice
   `EPLI -> TC` geometry.
3. The default placeholder `EPLI` synapse rule is currently unsupported by the
   small-slice geometry.
4. The present surrogate still does **not** produce `MC` overlap in these small
   tests, so any future default should remain provisional until `MC` reach is
   improved or explained.
