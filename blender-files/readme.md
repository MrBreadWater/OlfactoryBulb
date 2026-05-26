# Blender Files

These files are archival construction and visualization assets. They are not
required for normal OBGPU notebook runs, remote Slurm runs, or benchmark smoke
tests.

Most assets were developed with Blender 2.79. For best results, open historical
`.blend` files with that version.

## Important Assets

- `ob-gloms-fast.blend`: locations of possible glomeruli and cell somas used by
  the original network construction workflow. Model slices select subsets of
  these cells and glomeruli.
- `layers-simplified.blend`: simplified olfactory bulb layer meshes.
- `gloms-aligned-with-Mig14.blend`: registration of Migliore 2014 glomeruli to
  this model's coordinate system.

## Purged Archival Presentation Assets

- `bulb-layers-only.blend`: nested OB layer animation uploaded to Sketchfab.
- `cell-galery.blend`: side-by-side cell figure source.

The following large archival files were removed from the active git tree and
purged from history for repository size:

- `10MC-net.blend`
- `cell-gallery-animated.blend`
- `ob-all-cells.blend.tar.gz`
- `ob-gloms.blend.tar.gz`
- `media/10MC-net.mp4`
- generated website-header GIF renders under
  `media/website_header_animated_concepts/`

Keep replacement copies in external storage or regenerate them locally. Large
binary Blender/media assets should not be added casually; prefer keeping new
generated renders outside git unless they are intentional documentation
deliverables.
