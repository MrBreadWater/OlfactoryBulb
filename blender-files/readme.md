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

## Archival Presentation Assets

- `10MC-net.blend`: static 10-MC network visualization used for a defense video.
- `bulb-layers-only.blend`: nested OB layer animation uploaded to Sketchfab.
- `cell-galery.blend`: side-by-side cell figure source.
- `cell-galery-animated.blend`: AP propagation animation source.
- `ob-all-cells.blend.tar.gz`: non-simplified layer meshes with full cell
  locations.
- `ob-gloms.blend.tar.gz`: non-simplified layers with glomerular locations.

Large binary Blender/media assets should not be added casually. Prefer keeping
new generated renders outside git unless they are intentional documentation
deliverables.
