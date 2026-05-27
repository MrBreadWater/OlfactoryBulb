"""Smoke tests for slice-builder CLI/environment configuration helpers.

Run with:
    python test_slice_builder_config.py
"""

import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from olfactorybulb.slicebuilder.config import slice_builder_env_kwargs, slice_builder_env_overrides_from_cli


env_kwargs = slice_builder_env_kwargs(
    {
        "OB_SLICE_NAME": "DorsalColumnSliceEPLI",
        "OB_SLICE_OUTPUT_NAME": "DorsalColumnSliceEPLI_smoke",
        "OB_SLICE_ODORS": "Apple,Mint",
        "OB_SLICE_MAX_MCS": "12",
        "OB_SLICE_MAX_TCS": "24",
        "OB_SLICE_MAX_GCS": "120",
        "OB_SLICE_ENABLE_EPLI": "1",
        "OB_SLICE_MAX_EPLIS": "18",
        "OB_SLICE_EPLI_PARTICLES": "1 OPL Particles",
        "OB_SLICE_EPLI_MODEL": "SyntheticEPL2026.PVCRH_FSI1",
        "OB_SLICE_EPLI_DEPTH_MIN": "0.3",
        "OB_SLICE_EPLI_DEPTH_MAX": "0.7",
        "OB_SLICE_EPLI_DEND_DEPTH_MIN": "0.0",
        "OB_SLICE_EPLI_DEND_DEPTH_MAX": "1.0",
        "OB_SLICE_EPLI_SELECTION": "principal_proximity",
    }
)

assert env_kwargs["slice_object_name"] == "DorsalColumnSliceEPLI"
assert env_kwargs["slice_output_name"] == "DorsalColumnSliceEPLI_smoke"
assert env_kwargs["odors"] == ["Apple", "Mint"]
assert env_kwargs["max_mcs"] == 12
assert env_kwargs["max_tcs"] == 24
assert env_kwargs["max_gcs"] == 120
assert env_kwargs["enable_epl_interneurons"] is True
assert env_kwargs["max_eplis"] == 18
assert env_kwargs["epli_particles_object_name"] == "1 OPL Particles"
assert env_kwargs["epl_interneuron_model"] == "SyntheticEPL2026.PVCRH_FSI1"
assert abs(env_kwargs["epli_depth_min_fraction"] - 0.3) < 1e-9
assert abs(env_kwargs["epli_depth_max_fraction"] - 0.7) < 1e-9
assert abs(env_kwargs["epli_dend_depth_min_fraction"] - 0.0) < 1e-9
assert abs(env_kwargs["epli_dend_depth_max_fraction"] - 1.0) < 1e-9
assert env_kwargs["epli_selection_strategy"] == "principal_proximity"

args = SimpleNamespace(
    slice_name="DorsalColumnSliceEPLI",
    slice_output_name="DorsalColumnSliceEPLI_smoke",
    odors=["Apple", "Mint"],
    max_mcs=12,
    max_tcs=None,
    max_gcs=120,
    max_eplis=18,
    mc_particles_object_name=None,
    tc_particles_object_name=None,
    gc_particles_object_name=None,
    epli_particles_object_name="1 OPL Particles",
    glom_particles_object_name=None,
    glom_layer_object_name=None,
    outer_opl_object_name=None,
    inner_opl_object_name=None,
    enable_epl_interneurons=True,
    epl_interneuron_model="SyntheticEPL2026.PVCRH_FSI1",
    epl_interneuron_family=None,
    epli_depth_min_fraction=0.3,
    epli_depth_max_fraction=0.7,
    epli_dend_depth_min_fraction=0.0,
    epli_dend_depth_max_fraction=1.0,
    epli_selection_strategy="principal_proximity",
)
overrides = slice_builder_env_overrides_from_cli(args)
assert overrides["OB_SLICE_NAME"] == "DorsalColumnSliceEPLI"
assert overrides["OB_SLICE_OUTPUT_NAME"] == "DorsalColumnSliceEPLI_smoke"
assert overrides["OB_SLICE_ODORS"] == "Apple,Mint"
assert overrides["OB_SLICE_ENABLE_EPLI"] == "1"
assert overrides["OB_SLICE_MAX_EPLIS"] == "18"
assert overrides["OB_SLICE_EPLI_DEND_DEPTH_MIN"] == "0.0"
assert overrides["OB_SLICE_EPLI_DEND_DEPTH_MAX"] == "1.0"
assert overrides["OB_SLICE_EPLI_SELECTION"] == "principal_proximity"

repo_root = Path(__file__).resolve().parent
cli_completed = subprocess.run(
    [
        sys.executable,
        "build-slice.py",
        "--dry-run",
        "--background",
        "--slice-name",
        "DorsalColumnSliceEPLI",
        "--slice-output-name",
        "DorsalColumnSliceEPLI_smoke",
        "--enable-epl-interneurons",
        "--max-eplis",
        "18",
        "--epli-dend-depth-min-fraction",
        "0.0",
        "--epli-dend-depth-max-fraction",
        "1.0",
        "--epli-selection-strategy",
        "principal_proximity",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
)
assert "COMMAND: blender -b blender-files/ob-gloms-fast.blend --python olfactorybulb/slicebuilder/blender.py" in cli_completed.stdout
assert "OB_SLICE_EPLI_DEND_DEPTH_MIN=0.0" in cli_completed.stdout
assert "OB_SLICE_EPLI_DEND_DEPTH_MAX=1.0" in cli_completed.stdout
assert "OB_SLICE_EPLI_SELECTION=principal_proximity" in cli_completed.stdout

print("slice-builder config smoke test: OK")
