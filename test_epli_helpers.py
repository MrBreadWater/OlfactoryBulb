"""Smoke tests for shared optional-EPLI configuration helpers.

Run with:
    conda run -n OBGPU python test_epli_helpers.py
"""

from olfactorybulb.epli import (
    DEFAULT_EPLI_MODEL_KEY,
    DEFAULT_EPLI_SYNAPSE_SET_NAMES,
    EPLI_CELL_TYPE,
    EPLI_GROUP_NAME,
    default_slice_group_colors,
    default_slice_group_names,
    default_slice_synapse_blueprints,
    epli_population_enabled,
    extend_runtime_cell_types,
    extend_runtime_synapse_sets,
    resolve_epli_model_spec,
)


assert not epli_population_enabled(enable_epl_interneurons=False, max_epl_interneurons=10)
assert not epli_population_enabled(enable_epl_interneurons=True, max_epl_interneurons=0)
assert epli_population_enabled(enable_epl_interneurons=True, max_epl_interneurons=1)

spec = resolve_epli_model_spec()
assert spec.key == DEFAULT_EPLI_MODEL_KEY
assert spec.role == EPLI_CELL_TYPE

assert extend_runtime_cell_types(["MC", "GC", "TC"]) == ["MC", "GC", "TC"]
assert extend_runtime_cell_types(
    ["MC", "GC", "TC"],
    enable_epl_interneurons=True,
    max_epl_interneurons=8,
) == ["MC", "GC", "TC", "EPLI"]

assert extend_runtime_synapse_sets(["GCs__MCs", "GCs__TCs"]) == ["GCs__MCs", "GCs__TCs"]
assert extend_runtime_synapse_sets(
    ["GCs__MCs", "GCs__TCs"],
    enable_epl_interneurons=True,
    max_epl_interneurons=8,
) == ["GCs__MCs", "GCs__TCs", *DEFAULT_EPLI_SYNAPSE_SET_NAMES]

assert default_slice_group_names() == ["MCs", "TCs", "GCs"]
assert default_slice_group_names(include_epli=True) == ["MCs", "TCs", "GCs", EPLI_GROUP_NAME]

colors = default_slice_group_colors(include_epli=True)
assert EPLI_GROUP_NAME in colors

default_blueprints = default_slice_synapse_blueprints()
assert [row["group_from"] for row in default_blueprints] == ["GCs", "GCs"]

epli_blueprints = default_slice_synapse_blueprints(include_epli=True)
assert len(epli_blueprints) == 4
assert epli_blueprints[2]["group_from"] == EPLI_GROUP_NAME
assert epli_blueprints[2]["group_to"] == "MCs"
assert epli_blueprints[2]["section_pattern_source"] == "*dend*"
assert epli_blueprints[2]["section_pattern_dest"] == "*soma*"
assert epli_blueprints[3]["group_to"] == "TCs"

print("EPLI helper smoke test: OK")
