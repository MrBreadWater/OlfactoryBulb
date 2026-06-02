"""Smoke checks for the permanent provisional EPLI paramsets."""

from olfactorybulb.paramsets.case_studies import (
    GammaSignature_EPLI_Provisional_MTC,
    GammaSignature_EPLI_Provisional_TCOnly,
)


tc_only = GammaSignature_EPLI_Provisional_TCOnly()
assert tc_only.slice_name == "DorsalColumnSliceEPLIProvisional"
assert tc_only.enable_epl_interneurons is True
assert tc_only.max_epl_interneurons == 24
assert tc_only.epl_interneuron_model == "SyntheticEPL2026.PVCRH_FSI1"
assert tc_only.epl_interneuron_synapse_sets == ["EPLIs__TCs"]
assert tc_only.record_from_somas == ["MC", "TC", "GC", "EPLI"]

mtc = GammaSignature_EPLI_Provisional_MTC()
assert mtc.slice_name == "DorsalColumnSliceEPLIProvisional"
assert mtc.epl_interneuron_synapse_sets == ["EPLIs__TCs", "EPLIs__MCs"]

print("provisional EPLI paramset smoke test: OK")
