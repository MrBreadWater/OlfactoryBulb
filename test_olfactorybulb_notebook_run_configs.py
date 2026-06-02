"""Focused tests for olfactory-bulb notebook run-config helpers."""

from __future__ import annotations

import warnings

from olfactorybulb.notebook_run_configs import (
    build_run_config,
    build_slurm_remote_config,
    build_sol_remote_config,
    build_param_overrides,
    default_sol_runtime_profiles,
    extract_runtime_control_snapshot,
    make_label,
    normalize_input_odors,
    resolve_effective_params,
    resolve_execution_mode,
    resolve_paramset_defaults,
)


def main() -> None:
    cfg = build_run_config(paramset="GammaSignature", mode="fast", label_prefix="demo")
    assert cfg["paramset"] == "GammaSignature"
    assert cfg["label_prefix"] == "demo"
    assert cfg["runner_backend"] == "local"
    assert cfg["nranks"] == 1
    assert cfg["results_base"].endswith("results/notebook_runs")
    assert cfg["remote_mpi_exec"] == "srun --mpi=pmix_v4 --cpu-bind=none"

    label = make_label(cfg, timestamp="20260602_120000")
    assert label == "demo_GammaSignature_fast_20260602_120000"

    normalized_odors = normalize_input_odors({"0": {"name": "Apple"}, "200.0": {"name": "Mint"}})
    assert list(normalized_odors.keys()) == [0, 200]

    execution_mode = resolve_execution_mode({"runner_backend": "slurm_remote", "slurm_gpus": 1})
    assert execution_mode == {"use_corenrn": True, "use_gpu": True, "source": "remote_slurm"}
    explicit_mode = resolve_execution_mode({"use_corenrn": False, "use_gpu": True})
    assert explicit_mode == {"use_corenrn": True, "use_gpu": True, "source": "explicit"}

    runtime_snapshot = extract_runtime_control_snapshot(
        build_run_config(
            runner_backend="slurm_remote",
            remote_host="user@host",
            remote_repo_root="/remote/OlfactoryBulb",
            slurm_gpus=1,
        )
    )
    assert runtime_snapshot["remote_host"] == "user@host"
    assert runtime_snapshot["resolved_execution_mode"]["source"] == "remote_slurm"

    overrides = build_param_overrides(
        build_run_config(
            input_odors={"0": {"name": "Apple", "rel_conc": 0.1}},
            enable_epl_interneurons=True,
            epl_interneuron_cell_type="EPLI",
        )
    )
    assert overrides["input_odors"] == {0: {"name": "Apple", "rel_conc": 0.1}}
    assert "EPLI" in overrides["record_from_somas"]

    defaults = resolve_paramset_defaults("GammaSignature")
    assert defaults["name"] == "GammaSignature"

    effective = resolve_effective_params({"paramset": "GammaSignature"})
    assert effective["paramset"] == "GammaSignature"
    assert "full_param_snapshot" in effective
    assert "lfp_electrode_location" in effective

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        remote_cfg = build_slurm_remote_config(
            remote_host="user@host",
            remote_repo_root="/remote/OlfactoryBulb",
        )
    assert remote_cfg["runner_backend"] == "slurm_remote"
    assert remote_cfg["remote_results_root"] == "/remote/OlfactoryBulb/results/notebook_runs"
    assert captured and "Remote Slurm configs reset use_corenrn/use_gpu to auto" in str(captured[0].message)

    sol_cfg = build_sol_remote_config(
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
    )
    assert sol_cfg["runner_backend"] == "sol_slurm"
    assert sol_cfg["remote_conda_activate_cmd"] == "source tools/setup/activate_sol_obgpu.sh"

    profiles = default_sol_runtime_profiles()
    assert [profile["name"] for profile in profiles] == [
        "sol-grace-hopper",
        "sol-arm",
        "sol-x86_64",
    ]

    print("olfactorybulb notebook run configs: OK")


if __name__ == "__main__":
    main()
