"""Simple smoke tests for the config save/load helpers in obgpu_experiment_helpers.

Run with:
    python test_config_helpers.py
"""

import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath

import obgpu_experiment_helpers as hlp
from obgpu_experiment_helpers import (
    build_param_overrides,
    build_run_config,
    config_diff,
    list_paramsets,
    list_saved_configs,
    load_config,
    save_config,
)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # --- save_config / load_config round-trip ---
    cfg = build_run_config(paramset="GammaSignature", gaba_tau2_ms=36.0, gap_mc=32.0)
    p = save_config(cfg, tmp / "smoke.json")
    assert p.exists(), "save_config did not create file"
    loaded = load_config(p)
    assert loaded["paramset"] == "GammaSignature"
    assert loaded["gaba_tau2_ms"] == 36.0
    assert loaded["gap_mc"] == 32.0
    print("save_config / load_config: OK")

    # --- odor key normalization (JSON turns int keys to strings) ---
    cfg2 = build_run_config(
        input_odors={0: {"name": "Apple", "rel_conc": 0.1}, 200: {"name": "Apple", "rel_conc": 0.2}}
    )
    save_config(cfg2, tmp / "odors.json")
    loaded2 = load_config(tmp / "odors.json")
    assert all(isinstance(k, int) for k in loaded2["input_odors"])
    assert loaded2["input_odors"][0]["name"] == "Apple"
    print("odor key normalization: OK")

    # --- list_saved_configs ---
    results = list_saved_configs(tmp)
    assert len(results) == 2
    assert all(p.suffix == ".json" for p in results)
    print("list_saved_configs: OK")

    # --- list_paramsets (builtin only) ---
    names = list_paramsets()
    assert isinstance(names, list) and len(names) > 0
    assert "GammaSignature" in names
    assert "SilentNetwork" not in names  # base class should be excluded
    assert names == sorted(names)
    print(f"list_paramsets: OK  ({len(names)} paramsets found)")

    # --- list_paramsets (include_saved) ---
    save_config(cfg, tmp / "custom_experiment.json")
    sources = list_paramsets(include_saved=True, configs_dir=tmp)
    assert isinstance(sources, dict)
    assert "builtin" in sources and "saved" in sources
    assert "GammaSignature" in sources["builtin"]
    assert any(p.name == "custom_experiment.json" for p in sources["saved"])
    print(f"list_paramsets(include_saved=True): OK  ({len(sources['saved'])} saved config(s) found)")

    # --- config_diff ---
    cfg_a = build_run_config(paramset="GammaSignature", gaba_tau2_ms=36.0)
    cfg_b = build_run_config(paramset="GammaSignature", gaba_tau2_ms=50.0)
    changes = config_diff(cfg_a, cfg_b)
    assert len(changes) > 0
    tau_change = next(c for c in changes if "tau2" in c["path"])
    assert tau_change["before"] == 36.0
    assert tau_change["after"] == 50.0

    no_changes = config_diff(cfg_a, deepcopy(cfg_a))
    assert no_changes == []
    print("config_diff: OK")

    # --- KAR / ketamine controls ---
    cfg_kar = build_run_config(
        ketamine_block=0.05,
        ampa_block=1.0,
        kar_mt_gmax=0.002,
        kar_tau2_ms=90.0,
        kar_tau3_ms=480.0,
        kar_amp2=0.01,
        enable_gc_kar=True,
        kar_gc_gmax=0.001,
        gc_ka_gbar_scale=0.5,
    )
    overrides = build_param_overrides(cfg_kar)
    assert overrides["synapse_properties"]["AmpaNmdaSyn"]["ketamine_block"] == 0.05
    assert overrides["synapse_properties"]["AmpaNmdaSyn"]["ampa_block"] == 1.0
    assert overrides["kar_mt_gmax"] == 0.002
    assert overrides["kar_tau2"] == 90.0
    assert overrides["kar_tau3"] == 480.0
    assert overrides["kar_amp2"] == 0.01
    assert overrides["enable_gc_kar"] is True
    assert overrides["kar_gc_gmax"] == 0.001
    assert overrides["gc_ka_gbar_scale"] == 0.5
    print("KAR / ketamine controls: OK")

    # --- Paramiko SFTP should be lazy and cached ---
    original_connect = hlp._connect_paramiko
    original_paramiko = hlp.paramiko
    try:
        connection = {"transport": "dummy-transport", "sftp": None}
        sftp_calls = []

        class _DummySFTPClient:
            @staticmethod
            def from_transport(transport):
                sftp_calls.append(transport)
                return {"transport": transport}

        class _DummyParamiko:
            SFTPClient = _DummySFTPClient

        hlp._connect_paramiko = lambda _config: connection
        hlp.paramiko = _DummyParamiko

        sftp_1 = hlp._get_paramiko_sftp({})
        sftp_2 = hlp._get_paramiko_sftp({})
        assert sftp_1 is sftp_2
        assert sftp_calls == ["dummy-transport"]
        print("Paramiko SFTP lazy-open: OK")
    finally:
        hlp._connect_paramiko = original_connect
        hlp.paramiko = original_paramiko

    # --- Remote sweep submit should use an uploaded manifest file ---
    sweep_cfg = build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        sweep_engine="remote_batch",
    )
    sweep_plan = hlp._prepare_sweep_plan(sweep_cfg, "gaba_gmax", [0.0, 0.1])
    driver_command, manifest_items, manifest_json, manifest_path, _parallelism = hlp._build_remote_sweep_driver_command(
        sweep_cfg,
        sweep_plan=sweep_plan,
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/test_sweep"),
    )
    assert "--items-json" in driver_command
    assert "--items-b64" not in driver_command
    assert manifest_path.as_posix() in driver_command
    parsed_manifest = json.loads(manifest_json)
    assert isinstance(parsed_manifest, list) and len(parsed_manifest) == len(manifest_items)
    assert parsed_manifest[0]["label"] == manifest_items[0]["label"]
    print("Remote sweep manifest upload path: OK")

    # --- Remote helper cache should shrink submit/poll command payloads ---
    remote_cfg = build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        remote_conda_activate_cmd="source activate OBGPU",
        remote_git_ref="abcdef1234567890",
    )
    remote_submit_inline = hlp._build_remote_submit_command(
        remote_cfg,
        label="test_label",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        benchmark_command=["nrniv", "-mpi", "-python", "bench.py"],
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        remote_git_ref="abcdef1234567890",
        remote_helper_dir=None,
    )
    remote_submit_cached = hlp._build_remote_submit_command(
        remote_cfg,
        label="test_label",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        benchmark_command=["nrniv", "-mpi", "-python", "bench.py"],
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        remote_git_ref="abcdef1234567890",
        remote_helper_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test"),
    )
    assert len(remote_submit_cached) < len(remote_submit_inline)
    assert "/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test/submit_sol_run.py" in remote_submit_cached
    remote_poll_cached = hlp._build_remote_poll_command(
        remote_cfg,
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_result_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label"),
        job_id="12345",
        remote_helper_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test"),
        include_sacct=False,
        include_tails=False,
    )
    assert "--skip-sacct" in remote_poll_cached
    assert "--skip-tails" in remote_poll_cached
    print("Remote helper cache command shrink: OK")

    # --- Remote preflight should cache successful probes within one notebook runtime ---
    original_run_ssh_shell = hlp._run_ssh_shell
    try:
        hlp._LIVE_REMOTE_PREFLIGHTS.clear()
        preflight_calls = []

        def _fake_run_ssh_shell(_config, command, check=False):
            preflight_calls.append((command, check))
            return subprocess.CompletedProcess(
                args=["ssh", "bash", "-lc", command],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )

        hlp._run_ssh_shell = _fake_run_ssh_shell
        completed_1, cached_1 = hlp._run_remote_preflight_cached(
            remote_cfg,
            remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        )
        completed_2, cached_2 = hlp._run_remote_preflight_cached(
            remote_cfg,
            remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        )
        assert completed_1.returncode == 0 and completed_2.returncode == 0
        assert cached_1 is False and cached_2 is True
        assert len(preflight_calls) == 1
        print("Remote preflight cache: OK")
    finally:
        hlp._run_ssh_shell = original_run_ssh_shell
        hlp._LIVE_REMOTE_PREFLIGHTS.clear()

    # --- Stale allocation cleanup should skip manual allocations and reuse its session cache ---
    original_cleanup = hlp._cleanup_stale_remote_slurm_allocations
    try:
        hlp._LIVE_REMOTE_STALE_CLEANUPS.clear()
        cleanup_calls = []

        def _fake_cleanup(_config, *, remote_helper_dir=None):
            cleanup_calls.append(remote_helper_dir)
            return [{"job_id": "1", "action": "cancel_requested"}]

        hlp._cleanup_stale_remote_slurm_allocations = _fake_cleanup
        cleanup_cfg = build_run_config(
            runner_backend="sol_slurm",
            remote_host="user@host",
            remote_repo_root="/remote/OlfactoryBulb",
            remote_results_root="/remote/OlfactoryBulb/results/notebook_runs_cleanup_test",
            slurm_reuse_allocation=True,
            slurm_allocation_job_id="999",
        )
        assert hlp._maybe_cleanup_stale_remote_slurm_allocations(cleanup_cfg) == []
        assert cleanup_calls == []
        cleanup_cfg["slurm_allocation_job_id"] = None
        cleanup_cfg["remote_cleanup_stale_allocations"] = True
        cleanup_1 = hlp._maybe_cleanup_stale_remote_slurm_allocations(cleanup_cfg)
        cleanup_2 = hlp._maybe_cleanup_stale_remote_slurm_allocations(cleanup_cfg)
        assert cleanup_1 == cleanup_2 == [{"job_id": "1", "action": "cancel_requested"}]
        assert len(cleanup_calls) == 1
        print("Stale allocation cleanup throttle: OK")
    finally:
        hlp._cleanup_stale_remote_slurm_allocations = original_cleanup
        hlp._LIVE_REMOTE_STALE_CLEANUPS.clear()

print("\nAll tests passed.")
