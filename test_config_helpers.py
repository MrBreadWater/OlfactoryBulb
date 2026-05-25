"""Simple smoke tests for the config save/load helpers in obgpu_experiment_helpers.

Run with:
    python test_config_helpers.py
"""

import json
import importlib.util
import pickle
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
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

    # --- Paramiko should fail closed instead of reauthing mid-run ---
    original_paramiko = hlp.paramiko
    try:
        fake_remote_cfg = {
            "remote_host": "user@host",
            "ssh_options": [],
            "remote_preserve_paramiko_session": True,
        }
        cache_key = hlp._paramiko_connection_key(fake_remote_cfg)
        original_cached = hlp._LIVE_PARAMIKO_CONNECTIONS.get(cache_key)
        original_authenticated = cache_key in hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS

        class _DeadTransport:
            def is_active(self):
                return False

            def is_authenticated(self):
                return False

        hlp.paramiko = object()
        hlp._LIVE_PARAMIKO_CONNECTIONS[cache_key] = {"transport": _DeadTransport(), "sftp": None}
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.add(cache_key)

        try:
            hlp._connect_paramiko(fake_remote_cfg)
            raise AssertionError("Expected mid-run Paramiko reauth to be refused")
        except RuntimeError as exc:
            assert "remote_preserve_paramiko_session=True" in str(exc)
            assert cache_key in str(exc)
        print("Paramiko mid-run reauth refusal: OK")
    finally:
        hlp.paramiko = original_paramiko
        if original_cached is None:
            hlp._LIVE_PARAMIKO_CONNECTIONS.pop(cache_key, None)
        else:
            hlp._LIVE_PARAMIKO_CONNECTIONS[cache_key] = original_cached
        if original_authenticated:
            hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.add(cache_key)
        else:
            hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.discard(cache_key)

    # --- Git sync base candidates should be unique ancestor SHAs ---
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=hlp.REPO_ROOT,
        text=True,
    ).strip()
    git_candidates = hlp._local_git_sync_base_candidates(head_sha, max_count=32)
    assert len(git_candidates) == len(set(git_candidates))
    assert head_sha not in git_candidates
    if git_candidates:
        probe_sha = git_candidates[0]
        assert hlp._git_ref_is_ancestor(probe_sha, head_sha)
    print("git sync base candidates: OK")

    # --- Notebook-published branch refs should have one stable remote tracking ref ---
    tracking_ref = hlp._remote_notebook_tracking_ref_for_source("refs/heads/Speedups")
    assert tracking_ref == "refs/obgpu-notebook-sync/heads/Speedups"
    assert hlp._remote_notebook_tracking_ref_for_source("refs/obgpu-notebook-sync/tmp") is None
    fetch_command = hlp._build_remote_git_bundle_fetch_command(
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_bundle_path="/tmp/example.bundle",
        source_ref="refs/heads/Speedups",
        remote_git_ref="abcdef1234567890",
    )
    assert "refs/obgpu-notebook-sync/abcdef1234567890" in fetch_command
    assert "refs/obgpu-notebook-sync/heads/Speedups" in fetch_command
    print("Remote git bundle tracking refs: OK")

    # --- Remote bundle base lookup should prefer the stable published branch tip when valid ---
    original_run_ssh_shell = hlp._run_ssh_shell
    original_git_ref_is_ancestor = hlp._git_ref_is_ancestor
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=hlp.REPO_ROOT,
            text=True,
        ).strip()
        parent_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"],
            cwd=hlp.REPO_ROOT,
            text=True,
        ).strip()

        def _fake_run_ssh_shell(_config, command, check=False):
            assert "refs/obgpu-notebook-sync/heads/Speedups" in command
            return subprocess.CompletedProcess(
                args=["ssh", "bash", "-lc", command],
                returncode=0,
                stdout=parent_sha + "\n",
                stderr="",
            )

        hlp._run_ssh_shell = _fake_run_ssh_shell
        hlp._git_ref_is_ancestor = original_git_ref_is_ancestor
        resolved_base = hlp._resolve_remote_tracking_bundle_base(
            {"remote_host": "user@host", "ssh_options": []},
            remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
            commit_sha=head_sha,
            source_ref="refs/heads/Speedups",
        )
        assert resolved_base == parent_sha
        print("Remote tracked bundle base lookup: OK")
    finally:
        hlp._run_ssh_shell = original_run_ssh_shell
        hlp._git_ref_is_ancestor = original_git_ref_is_ancestor

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

    # --- Remote sweep should resolve the real timestamped payload dir after completion ---
    sweep_driver_spec = importlib.util.spec_from_file_location(
        "remote_sweep_driver_test",
        hlp.REPO_ROOT / "tools" / "remote" / "remote_sweep_driver.py",
    )
    assert sweep_driver_spec is not None and sweep_driver_spec.loader is not None
    remote_sweep_driver = importlib.util.module_from_spec(sweep_driver_spec)
    sweep_driver_spec.loader.exec_module(remote_sweep_driver)

    requested_dir = tmp / "remote-sweep-requested"
    requested_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = tmp / "remote-sweep-requested_20260525_120000"
    payload_dir.mkdir(parents=True, exist_ok=True)
    (payload_dir / "summary.json").write_text(
        json.dumps(
            {
                "label": payload_dir.name,
                "requested_label": requested_dir.name,
                "timestamp": "20260525_120000",
            }
        )
    )
    resolved_dir = remote_sweep_driver.resolve_completed_result_dir(requested_dir, requested_dir.name)
    assert resolved_dir == payload_dir
    print("Remote sweep actual payload dir resolution: OK")

    # --- Incremental sweep final sync should only run when most item payloads already exist locally ---
    assert hlp._remote_sweep_metadata_files() == (
        "summary.json",
        "sim_progress.json",
        "sweep_manifest.json",
        "mpi_preflight.log",
    )
    sweep_local_runs = tmp / "sweep-local-runs"
    manifest_stub = [
        {"label": "item_000"},
        {"label": "item_001"},
        {"label": "item_002"},
        {"label": "item_003"},
    ]
    for label in ("item_000", "item_001"):
        item_dir = sweep_local_runs / label
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "summary.json").write_text("{}")
        (item_dir / "lfp.pkl").write_bytes(b"payload")
    assert hlp._should_use_incremental_sweep_final_sync(
        manifest_stub,
        local_runs_dir=sweep_local_runs,
    ) is True
    assert hlp._should_use_incremental_sweep_final_sync(
        manifest_stub,
        local_runs_dir=tmp / "no-sweep-payloads",
    ) is False
    print("Incremental sweep final sync selection: OK")

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

    # --- Successful fast remote sync should only request essential result artifacts ---
    fast_files_default = hlp._remote_fast_sync_files()
    assert fast_files_default == ("summary.json", "input_times.pkl", "lfp.pkl", "gc_output_events.pkl")
    fast_files_minimal = hlp._remote_fast_sync_files(
        {"enable_lfp": False, "record_gc_output_events": False}
    )
    assert fast_files_minimal == ("summary.json", "input_times.pkl")
    print("Remote fast sync file set: OK")

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

    # --- Streamed Paramiko sync should fall back when expected artifacts are still missing ---
    original_remote_transport = hlp._remote_transport
    original_run_paramiko_shell = hlp._run_paramiko_shell
    original_stream_to_dir = hlp._stream_paramiko_archive_to_local_dir
    original_sftp_copy_tree = hlp._sftp_copy_tree
    original_get_paramiko_sftp = hlp._get_paramiko_sftp
    original_close_paramiko_sftp = hlp._close_paramiko_sftp
    try:
        fallback_calls = []
        fake_remote_cfg = {"remote_host": "user@host", "ssh_options": [], "ssh_transport": "paramiko"}

        hlp._remote_transport = lambda _config: "paramiko"
        hlp._run_paramiko_shell = lambda _config, _command: subprocess.CompletedProcess(
            args=["ssh", "bash", "-lc", _command],
            returncode=0,
            stdout="gzip\n0\n.tar.gz\n",
            stderr="",
        )
        hlp._stream_paramiko_archive_to_local_dir = lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["paramiko-stream"],
            returncode=0,
            stdout="",
            stderr="",
        )
        hlp._get_paramiko_sftp = lambda _config: object()
        hlp._close_paramiko_sftp = lambda _config: None

        def _fake_sftp_copy_tree(_sftp, remote_dir, local_dir):
            fallback_calls.append(remote_dir)
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        hlp._sftp_copy_tree = _fake_sftp_copy_tree
        sync_dir = tmp / "paramiko-sync-fallback"
        completed = hlp._sync_remote_result_dir(
            fake_remote_cfg,
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("summary.json",),
        )
        assert completed.returncode == 0
        assert fallback_calls == ["/remote/result"]
        assert (sync_dir / "summary.json").exists()
        assert "fallback completed successfully" in (completed.stderr or "")
        print("Paramiko stream sync artifact validation: OK")
    finally:
        hlp._remote_transport = original_remote_transport
        hlp._run_paramiko_shell = original_run_paramiko_shell
        hlp._stream_paramiko_archive_to_local_dir = original_stream_to_dir
        hlp._sftp_copy_tree = original_sftp_copy_tree
        hlp._get_paramiko_sftp = original_get_paramiko_sftp
        hlp._close_paramiko_sftp = original_close_paramiko_sftp

    # --- Selected-file Paramiko sync should retry on the same session without dropping auth ---
    original_remote_transport = hlp._remote_transport
    original_sftp_copy_files = hlp._sftp_copy_files
    original_get_paramiko_sftp = hlp._get_paramiko_sftp
    original_close_paramiko_sftp = hlp._close_paramiko_sftp
    original_drop_paramiko_connection = hlp._drop_paramiko_connection
    try:
        sync_attempts = []
        close_calls = []
        drop_calls = []
        fake_remote_cfg = {
            "remote_host": "user@host",
            "ssh_options": [],
            "ssh_transport": "paramiko",
            "remote_sync_compress": False,
            "remote_preserve_paramiko_session": True,
        }
        cache_key = hlp._paramiko_connection_key(fake_remote_cfg)
        original_cached = hlp._LIVE_PARAMIKO_CONNECTIONS.get(cache_key)
        original_authenticated = cache_key in hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS

        class _LiveTransport:
            def is_active(self):
                return True

            def is_authenticated(self):
                return True

        hlp._LIVE_PARAMIKO_CONNECTIONS[cache_key] = {"transport": _LiveTransport(), "sftp": object()}
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.add(cache_key)
        hlp._remote_transport = lambda _config: "paramiko"
        hlp._get_paramiko_sftp = lambda _config: object()
        hlp._close_paramiko_sftp = lambda _config: close_calls.append("close")
        hlp._drop_paramiko_connection = lambda _config: drop_calls.append("drop")

        def _fake_sftp_copy_files(_sftp, _remote_dir, local_dir, file_names):
            sync_attempts.append(tuple(file_names))
            if len(sync_attempts) == 1:
                raise OSError("transient sftp failure")
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        hlp._sftp_copy_files = _fake_sftp_copy_files
        sync_dir = tmp / "paramiko-selected-retry"
        completed = hlp._sync_remote_result_dir(
            fake_remote_cfg,
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
        )
        assert completed.returncode == 0
        assert sync_attempts == [("summary.json",), ("summary.json",)]
        assert drop_calls == []
        assert (sync_dir / "summary.json").exists()
        print("Paramiko selected-file retry preserves session: OK")
    finally:
        hlp._remote_transport = original_remote_transport
        hlp._sftp_copy_files = original_sftp_copy_files
        hlp._get_paramiko_sftp = original_get_paramiko_sftp
        hlp._close_paramiko_sftp = original_close_paramiko_sftp
        hlp._drop_paramiko_connection = original_drop_paramiko_connection
        if original_cached is None:
            hlp._LIVE_PARAMIKO_CONNECTIONS.pop(cache_key, None)
        else:
            hlp._LIVE_PARAMIKO_CONNECTIONS[cache_key] = original_cached
        if original_authenticated:
            hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.add(cache_key)
        else:
            hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.discard(cache_key)

    # --- Selected-file Paramiko sync should use plain SFTP when compression is disabled ---
    original_remote_transport = hlp._remote_transport
    original_stream_to_dir = hlp._stream_paramiko_archive_to_local_dir
    original_sftp_copy_files = hlp._sftp_copy_files
    original_get_paramiko_sftp = hlp._get_paramiko_sftp
    original_close_paramiko_sftp = hlp._close_paramiko_sftp
    try:
        selected_calls = []
        fake_remote_cfg = {
            "remote_host": "user@host",
            "ssh_options": [],
            "ssh_transport": "paramiko",
            "remote_sync_compress": False,
        }

        hlp._remote_transport = lambda _config: "paramiko"
        hlp._stream_paramiko_archive_to_local_dir = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected-file sync should not call the archive stream path")
        )
        hlp._get_paramiko_sftp = lambda _config: object()
        hlp._close_paramiko_sftp = lambda _config: None

        def _fake_sftp_copy_files(_sftp, _remote_dir, local_dir, file_names):
            selected_calls.append(tuple(file_names))
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        hlp._sftp_copy_files = _fake_sftp_copy_files
        sync_dir = tmp / "paramiko-selected-sync"
        completed = hlp._sync_remote_result_dir(
            fake_remote_cfg,
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
        )
        assert completed.returncode == 0
        assert selected_calls == [("summary.json",)]
        assert (sync_dir / "summary.json").exists()
        print("Paramiko selected-file SFTP path: OK")
    finally:
        hlp._remote_transport = original_remote_transport
        hlp._stream_paramiko_archive_to_local_dir = original_stream_to_dir
        hlp._sftp_copy_files = original_sftp_copy_files
        hlp._get_paramiko_sftp = original_get_paramiko_sftp
        hlp._close_paramiko_sftp = original_close_paramiko_sftp

    # --- Selected-file Paramiko sync should use the compressed stream path when enabled ---
    original_remote_transport = hlp._remote_transport
    original_run_paramiko_shell = hlp._run_paramiko_shell
    original_stream_to_dir = hlp._stream_paramiko_archive_to_local_dir
    original_sftp_copy_files = hlp._sftp_copy_files
    original_get_paramiko_sftp = hlp._get_paramiko_sftp
    original_close_paramiko_sftp = hlp._close_paramiko_sftp
    try:
        stream_calls = []
        fake_remote_cfg = {
            "remote_host": "user@host",
            "ssh_options": [],
            "ssh_transport": "paramiko",
            "remote_sync_compress": True,
        }

        hlp._remote_transport = lambda _config: "paramiko"
        hlp._run_paramiko_shell = lambda _config, _command: subprocess.CompletedProcess(
            args=["ssh", "bash", "-lc", _command],
            returncode=0,
            stdout="gzip\n67\n.tar.gz\nsummary.json\n",
            stderr="",
        )

        def _fake_stream_to_dir(_config, *, remote_result_dir, local_result_dir, compressor, raw_bytes, stream_command=None):
            stream_calls.append((remote_result_dir, compressor, raw_bytes, stream_command))
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            (local_result_dir / "summary.json").write_text("{}")
            return subprocess.CompletedProcess(args=["paramiko-stream-extract"], returncode=0, stdout="", stderr="")

        hlp._stream_paramiko_archive_to_local_dir = _fake_stream_to_dir
        hlp._sftp_copy_files = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compressed selected-file sync should not use plain SFTP on the happy path")
        )
        hlp._get_paramiko_sftp = lambda _config: object()
        hlp._close_paramiko_sftp = lambda _config: None

        sync_dir = tmp / "paramiko-selected-summary-stream"
        completed = hlp._sync_remote_result_dir(
            fake_remote_cfg,
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
        )
        assert completed.returncode == 0
        assert len(stream_calls) == 1
        assert stream_calls[0][1] == "gzip"
        assert stream_calls[0][2] == 67
        assert "summary.json" in (stream_calls[0][3] or "")
        assert (sync_dir / "summary.json").exists()
        print("Paramiko selected-file compressed stream path: OK")
    finally:
        hlp._remote_transport = original_remote_transport
        hlp._run_paramiko_shell = original_run_paramiko_shell
        hlp._stream_paramiko_archive_to_local_dir = original_stream_to_dir
        hlp._sftp_copy_files = original_sftp_copy_files
        hlp._get_paramiko_sftp = original_get_paramiko_sftp
        hlp._close_paramiko_sftp = original_close_paramiko_sftp

    # --- Selected-file Paramiko sync should ignore missing optional remote files instead of failing the stream ---
    original_remote_transport = hlp._remote_transport
    original_run_paramiko_shell = hlp._run_paramiko_shell
    original_stream_to_dir = hlp._stream_paramiko_archive_to_local_dir
    original_sftp_copy_files = hlp._sftp_copy_files
    original_get_paramiko_sftp = hlp._get_paramiko_sftp
    original_close_paramiko_sftp = hlp._close_paramiko_sftp
    try:
        stream_calls = []
        fake_remote_cfg = {
            "remote_host": "user@host",
            "ssh_options": [],
            "ssh_transport": "paramiko",
            "remote_sync_compress": True,
        }

        hlp._remote_transport = lambda _config: "paramiko"

        def _fake_run_paramiko_shell(_config, _command):
            return subprocess.CompletedProcess(
                args=["ssh", "bash", "-lc", _command],
                returncode=0,
                stdout="gzip\n67\n.tar.gz\nsummary.json\n",
                stderr="",
            )

        def _fake_stream_to_dir(_config, *, remote_result_dir, local_result_dir, compressor, raw_bytes, stream_command=None):
            stream_calls.append((remote_result_dir, compressor, raw_bytes, stream_command))
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            (local_result_dir / "summary.json").write_text("{}")
            return subprocess.CompletedProcess(args=["paramiko-stream-extract"], returncode=0, stdout="", stderr="")

        hlp._run_paramiko_shell = _fake_run_paramiko_shell
        hlp._stream_paramiko_archive_to_local_dir = _fake_stream_to_dir
        hlp._sftp_copy_files = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing optional selected files should not force SFTP fallback")
        )
        hlp._get_paramiko_sftp = lambda _config: object()
        hlp._close_paramiko_sftp = lambda _config: None

        sync_dir = tmp / "paramiko-selected-optional-filter"
        completed = hlp._sync_remote_result_dir(
            fake_remote_cfg,
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("summary.json",),
            include_files=("summary.json", "stderr.txt"),
        )
        assert completed.returncode == 0
        assert len(stream_calls) == 1
        assert "summary.json" in (stream_calls[0][3] or "")
        assert "stderr.txt" not in (stream_calls[0][3] or "")
        assert (sync_dir / "summary.json").exists()
        print("Paramiko selected-file probe filters missing optionals: OK")
    finally:
        hlp._remote_transport = original_remote_transport
        hlp._run_paramiko_shell = original_run_paramiko_shell
        hlp._stream_paramiko_archive_to_local_dir = original_stream_to_dir
        hlp._sftp_copy_files = original_sftp_copy_files
        hlp._get_paramiko_sftp = original_get_paramiko_sftp
        hlp._close_paramiko_sftp = original_close_paramiko_sftp

    # --- Deferred soma selected-file sync should use the compressed stream path ---
    original_remote_transport = hlp._remote_transport
    original_run_paramiko_shell = hlp._run_paramiko_shell
    original_stream_to_dir = hlp._stream_paramiko_archive_to_local_dir
    original_sftp_copy_files = hlp._sftp_copy_files
    original_get_paramiko_sftp = hlp._get_paramiko_sftp
    original_close_paramiko_sftp = hlp._close_paramiko_sftp
    try:
        stream_calls = []
        fake_remote_cfg = {
            "remote_host": "user@host",
            "ssh_options": [],
            "ssh_transport": "paramiko",
            "remote_sync_compress": True,
        }

        hlp._remote_transport = lambda _config: "paramiko"
        hlp._run_paramiko_shell = lambda _config, _command: subprocess.CompletedProcess(
            args=["ssh", "bash", "-lc", _command],
            returncode=0,
            stdout="gzip\n67\n.tar.gz\nsoma_vs.pkl\n",
            stderr="",
        )

        def _fake_stream_to_dir(_config, *, remote_result_dir, local_result_dir, compressor, raw_bytes, stream_command=None):
            stream_calls.append((remote_result_dir, compressor, raw_bytes, stream_command))
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            with open(local_result_dir / "soma_vs.pkl", "wb") as handle:
                handle.write(b"abc")
            return subprocess.CompletedProcess(args=["paramiko-stream-extract"], returncode=0, stdout="", stderr="")

        hlp._stream_paramiko_archive_to_local_dir = _fake_stream_to_dir
        hlp._sftp_copy_files = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deferred soma selected sync should not use plain SFTP on the happy path")
        )
        hlp._get_paramiko_sftp = lambda _config: object()
        hlp._close_paramiko_sftp = lambda _config: None

        sync_dir = tmp / "paramiko-selected-soma-stream"
        completed = hlp._sync_remote_result_dir(
            fake_remote_cfg,
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=sync_dir,
            expected_files=("soma_vs.pkl",),
            include_files=("soma_vs.pkl",),
        )
        assert completed.returncode == 0
        assert len(stream_calls) == 1
        assert stream_calls[0][1] == "gzip"
        assert stream_calls[0][2] == 67
        assert "soma_vs.pkl" in (stream_calls[0][3] or "")
        assert (sync_dir / "soma_vs.pkl").exists()
        print("Deferred soma selected sync uses compressed stream: OK")
    finally:
        hlp._remote_transport = original_remote_transport
        hlp._run_paramiko_shell = original_run_paramiko_shell
        hlp._stream_paramiko_archive_to_local_dir = original_stream_to_dir
        hlp._sftp_copy_files = original_sftp_copy_files
        hlp._get_paramiko_sftp = original_get_paramiko_sftp
        hlp._close_paramiko_sftp = original_close_paramiko_sftp

    # --- Deferred remote soma traces should sync on first access ---
    original_sync_remote_result_dir = hlp._sync_remote_result_dir
    try:
        deferred_result_dir = tmp / "deferred-remote-result"
        deferred_result_dir.mkdir(parents=True, exist_ok=True)
        (deferred_result_dir / "summary.json").write_text("{}")
        (deferred_result_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "config": {
                        "runner_backend": "sol_slurm",
                        "remote_host": "user@host",
                        "remote_repo_root": "/remote/OlfactoryBulb",
                        "remote_results_root": "/remote/OlfactoryBulb/results/notebook_runs",
                        "ssh_transport": "paramiko",
                    },
                    "remote": {
                        "remote_result_dir": "/remote/OlfactoryBulb/results/notebook_runs/test_label",
                        "deferred_remote_artifacts": ["soma_vs.pkl"],
                    },
                }
            )
        )

        def _fake_sync_remote_result_dir(_config, *, remote_result_dir, local_result_dir, expected_files=None, include_files=None):
            assert remote_result_dir == PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label")
            assert include_files == ("soma_vs.pkl",)
            assert expected_files == ("soma_vs.pkl",)
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            with open(local_result_dir / "soma_vs.pkl", "wb") as handle:
                import pickle
                pickle.dump([("MC0", [0.0, 0.1], [-65.0, -64.0])], handle)
            return subprocess.CompletedProcess(args=["sync"], returncode=0, stdout="", stderr="")

        hlp._sync_remote_result_dir = _fake_sync_remote_result_dir
        result = hlp.load_result(deferred_result_dir)
        assert "soma_vs" in result
        assert result["soma_vs"][0][0] == "MC0"
        assert (deferred_result_dir / "soma_vs.pkl").exists()
        print("Deferred remote soma lazy sync: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir

    # --- Deferred soma traces should fall back to full-dir sync if selected-file sync fails ---
    original_sync_remote_result_dir = hlp._sync_remote_result_dir
    try:
        deferred_fallback_dir = tmp / "deferred-fallback-result"
        deferred_fallback_dir.mkdir(parents=True, exist_ok=True)
        (deferred_fallback_dir / "summary.json").write_text("{}")
        (deferred_fallback_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "config": {
                        "runner_backend": "sol_slurm",
                        "remote_host": "user@host",
                        "remote_repo_root": "/remote/OlfactoryBulb",
                        "remote_results_root": "/remote/OlfactoryBulb/results/notebook_runs",
                        "ssh_transport": "paramiko",
                    },
                    "remote": {
                        "remote_result_dir": "/remote/OlfactoryBulb/results/notebook_runs/test_label",
                        "deferred_remote_artifacts": ["soma_vs.pkl"],
                    },
                }
            )
        )

        sync_calls = []

        def _fake_sync_remote_result_dir(_config, *, remote_result_dir, local_result_dir, expected_files=None, include_files=None):
            sync_calls.append((remote_result_dir, expected_files, include_files))
            local_result_dir = Path(local_result_dir)
            local_result_dir.mkdir(parents=True, exist_ok=True)
            if include_files == ("soma_vs.pkl",):
                return subprocess.CompletedProcess(args=["sync"], returncode=1, stdout="", stderr="selected sync failed")
            with open(local_result_dir / "soma_vs.pkl", "wb") as handle:
                import pickle
                pickle.dump([("MC0", [0.0, 0.1], [-65.0, -64.0])], handle)
            return subprocess.CompletedProcess(args=["sync"], returncode=0, stdout="", stderr="")

        hlp._sync_remote_result_dir = _fake_sync_remote_result_dir
        result = hlp.load_result(deferred_fallback_dir)
        assert result["soma_vs"][0][0] == "MC0"
        assert sync_calls[0][2] == ("soma_vs.pkl",)
        assert sync_calls[1][2] is None
        print("Deferred soma fallback to full-dir sync: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir

    # --- Result overview should not trigger deferred soma trace downloads ---
    original_sync_remote_result_dir = hlp._sync_remote_result_dir
    try:
        overview_result_dir = tmp / "overview-result"
        overview_result_dir.mkdir(parents=True, exist_ok=True)
        (overview_result_dir / "summary.json").write_text(
            json.dumps(
                {
                    "label": "test_label",
                    "paramset": "GammaSignature",
                    "nranks": 16,
                    "params": {
                        "tstop": 3600.0,
                        "sim_dt": 0.1,
                        "actual_dt": 0.1,
                        "recording_period": 0.1,
                    },
                    "timing_seconds": {
                        "run_max_rank": 30.0,
                        "total_max_rank": 31.0,
                    },
                    "files": {
                        "input_times.pkl": {"items": 612},
                        "lfp.pkl": {"len_1": 36000},
                        "soma_vs.pkl": {"items": 193},
                    },
                }
            )
        )
        (overview_result_dir / "run_info.json").write_text(
            json.dumps(
                {
                    "config": {
                        "runner_backend": "sol_slurm",
                        "remote_host": "user@host",
                        "remote_repo_root": "/remote/OlfactoryBulb",
                        "remote_results_root": "/remote/OlfactoryBulb/results/notebook_runs",
                    },
                    "remote": {
                        "remote_result_dir": "/remote/OlfactoryBulb/results/notebook_runs/test_label",
                        "deferred_remote_artifacts": ["soma_vs.pkl"],
                    },
                }
            )
        )

        def _fail_sync_remote_result_dir(*args, **kwargs):
            raise AssertionError("result_overview should not trigger deferred soma trace sync")

        hlp._sync_remote_result_dir = _fail_sync_remote_result_dir
        result = hlp.load_result(overview_result_dir)
        info = hlp.result_overview(result)
        assert info["n_inputs"] == 612
        assert info["n_soma_traces"] == 193
        assert info["n_lfp_samples"] == 36000
        print("Result overview avoids deferred soma sync: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir

    # --- Zero-byte sync placeholders should never count as valid artifacts or loadable payload ---
    poisoned_dir = tmp / "poisoned-sync-dir"
    poisoned_dir.mkdir(parents=True, exist_ok=True)
    (poisoned_dir / "summary.json").write_bytes(b"")
    (poisoned_dir / "gc_output_events.pkl").write_bytes(b"")
    assert hlp._missing_local_sync_artifacts(
        poisoned_dir,
        expected_files=("summary.json",),
    ) == ["summary.json"]
    assert not hlp._local_result_dir_has_loadable_payload(poisoned_dir)
    print("Zero-byte sync placeholders are rejected: OK")

    # --- load_result should skip zero-byte payload placeholders instead of trying to unpickle them ---
    zero_payload_dir = tmp / "zero-payload-load"
    zero_payload_dir.mkdir(parents=True, exist_ok=True)
    with (zero_payload_dir / "input_times.pkl").open("wb") as handle:
        pickle.dump([1, 2, 3], handle)
    with (zero_payload_dir / "lfp.pkl").open("wb") as handle:
        pickle.dump((np.array([0.0, 0.1]), np.array([1.0, 2.0])), handle)
    (zero_payload_dir / "gc_output_events.pkl").write_bytes(b"")
    zero_loaded = hlp.load_result(zero_payload_dir)
    assert zero_loaded["input_times"] == [1, 2, 3]
    assert zero_loaded["gc_output_events"] == []
    assert zero_loaded["lfp"].shape == (2,)
    print("Zero-byte payload placeholders are skipped during load: OK")

print("\nAll tests passed.")
