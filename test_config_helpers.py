"""Simple smoke tests for the config save/load helpers in obgpu_experiment_helpers.

Run with:
    python test_config_helpers.py
"""

import json
import importlib.util
import io
import pickle
import socket
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath
from types import SimpleNamespace

import numpy as np
import obgpu_experiment_helpers as hlp
from neuroinfra.remote.deferred_artifacts import DeferredArtifactSyncHooks
from neuroinfra.remote.helper_bundle import helper_bundle_manifest
from neuroinfra.remote.result_sync import RemoteResultSyncHooks
import olfactorybulb.notebook_remote_runs as ob_remote_runs
from olfactorybulb.notebook_remote_sweeps import build_remote_sweep_payload

REMOTE_TOOLS_DIR = Path(__file__).resolve().parent / "tools" / "remote"
if str(REMOTE_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_TOOLS_DIR))

import slurm_common
import submit_sol_run

from olfactorybulb.result_artifacts import (
    DEFAULT_SOMA_TRACE_DTYPE,
    DEFAULT_SOMA_TRACE_FORMAT,
    SOMA_SPIKES_FILENAME_NPZ,
    SOMA_TRACE_FILENAME_NPZ,
    VOLTAGE_SUMMARY_FILENAME_NPZ,
    find_soma_trace_artifact,
    load_soma_spike_artifact,
    load_soma_trace_artifact,
    load_voltage_summary_artifact,
    save_soma_spike_artifact,
    save_soma_trace_artifact,
    save_voltage_summary_artifact,
)
from obgpu_experiment_helpers import (
    build_param_overrides,
    build_run_config,
    config_diff,
    config_from_run,
    list_paramsets,
    list_notebook_runs,
    list_saved_configs,
    load_run_record,
    load_config,
    print_run_summary,
    save_config,
    save_figure,
)

assert "notebook-facing facade" in (hlp.__doc__ or "").lower()
assert "neuroinfra" in (hlp.__doc__ or "").lower()
print("notebook facade ownership contract: OK")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # --- save_config / load_config round-trip ---
    cfg = build_run_config(paramset="GammaSignature", gaba_tau2_ms=36.0, gap_mc=32.0)
    assert "legacy_parallel_dt" not in cfg
    assert cfg["remote_defer_soma_vs_sync"] is False
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

    # --- notebook run catalog wrappers ---
    run_dir = tmp / "2026-06-02_demo_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"label": "demo-summary", "timestamp": "2026-06-02T00:00:00"}))
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "requested_label": "demo-requested",
                "timestamp": "2026-06-02T00:00:00",
                "config": cfg,
                "overrides": {"gaba_tau2_ms": 36.0},
                "command": ["python", "demo.py"],
            }
        )
    )
    (run_dir / "stdout.txt").write_text("demo stdout\n")
    listed_runs = list_notebook_runs(results_base=tmp)
    assert run_dir in listed_runs
    record = load_run_record(run_dir, results_base=tmp)
    assert record.label == "demo-summary"
    assert record.command == ["python", "demo.py"]
    restored_cfg = config_from_run(run_dir, results_base=tmp)
    assert restored_cfg["paramset"] == "GammaSignature"
    assert restored_cfg["gaba_tau2_ms"] == 36.0
    print("notebook run catalog wrappers: OK")

    # --- sweep planning wrappers ---
    sweep_plan = hlp._prepare_sweep_plan(
        cfg,
        "gaba_tau2_ms",
        [36.0, 50.0],
    )
    assert [item["value"] for item in sweep_plan["items"]] == [36.0, 50.0]
    assert sweep_plan["items"][1]["config"]["gaba_tau2_ms"] == 50.0

    joint_plan = hlp._prepare_sweep_plan(
        cfg,
        {"gaba_tau2_ms": [36.0, 50.0], "gap_mc": [16.0, 32.0]},
    )
    assert joint_plan["items"][0]["config"]["gap_mc"] == 16.0
    assert joint_plan["items"][1]["config"]["gaba_tau2_ms"] == 50.0

    grid_plan = hlp._prepare_sweep_plan(
        cfg,
        {"gaba_tau2_ms": [36.0, 50.0], "gap_mc": [16.0, 32.0]},
        grid=True,
    )
    assert len(grid_plan["items"]) == 4
    assert grid_plan["items"][2]["value"] == {"gaba_tau2_ms": 50.0, "gap_mc": 16.0}
    print("sweep planning wrappers: OK")

    # --- local run wrapper delegates through notebook dispatch layer ---
    original_run_dispatch = hlp._ob_run_notebook_simulation
    local_run_calls = []
    try:
        def _fake_run_dispatch(hooks, config=None, *, label=None):
            local_run_calls.append((hooks, dict(config or {}), label))
            return "LOCAL-RUN-SENTINEL"

        hlp._ob_run_notebook_simulation = _fake_run_dispatch
        delegated = hlp.run_simulation(
            build_run_config(
                paramset="GammaSignature",
                results_base=str(tmp / "local_runs"),
            ),
            label="local_runner_delegate",
        )
        assert delegated == "LOCAL-RUN-SENTINEL"
        assert local_run_calls and local_run_calls[0][2] == "local_runner_delegate"
        assert local_run_calls[0][1]["results_base"] == str(tmp / "local_runs")
    finally:
        hlp._ob_run_notebook_simulation = original_run_dispatch
    print("local run wrapper delegation: OK")

    # --- remote run wrapper delegates through notebook remote session layer ---
    original_remote_session = ob_remote_runs.prepare_remote_job_session
    original_write_run_info = hlp._write_notebook_run_info
    remote_session_calls = []
    remote_run_info_calls = []
    try:
        def _fake_prepare_remote_job_session(
            config,
            *,
            remote_repo_root,
            remote_git_ref,
            remote_metadata,
            preflight_message,
            hooks,
            notebook_timings=None,
        ):
            remote_session_calls.append(
                {
                    "config": dict(config),
                    "remote_repo_root": remote_repo_root,
                    "remote_git_ref": remote_git_ref,
                    "remote_metadata": dict(remote_metadata),
                    "preflight_message": preflight_message,
                }
            )
            return SimpleNamespace(
                effective_config=dict(config),
                remote_metadata=dict(remote_metadata),
                notebook_timings={},
                preflight_completed=subprocess.CompletedProcess(
                    ["preflight"],
                    2,
                    stdout="",
                    stderr="bad preflight\n",
                ),
                remote_helper_dir=None,
                allocation_info={},
                allocation_heartbeat_path=None,
            )

        ob_remote_runs.prepare_remote_job_session = _fake_prepare_remote_job_session
        hlp._write_notebook_run_info = lambda *args, **kwargs: remote_run_info_calls.append((args, kwargs))

        remote_cfg = build_run_config(
            paramset="GammaSignature",
            runner_backend="slurm_remote",
            remote_host="user@host",
            remote_repo_root="/remote/OlfactoryBulb",
            remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
            results_base=str(tmp / "remote_runs"),
        )
        try:
            hlp.run_simulation(remote_cfg, label="remote_delegate")
            raise AssertionError("expected remote preflight failure to raise")
        except RuntimeError as exc:
            assert "Remote Sol preflight failed." in str(exc)
        assert remote_session_calls
        assert remote_session_calls[0]["preflight_message"] == "[Sol remote] Running remote preflight checks..."
        assert remote_run_info_calls
        assert remote_run_info_calls[0][1]["completed"].returncode == 2
    finally:
        ob_remote_runs.prepare_remote_job_session = original_remote_session
        hlp._write_notebook_run_info = original_write_run_info
    print("remote run workflow delegation: OK")

    # --- remote single-run wrapper delegates through neuroinfra remote run workflow ---
    original_remote_run_workflow = hlp._neuroinfra_execute_remote_run_workflow
    remote_run_workflow_calls = []
    try:
        def _fake_remote_run_workflow(
            config,
            *,
            record_config,
            label,
            timestamp,
            local_result_dir,
            remote_repo_root,
            remote_git_ref,
            remote_command,
            remote_metadata,
            submit_shell,
            hooks,
        ):
            remote_run_workflow_calls.append(
                {
                    "config": dict(config),
                    "record_config": dict(record_config),
                    "label": label,
                    "timestamp": timestamp,
                    "local_result_dir": Path(local_result_dir),
                    "remote_repo_root": remote_repo_root,
                    "remote_git_ref": remote_git_ref,
                    "remote_command": list(remote_command),
                    "remote_metadata": dict(remote_metadata),
                    "submit_shell": submit_shell,
                }
            )
            return "REMOTE-RUN-WORKFLOW-SENTINEL"

        hlp._neuroinfra_execute_remote_run_workflow = _fake_remote_run_workflow
        delegated = hlp.run_simulation(
            build_run_config(
                paramset="GammaSignature",
                runner_backend="slurm_remote",
                remote_host="user@host",
                remote_repo_root="/remote/OlfactoryBulb",
                remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
                results_base=str(tmp / "remote_runs_workflow"),
            ),
            label="remote_workflow_delegate",
        )
        assert delegated == "REMOTE-RUN-WORKFLOW-SENTINEL"
        assert remote_run_workflow_calls
        assert remote_run_workflow_calls[0]["label"] == "remote_workflow_delegate"
        assert remote_run_workflow_calls[0]["remote_repo_root"] == PurePosixPath("/remote/OlfactoryBulb")
    finally:
        hlp._neuroinfra_execute_remote_run_workflow = original_remote_run_workflow
    print("remote single-run workflow delegation: OK")

    # --- remote sweep wrapper delegates through neuroinfra remote sweep workflow ---
    original_remote_sweep_workflow = hlp._neuroinfra_execute_remote_sweep_workflow
    remote_sweep_workflow_calls = []
    try:
        def _fake_remote_sweep_workflow(
            config,
            *,
            sweep_plan,
            sweep_label,
            timestamp,
            local_sweep_dir,
            local_runs_dir,
            remote_repo_root,
            remote_git_ref,
            remote_sweeps_root,
            remote_sweep_root,
            manifest_items,
            manifest_json,
            max_concurrent,
            remote_metadata,
            hooks,
        ):
            remote_sweep_workflow_calls.append(
                {
                    "config": dict(config),
                    "sweep_label": sweep_label,
                    "timestamp": timestamp,
                    "local_sweep_dir": Path(local_sweep_dir),
                    "local_runs_dir": Path(local_runs_dir),
                    "remote_repo_root": remote_repo_root,
                    "remote_git_ref": remote_git_ref,
                    "remote_sweeps_root": remote_sweeps_root,
                    "remote_sweep_root": remote_sweep_root,
                    "manifest_items": manifest_items,
                    "manifest_json": manifest_json,
                    "max_concurrent": max_concurrent,
                    "remote_metadata": dict(remote_metadata),
                }
            )
            return {"items": [], "path": sweep_plan["path"], "values": sweep_plan["values"], "paramset": sweep_plan["paramset"]}

        hlp._neuroinfra_execute_remote_sweep_workflow = _fake_remote_sweep_workflow
        remote_cfg = build_run_config(
            paramset="GammaSignature",
            runner_backend="slurm_remote",
            remote_host="user@host",
            remote_repo_root="/remote/OlfactoryBulb",
            remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
            sweep_engine="remote_batch",
            results_base=str(tmp / "remote_sweeps_workflow"),
        )
        sweep_result = hlp.run_parameter_sweep(remote_cfg, "gaba_tau2_ms", [36.0, 50.0])
        assert sweep_result["paramset"] == "GammaSignature"
        assert remote_sweep_workflow_calls
        assert remote_sweep_workflow_calls[0]["remote_repo_root"] == PurePosixPath("/remote/OlfactoryBulb")
        assert len(remote_sweep_workflow_calls[0]["manifest_items"]) == 2
        assert remote_sweep_workflow_calls[0]["remote_metadata"]["sweep_items"] == 2
    finally:
        hlp._neuroinfra_execute_remote_sweep_workflow = original_remote_sweep_workflow
    print("remote sweep workflow delegation: OK")

    # --- run_and_load wrapper delegates through neuroinfra workflow ---
    original_run_and_load_workflow = hlp._neuroinfra_run_and_load_workflow
    run_and_load_calls = []
    try:
        def _fake_run_and_load_workflow(hooks, config=None, *, label=None):
            run_and_load_calls.append((hooks, config, label))
            return SimpleNamespace(result_dir=tmp / "delegated-run", label=label), {"artifact_sizes": {}}

        hlp._neuroinfra_run_and_load_workflow = _fake_run_and_load_workflow
        delegated_run, delegated_result = hlp.run_and_load(cfg, label="delegated")
        assert delegated_run.label == "delegated"
        assert delegated_result == {"artifact_sizes": {}}
        assert run_and_load_calls and run_and_load_calls[0][2] == "delegated"
    finally:
        hlp._neuroinfra_run_and_load_workflow = original_run_and_load_workflow
    print("run_and_load wrapper delegation: OK")

    # --- local sweep wrapper delegates through notebook dispatch layer ---
    original_local_sweep_dispatch = hlp._ob_run_notebook_parameter_sweep
    local_sweep_calls = []
    try:
        def _fake_local_sweep_dispatch(hooks, base_config, sweep_path, values=None):
            local_sweep_calls.append((hooks, dict(base_config), sweep_path, list(values or [])))
            return {"items": [], "path": sweep_path, "values": list(values or []), "paramset": base_config["paramset"]}

        hlp._ob_run_notebook_parameter_sweep = _fake_local_sweep_dispatch
        sweep_result = hlp.run_parameter_sweep(cfg, "gaba_tau2_ms", [36.0, 50.0])
        assert sweep_result["paramset"] == "GammaSignature"
        assert local_sweep_calls and local_sweep_calls[0][2] == "gaba_tau2_ms"
        assert local_sweep_calls[0][3] == [36.0, 50.0]
    finally:
        hlp._ob_run_notebook_parameter_sweep = original_local_sweep_dispatch
    print("local sweep workflow delegation: OK")

    # --- save_figure wrapper ---
    class _FakeFigure:
        def __init__(self) -> None:
            self.calls = []

        def savefig(self, path, **kwargs) -> None:
            self.calls.append((Path(path), kwargs))
            Path(path).write_text("fake-figure")

    figure = _FakeFigure()
    saved_figure = save_figure("Notebook Summary", fig=figure, run_or_result=record)
    assert saved_figure.parent == run_dir
    assert saved_figure.name.endswith(".png")
    assert saved_figure.exists()
    assert figure.calls[0][1]["dpi"] == 200
    print("save_figure wrapper: OK")

    # --- print_run_summary wrapper ---
    summary_result = {
        "result_dir": str(run_dir),
        "summary": {
            "label": "demo-summary",
            "paramset": "GammaSignature",
            "params": {"tstop": 1000.0, "sim_dt": 0.1},
        },
        "run_info": {"config": cfg},
        "input_times": [],
        "gc_output_events": [],
        "lfp": [],
    }
    summary_stdout = io.StringIO()
    with redirect_stdout(summary_stdout):
        print_run_summary(record, summary_result)
    summary_text = summary_stdout.getvalue()
    assert "Effective inputs:" in summary_text
    assert "Runtime and analysis controls:" in summary_text
    assert "Result directory:" in summary_text
    assert "Command: python demo.py" in summary_text
    print("print_run_summary wrapper: OK")

    # --- run_info wrapper paths ---
    run_info_dir = tmp / "run_info_case"
    completed = SimpleNamespace(returncode=0, stdout="", stderr="")
    run_info_path = hlp._write_notebook_run_info(
        run_info_dir,
        config=cfg,
        label="demo-run",
        timestamp="2026-06-02T00:00:00",
        command=["python", "demo.py"],
        env={
            "OB_RUN_TIMESTAMP": "20260602_000000",
            "OB_RESULT_LABEL": "demo-run",
            "OB_CORENRN_CELL_PERMUTE": "2",
            "OB_RESULTS_BASE": str(tmp),
        },
        completed=completed,
        summary={"label": "demo-run"},
        extra_payload={"remote": None},
    )
    assert run_info_path.name == "run_info.json"
    payload = json.loads(run_info_path.read_text())
    assert payload["label"] == "demo-run"
    assert payload["runner"] == "obgpu_experiment_helpers.run_simulation"
    assert payload["env"]["OB_RESULT_LABEL"] == "demo-run"
    hlp._merge_run_info_payload(run_info_dir, {"artifact_sizes": {"lfp.pkl": 11}})
    merged_payload = json.loads(run_info_path.read_text())
    assert merged_payload["artifact_sizes"]["lfp.pkl"] == 11
    print("run_info wrapper paths: OK")

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
    assert "legacy_parallel_dt" not in overrides
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

    # --- EPLI opt-in controls should bridge into param overrides cleanly ---
    cfg_epli = build_run_config(
        enable_epl_interneurons=True,
        max_epl_interneurons=12,
        epli_gaba_weight_scale=2.5,
        epli_ampa_weight_scale=1.75,
        gc_gaba_weight_scale=0.8,
        gc_ampa_weight_scale=1.2,
    )
    epli_overrides = build_param_overrides(cfg_epli)
    assert "legacy_parallel_dt" not in epli_overrides
    assert epli_overrides["enable_epl_interneurons"] is True
    assert epli_overrides["max_epl_interneurons"] == 12
    assert epli_overrides["epli_gaba_weight_scale"] == 2.5
    assert epli_overrides["epli_ampa_weight_scale"] == 1.75
    assert epli_overrides["gc_gaba_weight_scale"] == 0.8
    assert epli_overrides["gc_ampa_weight_scale"] == 1.2
    assert "EPLI" in epli_overrides["record_from_somas"]

    cfg_custom_epli = build_run_config(
        record_from_somas=["MC"],
        enable_epl_interneurons=True,
        max_epl_interneurons=4,
        epl_interneuron_cell_type="PVI",
    )
    custom_epli_overrides = build_param_overrides(cfg_custom_epli)
    assert custom_epli_overrides["epl_interneuron_cell_type"] == "PVI"
    assert custom_epli_overrides["record_from_somas"] == ["MC", "PVI"]
    print("EPLI controls: OK")

    # --- Cached Paramiko auth should enable silent reconnects without notebook prompts ---
    prompt_cfg = hlp.build_run_config(remote_host="jmpaniag@localhost", remote_preserve_paramiko_session=True)
    prompt_key = hlp._paramiko_connection_key(prompt_cfg)
    saved_prompt_cache = deepcopy(hlp._LIVE_PARAMIKO_PROMPT_CACHE)
    saved_authenticated = set(hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS)
    original_getpass = hlp.getpass
    original_ipython_module = sys.modules.get("IPython")
    try:
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.add(prompt_key)
        assert hlp._paramiko_can_reconnect(prompt_cfg) is False
        hlp._cache_paramiko_prompt_response(prompt_cfg, "Password for jmpaniag@localhost:", "cached-secret")
        assert hlp._paramiko_can_reconnect(prompt_cfg) is True
        assert (
            hlp._paramiko_prompt_response("Password for jmpaniag@localhost:", config=prompt_cfg)
            == "cached-secret"
        )

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp.getpass = _raise_eof
        try:
            hlp._paramiko_prompt_response("Password for jmpaniag@localhost:", config=prompt_cfg)
            raise AssertionError("expected notebook auth prompt to raise RuntimeError on EOF")
        except RuntimeError as exc:
            assert "could not read notebook input" in str(exc)
            assert "paramiko_auth_probe(REMOTE_CONFIG)" in str(exc)

        class _FakeKernel:
            def getpass(self, prompt: str) -> str:
                assert "Password for jmpaniag@localhost:" in prompt
                return "kernel-secret"

            def raw_input(self, prompt: str) -> str:
                assert "OTP code:" in prompt
                return "123456"

        class _FakeShell:
            kernel = _FakeKernel()

        sys.modules["IPython"] = SimpleNamespace(get_ipython=lambda: _FakeShell())
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp.getpass = _raise_eof
        assert hlp._paramiko_prompt_response("Password for jmpaniag@localhost:", config=prompt_cfg) == "kernel-secret"
        assert hlp._paramiko_prompt_response("OTP code:", config=prompt_cfg) == "123456"
    finally:
        hlp.getpass = original_getpass
        if original_ipython_module is None:
            sys.modules.pop("IPython", None)
        else:
            sys.modules["IPython"] = original_ipython_module
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.update(saved_prompt_cache)
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.update(saved_authenticated)
    print("Paramiko cached auth / EOF guard: OK")

    # --- Paramiko should retry transient SSH banner failures before giving up ---
    retry_cfg = dict(prompt_cfg)
    retry_cfg["ssh_connect_retries"] = 2
    retry_cfg["ssh_connect_retry_backoff_s"] = 0.0
    retry_key = hlp._paramiko_connection_key(retry_cfg)
    saved_connections = deepcopy(hlp._LIVE_PARAMIKO_CONNECTIONS)
    saved_prompt_cache = deepcopy(hlp._LIVE_PARAMIKO_PROMPT_CACHE)
    saved_authenticated = set(hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS)
    original_create_connection = socket.create_connection
    original_sleep = hlp.time.sleep
    original_transport = hlp.paramiko.Transport
    attempts = {"start_client": 0, "interactive": 0}
    sleeps: list[float] = []

    class _FakeSocket:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _FakeTransport:
        def __init__(self, sock: _FakeSocket) -> None:
            self.sock = sock
            self._authenticated = False
            self._active = True

        def start_client(self, timeout: float = 30.0) -> None:
            attempts["start_client"] += 1
            if attempts["start_client"] == 1:
                self._active = False
                raise hlp.paramiko.SSHException("Error reading SSH protocol banner")

        def set_keepalive(self, seconds: int) -> None:
            self.keepalive = seconds

        def auth_none(self, username: str) -> None:
            raise hlp.paramiko.BadAuthenticationType("auth", ["keyboard-interactive"])

        def auth_interactive(self, username: str, handler) -> None:
            attempts["interactive"] += 1
            responses = handler("", "", [("Password for jmpaniag@localhost:", False)])
            assert responses == ["cached-secret"]
            self._authenticated = True
            self._active = True

        def is_authenticated(self) -> bool:
            return self._authenticated

        def is_active(self) -> bool:
            return self._active

        def close(self) -> None:
            self._active = False
            self.sock.close()

    def _fake_create_connection(_endpoint, timeout: float = 30.0) -> _FakeSocket:
        return _FakeSocket()

    try:
        hlp._LIVE_PARAMIKO_CONNECTIONS.clear()
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.clear()
        hlp._cache_paramiko_prompt_response(retry_cfg, "Password for jmpaniag@localhost:", "cached-secret")
        socket.create_connection = _fake_create_connection
        hlp.time.sleep = lambda seconds: sleeps.append(float(seconds))
        hlp.paramiko.Transport = _FakeTransport
        connection = hlp._connect_paramiko(retry_cfg)
        assert hlp._LIVE_PARAMIKO_CONNECTIONS[retry_key] is connection
        assert attempts["start_client"] == 2
        assert attempts["interactive"] == 1
        assert sleeps == []
        assert connection["transport"].is_authenticated() is True
    finally:
        socket.create_connection = original_create_connection
        hlp.time.sleep = original_sleep
        hlp.paramiko.Transport = original_transport
        hlp._LIVE_PARAMIKO_CONNECTIONS.clear()
        hlp._LIVE_PARAMIKO_CONNECTIONS.update(saved_connections)
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.update(saved_prompt_cache)
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.update(saved_authenticated)
    print("Paramiko transient reconnect retry: OK")

    # --- Compressed soma trace artifacts should default to float32 NPZ and round-trip cleanly ---
    assert cfg["soma_trace_format"] == DEFAULT_SOMA_TRACE_FORMAT
    assert cfg["soma_trace_dtype"] == DEFAULT_SOMA_TRACE_DTYPE
    traces = [
        ("MC0", [0.0, 0.1, 0.2], [-65.0, -64.5, -64.0]),
        ("TC0", [0.0, 0.1, 0.2], [-63.0, -62.5, -62.0]),
    ]
    npz_path = save_soma_trace_artifact(
        traces,
        tmp,
        trace_format="npz",
        trace_dtype="float32",
    )
    assert npz_path.name == SOMA_TRACE_FILENAME_NPZ
    loaded_traces = load_soma_trace_artifact(npz_path)
    assert [row[0] for row in loaded_traces] == ["MC0", "TC0"]
    assert loaded_traces[0][1].dtype == np.float32
    assert loaded_traces[0][2].dtype == np.float32
    assert find_soma_trace_artifact(tmp) == npz_path
    benchmark_spec = importlib.util.spec_from_file_location(
        "benchmark_ob_test",
        hlp.REPO_ROOT / "tools" / "benchmarks" / "benchmark_ob.py",
    )
    assert benchmark_spec is not None and benchmark_spec.loader is not None
    benchmark_ob = importlib.util.module_from_spec(benchmark_spec)
    benchmark_spec.loader.exec_module(benchmark_ob)
    soma_summary = benchmark_ob.summarize_pickle(npz_path)
    assert soma_summary["type"] == "list"
    assert soma_summary["items"] == 2
    assert "canonical_sha256" in soma_summary
    print("Compressed soma trace artifact round-trip: OK")

    # --- Compact soma spike and voltage summary artifacts support no-raw-soma analysis ---
    compact_dir = tmp / "compact_artifacts"
    compact_dir.mkdir()
    compact_traces = [
        ("MC0[0].soma", [0, 1, 2, 3, 4, 5, 6], [-65, -20, 30, -62, -12, 35, -64]),
        ("MC1[0].soma", [0, 1, 2, 3, 4, 5, 6], [-64, -18, 32, -61, -11, 34, -63]),
        ("TC0[0].soma", [0, 1, 2, 3, 4, 5, 6], [-60, -60, -58, -59, -58, -60, -59]),
    ]
    spike_path = save_soma_spike_artifact(compact_traces, compact_dir, threshold=0.0)
    voltage_summary_path = save_voltage_summary_artifact(compact_traces, compact_dir)
    assert spike_path.name == SOMA_SPIKES_FILENAME_NPZ
    assert voltage_summary_path.name == VOLTAGE_SUMMARY_FILENAME_NPZ
    loaded_spikes = load_soma_spike_artifact(compact_dir)
    assert loaded_spikes["metadata"]["threshold_mv"] == 0.0
    assert [len(row) for row in loaded_spikes["spike_times"][:2]] == [2, 2]
    loaded_voltage_summary = load_voltage_summary_artifact(compact_dir)
    assert "MC" in loaded_voltage_summary["mean_by_type"]
    np.testing.assert_allclose(
        loaded_voltage_summary["mean_by_type"]["MC"],
        np.mean(np.asarray([compact_traces[0][2], compact_traces[1][2]], dtype=float), axis=0),
        atol=1e-5,
    )

    loaded_result = hlp.load_result(compact_dir)
    assert dict.get(loaded_result, "soma_vs") == []
    freq_samples = hlp.collect_spike_frequency_samples(
        loaded_result,
        cell_types=("MC",),
        threshold=0.0,
    )
    assert freq_samples["n_traces"] == 2
    assert len(freq_samples["freqs"]) == 2
    t_mean, v_mean = hlp.get_named_signal(loaded_result, signal="mean_MC_voltage")
    assert len(t_mean) == len(v_mean) == 7
    assert "MC" in hlp.list_available_cell_types(loaded_result)
    assert "mean_MC_voltage" in hlp.list_available_named_signals(loaded_result)
    assert "soma_vs" not in loaded_result._lazy_loaders
    print("Compact soma spike / voltage-summary artifacts: OK")

    # --- Phase-lock wrapper should reuse saved spike artifacts against named signals ---
    phase_t = np.arange(0.0, 1000.0, 0.1, dtype=float)
    phase_signal = np.sin(2.0 * np.pi * 100.0 * phase_t / 1000.0)
    phase_spikes = np.arange(52.5, 950.0, 10.0, dtype=float)
    phase_result = {
        "lfp_t": phase_t,
        "lfp": phase_signal,
        "soma_spikes": {
            "labels": ["MC0[0].soma"],
            "spike_times": [phase_spikes],
            "metadata": {"threshold_mv": 0.0},
        },
    }
    phase_summary = hlp.compute_spike_phase_locking(
        phase_result,
        signal="lfp",
        band=(80.0, 120.0),
        cell_types=("MC",),
        threshold=0.0,
        dt_ms=0.1,
    )
    assert phase_summary["signal"] == "lfp"
    assert phase_summary["cell_types"] == ["MC"]
    assert phase_summary["n_spikes"] == len(phase_spikes)
    assert phase_summary["vector_strength"] > 0.95
    assert len(phase_summary["per_cell"]) == 1
    assert phase_summary["per_cell"][0]["label"] == "MC0[0].soma"
    print("Phase-lock wrapper: OK")

    bp_t, bp_y = hlp.compute_lfp_bandpassed(
        phase_result,
        dt_ms=0.1,
        lowcut_hz=80.0,
        highcut_hz=120.0,
    )
    assert len(bp_t) == len(phase_t)
    assert len(bp_y) == len(phase_signal)

    signal_ax = hlp.plot_named_signal(
        phase_result,
        signal="lfp",
        dt_ms=0.1,
    )
    try:
        assert signal_ax.get_title() == "lfp Trace"
        assert len(signal_ax.lines) == 1
    finally:
        hlp.plt.close(signal_ax.figure)

    spectrogram_ax = hlp.plot_spectrogram(
        phase_result,
        signal="lfp",
        dt_ms=0.1,
        nperseg=128,
        noverlap=96,
    )
    try:
        assert spectrogram_ax.get_title() == "LFP Spectrogram"
        assert len(spectrogram_ax.collections) > 0
    finally:
        hlp.plt.close(spectrogram_ax.figure)

    hfo_fig, hfo_axes, hfo_summary = hlp.plot_hfo_power_summary(
        phase_result,
        signal="lfp",
        dt_ms=0.1,
        bands={"hfo": (80.0, 120.0)},
        relative_band=(60.0, 140.0),
    )
    try:
        assert hfo_summary["signal"] == "lfp"
        assert hfo_axes[0].get_title() == "lfp Band Power"
        assert hfo_axes[1].get_title() == "Relative Band Power"
    finally:
        hlp.plt.close(hfo_fig)

    lfp_fig, lfp_axes = hlp.plot_lfp_overview(
        phase_result,
        dt_ms=0.1,
        lowcut_hz=80.0,
        highcut_hz=120.0,
        show_psd_target_template=False,
    )
    try:
        assert len(lfp_axes) == 3
        assert lfp_axes[0].get_title() == "Raw LFP"
        assert lfp_axes[1].get_title() == "Band-passed LFP (80-120 Hz)"
        assert lfp_axes[2].get_title() == "Welch Power Spectrum"
    finally:
        hlp.plt.close(lfp_fig)
    print("Resolved signal view wrappers: OK")

    freq_ax = hlp.plot_spike_frequency_kde_1d(
        loaded_result,
        cell_types=("MC",),
        threshold=0.0,
        config={"max_freq_hz": 400.0},
    )
    try:
        assert freq_ax.get_title() == "Soma Spike Frequency Distribution (MC)"
        assert freq_ax.get_xlabel() == "Frequency (Hz)"
    finally:
        hlp.plt.close(freq_ax.figure)
    print("Frequency KDE wrapper: OK")

    voltage_ax = hlp.plot_voltage_traces(
        {
            "soma_vs": [
                (label, np.asarray(t, dtype=float), np.asarray(v, dtype=float))
                for label, t, v in compact_traces
            ]
        },
        max_per_type=1,
    )
    try:
        assert voltage_ax.get_title() == "Sample Soma Voltages (MT grouped)"
        assert len(voltage_ax.lines) == 1
    finally:
        hlp.plt.close(voltage_ax.figure)

    spike_ax = hlp.plot_spike_raster(
        loaded_result,
        max_cells_per_type=1,
        threshold=0.0,
    )
    try:
        assert spike_ax.get_title() == "Detected Soma Spike Raster (MT grouped)"
        assert len(spike_ax.collections) > 0
        assert len(spike_ax.get_yticklabels()) == 1
    finally:
        hlp.plt.close(spike_ax.figure)
    print("Voltage / spike wrapper grouping: OK")

    gc_freq_result = {
        "gc_output_events": [
            {
                "source_section": "GC0[0].soma",
                "dest_section": "MC0[0].soma",
                "times": np.array([0.0, 20.0, 50.0], dtype=float),
            },
            {
                "source_section": "GC1[0].soma",
                "dest_section": "TC0[0].soma",
                "times": np.array([10.0], dtype=float),
            },
        ]
    }
    gc_freq_samples = hlp.collect_gc_output_frequency_samples(
        gc_freq_result,
        target_types=("MC",),
        modulus=40.0,
    )
    assert gc_freq_samples["n_events"] == 1
    assert len(gc_freq_samples["events"]) == 1
    np.testing.assert_allclose(gc_freq_samples["times"], [10.0, 35.0])
    np.testing.assert_allclose(gc_freq_samples["freqs"], [50.0, 33.3333333333])
    print("GC output frequency wrapper: OK")

    input_rate_result = {
        "input_times": [
            ("h.MC0[0].tuft", np.array([10.0, 20.0, 30.0], dtype=float)),
            ("h.TC0[0].tuft", np.array([15.0, 25.0], dtype=float)),
        ],
        "lfp_t": np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=float),
    }
    filtered_inputs = hlp.filter_input_events(input_rate_result, target_types=("MC",))
    assert len(filtered_inputs) == 1
    input_t, input_rate_hz, input_meta = hlp.compute_input_rate(
        input_rate_result,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        normalization="per_cell",
        return_metadata=True,
    )
    assert input_meta["normalization"] == "per_target_cell"
    assert input_meta["n_target_cells"] == 2
    np.testing.assert_allclose(input_t, [5.0, 15.0, 25.0, 35.0])
    np.testing.assert_allclose(input_rate_hz, [0.0, 100.0, 100.0, 50.0])

    gc_rate_t, gc_rate_hz, gc_rate_meta = hlp.compute_gc_output_rate(
        gc_freq_result,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        normalization="per_target_cell",
        return_metadata=True,
    )
    assert gc_rate_meta["n_connections"] == 2
    assert gc_rate_meta["n_target_cells"] == 2
    np.testing.assert_allclose(gc_rate_t, [5.0, 15.0, 25.0, 35.0, 45.0])
    np.testing.assert_allclose(gc_rate_hz, [50.0, 50.0, 50.0, 0.0, 50.0])

    input_raster_ax = hlp.plot_input_raster(input_rate_result, max_segments=10)
    try:
        assert input_raster_ax.get_title() == "Odor Input Raster"
        assert len(input_raster_ax.collections) > 0
        assert [tick.get_text() for tick in input_raster_ax.get_yticklabels()] == ["MC0[0].tuft", "TC0[0].tuft"]
    finally:
        hlp.plt.close(input_raster_ax.figure)

    input_overview_fig, input_overview_axes = hlp.plot_input_overview(
        input_rate_result,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        max_segments=10,
    )
    try:
        assert input_overview_axes.shape == (2,)
        assert input_overview_axes[0].get_title() == "Odor Input Raster"
        assert input_overview_axes[1].get_title() == "Odor Input Event Rate"
    finally:
        hlp.plt.close(input_overview_fig)

    gc_raster_ax = hlp.plot_gc_output_event_raster(gc_freq_result, max_connections=10)
    try:
        assert gc_raster_ax.get_title() == "GC Inhibitory Output Events"
        assert len(gc_raster_ax.collections) > 0
    finally:
        hlp.plt.close(gc_raster_ax.figure)

    gc_overview_fig, gc_overview_axes = hlp.plot_gc_output_overview(
        gc_freq_result,
        bin_ms=10.0,
        smooth_sigma_ms=0.0,
        max_connections=10,
    )
    try:
        assert gc_overview_axes.shape == (2,)
        assert gc_overview_axes[0].get_title() == "GC Inhibitory Output Events"
        assert gc_overview_axes[1].get_title() == "GC Inhibitory Output Rate"
    finally:
        hlp.plt.close(gc_overview_fig)
    print("Input / GC rate wrappers: OK")

    # --- Optional int16 soma traces should round-trip with bounded quantization error ---
    quantized_dir = tmp / "quantized_soma"
    quantized_dir.mkdir()
    q_path = save_soma_trace_artifact(
        compact_traces,
        quantized_dir,
        trace_format="npz",
        trace_dtype="int16",
    )
    q_loaded = load_soma_trace_artifact(q_path)
    for (_label, _t, expected_v), (_loaded_label, _loaded_t, observed_v) in zip(compact_traces, q_loaded):
        np.testing.assert_allclose(observed_v, expected_v, atol=0.01)
    ragged_quantized_dir = tmp / "ragged_quantized_soma"
    ragged_quantized_dir.mkdir()
    ragged_quantized_traces = [
        ("MC_empty", [], []),
        ("TC_short", [0.0, 0.1], [-64.0, -63.0]),
        ("GC_single", [0.0], [-70.0]),
    ]
    ragged_q_path = save_soma_trace_artifact(
        ragged_quantized_traces,
        ragged_quantized_dir,
        trace_format="npz",
        trace_dtype="int16",
    )
    ragged_q_loaded = load_soma_trace_artifact(ragged_q_path)
    assert [label for label, _t, _v in ragged_q_loaded] == ["MC_empty", "TC_short", "GC_single"]
    assert len(ragged_q_loaded[0][1]) == 0
    assert len(ragged_q_loaded[0][2]) == 0
    np.testing.assert_allclose(ragged_q_loaded[1][2], [-64.0, -63.0], atol=0.01)
    np.testing.assert_allclose(ragged_q_loaded[2][2], [-70.0], atol=0.01)
    print("Quantized int16 soma trace artifact: OK")

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
        hlp._LIVE_PARAMIKO_CONNECTIONS.pop(cache_key, None)
        try:
            hlp._connect_paramiko(fake_remote_cfg)
            raise AssertionError("Expected missing cached Paramiko transport to be refused")
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
    driver_command, manifest_items, manifest_json, manifest_path, _parallelism, _metadata = build_remote_sweep_payload(
        hlp._ob_remote_sweep_payload_hooks(),
        sweep_cfg,
        sweep_plan=sweep_plan,
        sweep_label="test_sweep",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_sweeps_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
        remote_sweep_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps/test_sweep"),
        remote_git_ref="HEAD",
    )
    assert "--items-json" in driver_command
    assert "--items-b64" not in driver_command
    assert manifest_path.as_posix() in driver_command
    parsed_manifest = json.loads(manifest_json)
    assert isinstance(parsed_manifest, list) and len(parsed_manifest) == len(manifest_items)
    assert parsed_manifest[0]["label"] == manifest_items[0]["label"]

    # --- Sweep parallelism should use optimizer task-size metadata even for non-srun launchers ---
    assert hlp._remote_sweep_parallelism(
        {
            "remote_mpi_exec": "mpiexec --bind-to none",
            "optimizer_total_tasks": 120,
            "nranks": 15,
        },
        tasks_per_item=15,
    ) == 8
    assert hlp._remote_sweep_parallelism(
        {
            "remote_mpi_exec": "srun --mpi=pmix_v4 --cpu-bind=none",
            "slurm_extra_args": ["--nodes", "3", "--ntasks-per-node", "40"],
        },
        tasks_per_item=16,
    ) == 7
    assert hlp._remote_sweep_parallelism(
        {
            "remote_mpi_exec": "srun --mpi=pmix_v4",
            "sweep_parallelism": 4,
            "optimizer_total_tasks": 120,
        },
        tasks_per_item=15,
    ) == 4
    long_joint_sweep_label = hlp._safe_sweep_path_label(
        {
            "kar_mt_gmax": [1.0],
            "kar_gc_gmax": [1.0],
            "gaba_gmax": [1.0],
            "ampa_nmda_gmax": [1.0],
            "gap_tc": [1.0],
            "gap_mc": [1.0],
            "tc_input_weight": [1.0],
            "mc_input_weight": [1.0],
            "optimizer_candidate_id": ["C00000"],
            "optimizer_method": ["latin_hypercube"],
            "optimizer_stage": ["wide_seed"],
            "optimizer_batch_name": ["batch_0000"],
            "ketamine_block": [0.0],
            "optimizer_condition": ["ketamine"],
            "optimizer_pair_id": ["C00000"],
        }
    )
    assert len(long_joint_sweep_label) <= 64
    print("Remote sweep manifest upload path: OK")

    # --- Remote sweep should resolve the real timestamped payload dir after completion ---
    sweep_driver_spec = importlib.util.spec_from_file_location(
        "remote_sweep_driver_test",
        hlp.REPO_ROOT / "tools" / "remote" / "remote_sweep_driver.py",
    )
    assert sweep_driver_spec is not None and sweep_driver_spec.loader is not None
    remote_sweep_driver = importlib.util.module_from_spec(sweep_driver_spec)
    sweep_driver_spec.loader.exec_module(remote_sweep_driver)

    assert remote_sweep_driver.add_srun_parallel_step_flags(
        ["srun", "--mpi=pmix_v4", "--cpu-bind=none", "-n", "15", "nrniv", "-mpi"]
    ) == [
        "srun",
        "--exclusive",
        "--exact",
        "--mpi=pmix_v4",
        "--cpu-bind=none",
        "-n",
        "15",
        "nrniv",
        "-mpi",
    ]
    assert remote_sweep_driver.add_srun_parallel_step_flags(
        ["srun", "--exclusive", "--exact", "-n", "15", "nrniv"]
    ) == ["srun", "--exclusive", "--exact", "-n", "15", "nrniv"]
    print("Remote sweep parallel srun flags: OK")

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
        "sweep_manifest.submit.json",
        "mpi_preflight.log",
        "bootstrap.log",
        "stdout.txt",
        "stderr.txt",
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
    raw_only_sweep_item = tmp / "raw-only-sweep-item"
    raw_only_sweep_item.mkdir()
    (raw_only_sweep_item / "summary.json").write_text("{}")
    (raw_only_sweep_item / "soma_vs.pkl").write_bytes(b"payload")
    assert not hlp._local_sweep_item_sync_complete(raw_only_sweep_item)
    compact_sweep_item = tmp / "compact-sweep-item"
    compact_sweep_item.mkdir()
    (compact_sweep_item / "summary.json").write_text("{}")
    (compact_sweep_item / SOMA_SPIKES_FILENAME_NPZ).write_bytes(b"payload")
    assert hlp._local_sweep_item_sync_complete(compact_sweep_item)
    partial_sweep_dir = tmp / "partial-sweep"
    partial_sweep_dir.mkdir(parents=True, exist_ok=True)
    (partial_sweep_dir / "sim_progress.json").write_text(
        json.dumps(
            {
                "finished_items": [
                    {
                        "index": 0,
                        "label": "item_000",
                        "ok": True,
                        "result_dir": "/remote/item_000_20260525_120000",
                    }
                ],
                "pending_labels": ["item_001"],
                "running_items": [],
            }
        )
    )
    recovered_summary = hlp._recover_local_sweep_summary(
        partial_sweep_dir,
        sweep_label="partial-sweep",
        total_items=2,
    )
    assert recovered_summary["partial"] is True
    assert recovered_summary["completed_items"][0]["label"] == "item_000"
    assert (partial_sweep_dir / "summary.json").exists()
    scan_sweep_dir = tmp / "scan-recovered-sweep"
    scan_item_runs = scan_sweep_dir / "item_runs"
    scan_payload_dir = scan_item_runs / "item_000_20260525_120000"
    scan_payload_dir.mkdir(parents=True, exist_ok=True)
    for name in ("input_times.pkl", "lfp.pkl", "gc_output_events.pkl", "soma_vs.pkl"):
        (scan_payload_dir / name).write_bytes(b"payload")
    (scan_sweep_dir / "sweep_manifest.submit.json").write_text(
        json.dumps([{"index": 0, "label": "item_000", "value": 1.0}])
    )
    scanned_summary = hlp._recover_local_sweep_summary(
        scan_sweep_dir,
        sweep_label="scan-recovered-sweep",
        total_items=1,
    )
    assert scanned_summary["completed_items"][0]["result_dir"] == str(scan_payload_dir)
    assert hlp._resolve_local_sweep_item_dir(scan_item_runs, "item_000") == scan_payload_dir
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
    helper_sources = hlp._remote_helper_sources()
    assert helper_sources["slurm_common.py"] == hlp.REPO_ROOT / "tools" / "remote" / "slurm_common.py"
    assert helper_sources["neuroinfra/remote_script_common.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_common.py"
    assert helper_sources["neuroinfra/remote_script_submit.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_submit.py"
    assert helper_sources["neuroinfra/remote_script_polling.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_polling.py"
    assert helper_sources["neuroinfra/remote_script_allocations.py"] == hlp.REPO_ROOT / "neuroinfra" / "remote_script_allocations.py"
    helper_manifest = helper_bundle_manifest(
        hlp._remote_helper_bundle_entries(),
        signature=hlp._remote_helper_signature(),
    )
    assert helper_manifest["files"] == sorted(helper_sources.keys())
    assert helper_manifest["parent_dirs"] == ["neuroinfra"]
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

    submit_sol_run_source = Path(submit_sol_run.__file__).read_text()
    remote_submit_source = (hlp.REPO_ROOT / "neuroinfra" / "remote_script_submit.py").read_text()
    assert "remote_script_submit" in submit_sol_run_source
    assert '--nodes=1 --ntasks="$step_ntasks"' in remote_submit_source
    assert "--step-ntasks" in remote_submit_source
    print("Reusable allocation wrapper step launch: OK")

    allocation_cfg = build_run_config(
        runner_backend="sol_slurm",
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        remote_results_root="/remote/OlfactoryBulb/results/notebook_runs",
        remote_conda_activate_cmd="source activate OBGPU",
        remote_git_ref="abcdef1234567890",
        slurm_allocation_job_id="12345",
        slurm_step_ntasks=15,
        nranks=15,
    )
    single_run_submit = hlp._build_remote_submit_command(
        allocation_cfg,
        label="single_run",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        benchmark_command=["nrniv", "-mpi", "-python", "bench.py"],
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        remote_git_ref="abcdef1234567890",
        remote_helper_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test"),
    )
    sweep_driver_submit = hlp._build_remote_submit_command(
        allocation_cfg,
        label="sweep_driver",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/sweeps"),
        benchmark_command=["python3", "/remote/OlfactoryBulb/tools/remote/remote_sweep_driver.py"],
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        remote_git_ref="abcdef1234567890",
        step_ntasks=1,
        remote_helper_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test"),
    )
    assert "--step-ntasks 15" in single_run_submit
    assert "--step-ntasks 1" in sweep_driver_submit
    print("Remote sweep wrapper uses one allocation task: OK")

    # --- Bulky run overrides should stay out of process argv ---
    verbose_cfg = deepcopy(remote_cfg)
    verbose_cfg["input_odors"] = {
        step: {"name": "Apple", "rel_conc": 0.05}
        for step in range(0, 2000, 200)
    }
    verbose_overrides, verbose_input_spec = hlp._benchmark_param_overrides_payload(verbose_cfg)
    verbose_overrides_path = PurePosixPath(
        "/remote/OlfactoryBulb/results/notebook_runs/.obgpu-wrapper/test_label/overrides.json"
    )
    compact_command = hlp.build_run_command(
        verbose_cfg,
        "test_label",
        repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        results_base=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        overrides_file=verbose_overrides_path,
        param_overrides=verbose_overrides,
        input_spec_file=verbose_input_spec,
    )
    compact_command_text = " ".join(compact_command)
    assert "--overrides-file" in compact_command
    assert "--overrides-json" not in compact_command
    assert "Apple" in json.dumps(verbose_overrides)
    assert "Apple" not in compact_command_text
    compact_submit = hlp._build_remote_submit_command(
        verbose_cfg,
        label="test_label",
        remote_repo_root=PurePosixPath("/remote/OlfactoryBulb"),
        remote_results_root=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs"),
        benchmark_command=compact_command,
        remote_mpi_exec="srun --mpi=pmix_v4 --cpu-bind=none",
        remote_git_ref="abcdef1234567890",
        remote_helper_dir=PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/.obgpu-helper-cache/test"),
    )
    assert "Apple" not in compact_submit
    print("Benchmark overrides sidecar keeps argv compact: OK")

    remote_builder_cfg = hlp.build_slurm_remote_config(
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
    )
    assert remote_builder_cfg["sweep_sync_live"] is False
    assert remote_builder_cfg["sweep_sync_soma_vs"] is False
    assert remote_builder_cfg["sweep_sync_voltage_summary"] is False
    assert remote_builder_cfg["sweep_live_sync_max_items_per_poll"] == 8
    assert remote_builder_cfg["ssh_transport"] == "paramiko"
    assert "ssh_multiplex" not in remote_builder_cfg
    assert "rsync_options" not in remote_builder_cfg
    remote_builder_live_cfg = hlp.build_slurm_remote_config(
        remote_host="user@host",
        remote_repo_root="/remote/OlfactoryBulb",
        sweep_sync_live=True,
        sweep_sync_voltage_summary=True,
        sweep_live_sync_max_items_per_poll=2,
    )
    assert remote_builder_live_cfg["sweep_sync_live"] is True
    assert remote_builder_live_cfg["sweep_sync_voltage_summary"] is True
    assert remote_builder_live_cfg["sweep_live_sync_max_items_per_poll"] == 2
    assert hlp._should_sync_remote_sweep_finished_items({}, pending_count=4, running_count=4) is False
    assert hlp._should_sync_remote_sweep_finished_items({}, pending_count=0, running_count=2) is True
    assert hlp._should_sync_remote_sweep_finished_items({}, pending_count=0, running_count=0) is False
    assert hlp._should_sync_remote_sweep_finished_items(
        {"sweep_sync_live": True},
        pending_count=4,
        running_count=4,
    ) is True
    print("Remote sweep builder defaults favor robust final sync: OK")

    if hlp.paramiko is not None:
        assert hlp._remote_transport({"remote_host": "user@host", "ssh_options": []}) == "paramiko"
        assert hlp._remote_transport({"remote_host": "user@host", "ssh_transport": "auto"}) == "paramiko"
        try:
            hlp._remote_transport({"remote_host": "user@host", "ssh_transport": "openssh"})
            raise AssertionError("openssh transport should not be accepted")
        except ValueError:
            pass
    print("Remote transport is Paramiko-only: OK")

    assert slurm_common.shell_join(["python", "a b.py"]) == "python 'a b.py'"
    assert slurm_common.path_is_within("/repo/results/run", "/repo/results")
    assert not slurm_common.path_is_within("/repo/results-old/run", "/repo/results")
    assert slurm_common.normalize_sbatch_args(
        ["--qos", "general", "--constraint=cascadelake", "--exclusive"]
    ) == ["--qos general", "--constraint=cascadelake", "--exclusive"]
    directives = slurm_common.slurm_directives(
        SimpleNamespace(
            partition=None,
            account="grp_scrook",
            time="00:10:00",
            gpus=1,
            cpus_per_task=None,
            mem="24G",
            sbatch_arg=["--qos", "general", "--ntasks=64"],
        ),
        "remote_test",
    )
    assert "#SBATCH --partition=" not in "\n".join(directives)
    assert "#SBATCH --account=grp_scrook" in directives
    assert "#SBATCH --qos general" in directives
    assert "#SBATCH --ntasks=64" in directives
    assert slurm_common.requested_mpi_rank_count(["srun", "--mpi=pmix", "-n", "16"]) == 16
    assert slurm_common.requested_mpi_rank_count(["mpiexec", "-np8"]) == 8
    assert slurm_common.requested_mpi_rank_count(["srun", "--ntasks=32"]) == 32
    assert slurm_common.requested_mpi_rank_count(["python", "script.py"]) is None
    print("Remote Slurm shared helpers: OK")

    # --- Successful fast remote sync should only request essential result artifacts ---
    fast_files_default = hlp._remote_fast_sync_files()
    assert fast_files_default == (
        "summary.json",
        "input_times.pkl",
        "lfp.pkl",
        "gc_output_events.pkl",
        SOMA_SPIKES_FILENAME_NPZ,
        VOLTAGE_SUMMARY_FILENAME_NPZ,
    )
    fast_files_minimal = hlp._remote_fast_sync_files(
        {"enable_lfp": False, "record_gc_output_events": False}
    )
    assert fast_files_minimal == ("summary.json", "input_times.pkl")
    print("Remote fast sync file set: OK")

    # --- Remote sweep sync should stay compact unless bulk trace payloads are explicitly requested ---
    sweep_metadata_files = hlp._remote_sweep_metadata_files()
    assert "summary.json" in sweep_metadata_files
    assert "sim_progress.json" in sweep_metadata_files
    assert "bootstrap.log" in sweep_metadata_files
    assert "stdout.txt" in sweep_metadata_files
    assert "stderr.txt" in sweep_metadata_files
    sweep_files_default = hlp._remote_sweep_item_sync_files()
    assert SOMA_SPIKES_FILENAME_NPZ in sweep_files_default
    assert VOLTAGE_SUMMARY_FILENAME_NPZ not in sweep_files_default
    assert "soma_vs.pkl" not in sweep_files_default
    assert SOMA_TRACE_FILENAME_NPZ not in sweep_files_default
    sweep_files_voltage = hlp._remote_sweep_item_sync_files({"sweep_sync_voltage_summary": True})
    assert VOLTAGE_SUMMARY_FILENAME_NPZ in sweep_files_voltage
    sweep_files_mean_signal = hlp._remote_sweep_item_sync_files(
        {"spectrogram_signal": "mean_MC_voltage"}
    )
    assert VOLTAGE_SUMMARY_FILENAME_NPZ in sweep_files_mean_signal
    sweep_files_raw = hlp._remote_sweep_item_sync_files({"sweep_sync_soma_vs": True})
    assert "soma_vs.pkl" in sweep_files_raw
    assert SOMA_TRACE_FILENAME_NPZ in sweep_files_raw
    print("Remote sweep sync file set stays compact by default: OK")

    # --- Fast result sync should fall back to full sync when selected files are not visible ---
    original_result_sync_hooks = hlp._remote_result_sync_hooks
    try:
        resilient_dir = tmp / "resilient-sync"
        resilient_dir.mkdir()
        sync_calls = []

        def _fake_sftp_copy_files(_sftp, remote_dir, local_dir, file_names):
            sync_calls.append((PurePosixPath(remote_dir), ("summary.json",), tuple(file_names)))
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        def _fake_sftp_copy_tree(_sftp, remote_dir, local_dir):
            sync_calls.append((PurePosixPath(remote_dir), ("summary.json",), None))
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "summary.json").write_text("{}")

        hlp._remote_result_sync_hooks = lambda: RemoteResultSyncHooks(
            remote_transport_fn=lambda _config: "paramiko",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            build_remote_archive_probe_command_fn=lambda remote_result_dir: f"probe::{remote_result_dir.as_posix()}",
            probe_selected_sync_files_fn=lambda *_args, **_kwargs: ("gzip", 0, ("summary.json",)),
            build_remote_selected_stream_archive_command_fn=lambda remote_result_dir, include_files, compressor: (
                f"selected::{remote_result_dir.as_posix()}::{','.join(include_files)}::{compressor}"
            ),
            stream_archive_to_local_dir_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("non-compressed resilient sync should not use the archive stream path")
            ),
            get_paramiko_sftp_fn=lambda _config: object(),
            close_paramiko_sftp_fn=lambda _config: None,
            sftp_copy_files_fn=_fake_sftp_copy_files,
            sftp_copy_tree_fn=_fake_sftp_copy_tree,
            cached_transport_fn=lambda _config: object(),
            transport_is_usable_fn=lambda _transport: True,
            preserve_reauth_blocked_fn=lambda _config: False,
            drop_paramiko_connection_fn=lambda _config: None,
            midrun_reauth_error_fn=lambda _config: "midrun reauth blocked",
            progress_write=lambda _message: None,
            missing_local_sync_artifacts_fn=lambda local_result_dir, expected_files: hlp._missing_local_sync_artifacts(
                local_result_dir,
                expected_files=expected_files,
            ),
            local_sync_artifact_is_usable_fn=hlp._local_sync_artifact_is_usable,
            sleep_fn=lambda _seconds: None,
        )
        completed = hlp._sync_remote_result_dir_resilient(
            {**remote_cfg, "remote_sync_compress": False},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=resilient_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
            retry_delay_s=0,
        )
        assert completed.returncode == 0
        assert (resilient_dir / "summary.json").exists()
        assert sync_calls == [
            (PurePosixPath("/remote/result"), ("summary.json",), ("summary.json",)),
            (PurePosixPath("/remote/result"), ("summary.json",), ("summary.json",)),
            (PurePosixPath("/remote/result"), ("summary.json",), None),
        ]
        print("Resilient remote sync selected-to-full fallback: OK")
    finally:
        hlp._remote_result_sync_hooks = original_result_sync_hooks

    # --- Failed payload sync should still pull wrapper diagnostics before returning failure ---
    original_result_sync_hooks = hlp._remote_result_sync_hooks
    try:
        diagnostic_dir = tmp / "resilient-sync-diagnostics"
        diagnostic_dir.mkdir()
        sync_calls = []

        def _fake_sftp_copy_files(_sftp, remote_dir, local_dir, file_names):
            sync_calls.append((PurePosixPath(remote_dir), ("summary.json",), tuple(file_names)))
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        def _fake_sftp_copy_tree(_sftp, remote_dir, local_dir):
            sync_calls.append((PurePosixPath(remote_dir), None, None))
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            if PurePosixPath(remote_dir) == PurePosixPath("/remote/wrapper"):
                (local_dir / "bootstrap.log").write_text("wrapper diagnostics")

        hlp._remote_result_sync_hooks = lambda: RemoteResultSyncHooks(
            remote_transport_fn=lambda _config: "paramiko",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            build_remote_archive_probe_command_fn=lambda remote_result_dir: f"probe::{remote_result_dir.as_posix()}",
            probe_selected_sync_files_fn=lambda *_args, **_kwargs: ("gzip", 0, ("summary.json",)),
            build_remote_selected_stream_archive_command_fn=lambda remote_result_dir, include_files, compressor: (
                f"selected::{remote_result_dir.as_posix()}::{','.join(include_files)}::{compressor}"
            ),
            stream_archive_to_local_dir_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("non-compressed resilient sync should not use the archive stream path")
            ),
            get_paramiko_sftp_fn=lambda _config: object(),
            close_paramiko_sftp_fn=lambda _config: None,
            sftp_copy_files_fn=_fake_sftp_copy_files,
            sftp_copy_tree_fn=_fake_sftp_copy_tree,
            cached_transport_fn=lambda _config: object(),
            transport_is_usable_fn=lambda _transport: True,
            preserve_reauth_blocked_fn=lambda _config: False,
            drop_paramiko_connection_fn=lambda _config: None,
            midrun_reauth_error_fn=lambda _config: "midrun reauth blocked",
            progress_write=lambda _message: None,
            missing_local_sync_artifacts_fn=lambda local_result_dir, expected_files: hlp._missing_local_sync_artifacts(
                local_result_dir,
                expected_files=expected_files,
            ),
            local_sync_artifact_is_usable_fn=hlp._local_sync_artifact_is_usable,
            sleep_fn=lambda _seconds: None,
        )
        completed = hlp._sync_remote_result_dir_resilient(
            {**remote_cfg, "remote_sync_compress": False},
            remote_result_dir=PurePosixPath("/remote/result"),
            local_result_dir=diagnostic_dir,
            expected_files=("summary.json",),
            include_files=("summary.json",),
            wrapper_dir=PurePosixPath("/remote/wrapper"),
            retry_delay_s=0,
        )
        assert completed.returncode == 1
        assert (diagnostic_dir / "bootstrap.log").read_text() == "wrapper diagnostics"
        assert "[wrapper diagnostic sync]" in completed.stderr
        assert sync_calls[-1] == (PurePosixPath("/remote/wrapper"), None, None)
        print("Resilient remote sync wrapper diagnostics: OK")
    finally:
        hlp._remote_result_sync_hooks = original_result_sync_hooks

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

    # --- Heartbeat refresh should keep a bounded timeout even when generic SSH timeouts are disabled ---
    original_run_ssh_shell = hlp._run_ssh_shell
    try:
        heartbeat_calls = []

        def _fake_run_ssh_shell(_config, command, check=False, timeout_s=None):
            heartbeat_calls.append((command, check, timeout_s))
            return subprocess.CompletedProcess(
                args=["ssh", "bash", "-lc", command],
                returncode=0,
                stdout="",
                stderr="",
            )

        hlp._run_ssh_shell = _fake_run_ssh_shell
        heartbeat_cfg = dict(remote_cfg)
        heartbeat_cfg["remote_ssh_command_timeout_s"] = None
        heartbeat_cfg["remote_poll_command_timeout_s"] = None
        assert hlp._refresh_remote_heartbeat(
            heartbeat_cfg,
            "/remote/OlfactoryBulb/results/notebook_runs/.heartbeat/test.touch",
        )
        assert heartbeat_calls
        assert heartbeat_calls[-1][2] == 60.0
        print("Remote heartbeat refresh timeout fallback: OK")
    finally:
        hlp._run_ssh_shell = original_run_ssh_shell

    # --- Stale allocation cleanup should skip manual allocations and reuse its session cache ---
    original_runtime_context = hlp._remote_allocation_runtime_context
    try:
        hlp._LIVE_REMOTE_STALE_CLEANUPS.clear()
        cleanup_calls = []

        class _FakeAllocationRuntime:
            _cache = {}

            def __init__(self, cfg):
                self.cfg = cfg

            def maybe_cleanup_stale_allocations(self, *, remote_helper_dir=None):
                if self.cfg.get("slurm_allocation_job_id") not in (None, ""):
                    return []
                cache_key = (
                    self.cfg.get("remote_host"),
                    self.cfg.get("remote_results_root"),
                )
                if cache_key in self._cache:
                    return list(self._cache[cache_key])
                cleanup_calls.append(remote_helper_dir)
                actions = [{"job_id": "1", "action": "cancel_requested"}]
                self._cache[cache_key] = list(actions)
                return actions

        hlp._remote_allocation_runtime_context = lambda _config: _FakeAllocationRuntime(_config)
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
        hlp._remote_allocation_runtime_context = original_runtime_context
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

    # --- Deferred remote soma traces from old runs should sync during load by default ---
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
        print("Deferred remote soma eager sync: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir

    # --- Deferred soma traces should bypass SFTP with direct file streaming when selected sync fails ---
    original_sync_remote_result_dir = hlp._sync_remote_result_dir
    original_deferred_artifact_sync_hooks = hlp._deferred_remote_artifact_sync_hooks
    try:
        deferred_direct_dir = tmp / "deferred-direct-result"
        deferred_direct_dir.mkdir(parents=True, exist_ok=True)
        (deferred_direct_dir / "summary.json").write_text("{}")
        (deferred_direct_dir / "run_info.json").write_text(
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
        direct_calls = []

        def _fake_sync_remote_result_dir(_config, *, remote_result_dir, local_result_dir, expected_files=None, include_files=None):
            sync_calls.append((remote_result_dir, expected_files, include_files))
            return subprocess.CompletedProcess(args=["sync"], returncode=1, stdout="", stderr="selected sync failed")

        def _fake_direct_stream(_config, *, remote_file_path, local_path, expected_bytes):
            direct_calls.append((remote_file_path.parent, remote_file_path.name))
            local_path = Path(local_path)
            with open(local_path, "wb") as handle:
                import pickle
                pickle.dump([("MC0", [0.0, 0.1], [-65.0, -64.0])], handle)
            return subprocess.CompletedProcess(args=["direct"], returncode=0, stdout="", stderr="")

        hlp._sync_remote_result_dir = _fake_sync_remote_result_dir
        hlp._deferred_remote_artifact_sync_hooks = lambda: DeferredArtifactSyncHooks(
            local_sync_artifact_is_usable_fn=hlp._local_sync_artifact_is_usable,
            sync_remote_result_dir_fn=hlp._sync_remote_result_dir,
            progress_write=lambda _message: None,
            format_bytes_fn=lambda num_bytes: f"{num_bytes} B",
            direct_stream_supported_fn=lambda filename: filename == "soma_vs.pkl",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                args=["probe"],
                returncode=0,
                stdout="67\n",
                stderr="",
            ),
            stream_file_to_local_path_fn=_fake_direct_stream,
            perf_counter_fn=lambda: 0.0,
        )
        result = hlp.load_result(deferred_direct_dir)
        assert result["soma_vs"][0][0] == "MC0"
        assert sync_calls == [
            (
                PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label"),
                ("soma_vs.pkl",),
                ("soma_vs.pkl",),
            )
        ]
        assert direct_calls == [
            (PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label"), "soma_vs.pkl")
        ]
        print("Deferred soma direct-stream fallback: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir
        hlp._deferred_remote_artifact_sync_hooks = original_deferred_artifact_sync_hooks

    # --- Deferred soma traces should fall back to full-dir sync if selected and direct sync fail ---
    original_sync_remote_result_dir = hlp._sync_remote_result_dir
    original_deferred_artifact_sync_hooks = hlp._deferred_remote_artifact_sync_hooks
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
        direct_calls = []

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

        def _fake_direct_stream(_config, *, remote_file_path, local_path, expected_bytes):
            direct_calls.append((remote_file_path.parent, remote_file_path.name))
            return subprocess.CompletedProcess(args=["direct"], returncode=1, stdout="", stderr="direct sync failed")

        hlp._sync_remote_result_dir = _fake_sync_remote_result_dir
        hlp._deferred_remote_artifact_sync_hooks = lambda: DeferredArtifactSyncHooks(
            local_sync_artifact_is_usable_fn=hlp._local_sync_artifact_is_usable,
            sync_remote_result_dir_fn=hlp._sync_remote_result_dir,
            progress_write=lambda _message: None,
            format_bytes_fn=lambda num_bytes: f"{num_bytes} B",
            direct_stream_supported_fn=lambda filename: filename == "soma_vs.pkl",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                args=["probe"],
                returncode=0,
                stdout="67\n",
                stderr="",
            ),
            stream_file_to_local_path_fn=_fake_direct_stream,
            perf_counter_fn=lambda: 0.0,
        )
        result = hlp.load_result(deferred_fallback_dir)
        assert result["soma_vs"][0][0] == "MC0"
        assert sync_calls[0][2] == ("soma_vs.pkl",)
        assert sync_calls[1][2] is None
        assert direct_calls == [
            (PurePosixPath("/remote/OlfactoryBulb/results/notebook_runs/test_label"), "soma_vs.pkl")
        ]
        print("Deferred soma fallback to full-dir sync: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir
        hlp._deferred_remote_artifact_sync_hooks = original_deferred_artifact_sync_hooks

    # --- Deferred soma failures should retain all retry diagnostics ---
    original_sync_remote_result_dir = hlp._sync_remote_result_dir
    original_deferred_artifact_sync_hooks = hlp._deferred_remote_artifact_sync_hooks
    try:
        deferred_error_dir = tmp / "deferred-error-result"
        deferred_error_dir.mkdir(parents=True, exist_ok=True)
        (deferred_error_dir / "summary.json").write_text("{}")
        (deferred_error_dir / "run_info.json").write_text(
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
            if include_files == ("soma_vs.pkl",):
                return subprocess.CompletedProcess(args=["sync"], returncode=1, stdout="", stderr="selected failed")
            return subprocess.CompletedProcess(args=["sync"], returncode=1, stdout="", stderr="Failure")

        def _fake_direct_stream(_config, *, remote_file_path, local_path, expected_bytes):
            return subprocess.CompletedProcess(args=["direct"], returncode=1, stdout="", stderr="direct failed")

        hlp._sync_remote_result_dir = _fake_sync_remote_result_dir
        hlp._deferred_remote_artifact_sync_hooks = lambda: DeferredArtifactSyncHooks(
            local_sync_artifact_is_usable_fn=hlp._local_sync_artifact_is_usable,
            sync_remote_result_dir_fn=hlp._sync_remote_result_dir,
            progress_write=lambda _message: None,
            format_bytes_fn=lambda num_bytes: f"{num_bytes} B",
            direct_stream_supported_fn=lambda filename: filename == "soma_vs.pkl",
            run_paramiko_shell_fn=lambda _config, _command: subprocess.CompletedProcess(
                args=["probe"],
                returncode=0,
                stdout="67\n",
                stderr="",
            ),
            stream_file_to_local_path_fn=_fake_direct_stream,
            perf_counter_fn=lambda: 0.0,
        )
        try:
            _ = hlp.load_result(deferred_error_dir)
            raise AssertionError("Expected deferred soma sync failure")
        except RuntimeError as exc:
            text = str(exc)
            assert "[selected-file sync]" in text
            assert "[direct file stream]" in text
            assert "[full result-dir sync]" in text
            assert "Failure" in text
        print("Deferred soma failure diagnostics: OK")
    finally:
        hlp._sync_remote_result_dir = original_sync_remote_result_dir
        hlp._deferred_remote_artifact_sync_hooks = original_deferred_artifact_sync_hooks

    # --- LazyResult should keep loaders available after transient failures ---
    lazy_attempts = []

    def _flaky_soma_loader():
        lazy_attempts.append("attempt")
        if len(lazy_attempts) == 1:
            raise RuntimeError("transient sync failure")
        return [("MC0", [0.0], [-65.0])]

    lazy_result = hlp.LazyResult({"soma_vs": []}, lazy_loaders={"soma_vs": _flaky_soma_loader})
    try:
        _ = lazy_result["soma_vs"]
        raise AssertionError("Expected first lazy load attempt to fail")
    except RuntimeError:
        pass
    assert "soma_vs" in lazy_result._lazy_loaders
    assert lazy_result["soma_vs"][0][0] == "MC0"
    assert "soma_vs" not in lazy_result._lazy_loaders
    print("LazyResult preserves failed loader for retry: OK")

    # --- Result overview can still inspect old deferred runs when lazy loading is explicitly requested ---
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
        result = hlp.load_result(overview_result_dir, lazy_soma_vs=True)
        info = hlp.result_overview(result)
        assert info["n_inputs"] == 612
        assert info["n_soma_traces"] == 193
        assert info["n_lfp_samples"] == 36000
        print("Result overview supports explicit legacy deferred soma sync: OK")
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

    # --- load_result should read compressed NPZ soma traces through the normal notebook path ---
    npz_result_dir = tmp / "npz-result-load"
    npz_result_dir.mkdir(parents=True, exist_ok=True)
    (npz_result_dir / "summary.json").write_text(json.dumps({"files": {"soma_vs.npz": {"items": 2}}}))
    save_soma_trace_artifact(
        [
            ("MC0", [0.0, 0.1], [-65.0, -64.5]),
            ("TC0", [0.0, 0.1], [-63.0, -62.5]),
        ],
        npz_result_dir,
        trace_format="npz",
        trace_dtype="float32",
    )
    npz_loaded = hlp.load_result(npz_result_dir)
    assert len(npz_loaded["soma_vs"]) == 2
    assert npz_loaded["soma_vs"][0][1].dtype == np.float32
    assert npz_loaded["soma_vs"][0][2].dtype == np.float32
    assert "MC0" in hlp.list_available_named_signals(npz_loaded, include_soma_labels=True)
    signal_t, signal_v = hlp.get_named_signal(npz_loaded, signal="MC0")
    assert len(signal_t) == len(signal_v) == 2
    print("load_result NPZ soma trace path: OK")

    # --- Recovered local runs may have no remote dict in run_info ---
    recovered_result_dir = tmp / "recovered-run"
    recovered_result_dir.mkdir(parents=True, exist_ok=True)
    (recovered_result_dir / "run_info.json").write_text(json.dumps({"remote": None}))
    (recovered_result_dir / "summary.json").write_text("{}")
    with open(recovered_result_dir / "input_times.pkl", "wb") as handle:
        pickle.dump([], handle)
    recovered_loaded = hlp.load_result(recovered_result_dir)
    assert recovered_loaded["input_times"] == []
    print("load_result tolerates empty remote run_info: OK")

    # --- Sweep metadata should preserve every planned slot and tolerate bad items ---
    robust_sweep_dir = tmp / "robust-sweep"
    robust_good_dir = tmp / "robust-good-run"
    robust_bad_dir = tmp / "robust-bad-run"
    robust_good_dir.mkdir(parents=True, exist_ok=True)
    robust_bad_dir.mkdir(parents=True, exist_ok=True)
    (robust_good_dir / "summary.json").write_text("{}")
    with open(robust_good_dir / "input_times.pkl", "wb") as handle:
        pickle.dump([1], handle)
    (robust_bad_dir / "summary.json").write_text("{}")
    (robust_bad_dir / "input_times.pkl").write_text("not a pickle")
    robust_sweep = {
        "path": "gaba_gmax",
        "values": [0.0, 0.1, 0.2],
        "paramset": "GammaSignature",
        "partial": True,
        "missing_labels": ["item_002"],
        "items": [
            {
                "label": "item_000",
                "value": 0.0,
                "run": SimpleNamespace(result_dir=robust_good_dir),
                "result": {"result_dir": robust_good_dir},
                "status": {"ok": True},
            },
            {
                "label": "item_001",
                "value": 0.1,
                "run": SimpleNamespace(result_dir=robust_bad_dir),
                "result": {"result_dir": robust_bad_dir},
                "status": {"ok": True},
            },
            {"label": "item_002", "value": 0.2, "run": None, "result": None, "status": {"ok": False}},
        ],
    }
    hlp._write_sweep_info(robust_sweep, sweep_dir=robust_sweep_dir, timestamp="20260525_120000")
    robust_loaded_sweep = hlp.load_sweep(robust_sweep_dir)
    assert len(robust_loaded_sweep["items"]) == 3
    assert robust_loaded_sweep["items"][0]["result"] is not None
    assert robust_loaded_sweep["items"][1]["result"] is None
    assert "load_error" in robust_loaded_sweep["items"][1]
    assert robust_loaded_sweep["items"][2]["result"] is None
    assert robust_loaded_sweep["partial"] is True
    assert robust_loaded_sweep["missing_labels"] == ["item_002"]
    print("load_sweep preserves partial sweep slots: OK")

    # --- Sweep animations should render placeholders for partial/bad items ---
    partial_anim_dir = tmp / "partial-animation-sweep"
    partial_anim_dir.mkdir(parents=True, exist_ok=True)
    partial_anim_sweep = {
        "path": "gaba_gmax",
        "sweep_dir": partial_anim_dir,
        "items": [
            {"label": "item_000", "value": 0.0, "result": {"y": 1.0}},
            {
                "label": "item_001",
                "value": 0.1,
                "result": None,
                "load_error": "missing payload",
                "status": {"ok": False},
            },
            {"label": "item_002", "value": 0.2, "result": {"raise": True}},
        ],
    }

    def _toy_sweep_plot(result):
        if result.get("raise"):
            raise RuntimeError("synthetic plot failure")
        fig, ax = hlp.plt.subplots(figsize=(2, 1.5))
        ax.plot([0, 1], [0, result["y"]])
        return fig

    partial_anim_artifacts = hlp.animate_sweep_plots(
        partial_anim_sweep,
        [
            hlp.make_sweep_plot_spec(
                _toy_sweep_plot,
                name="partial_placeholder_test",
                filename="partial_placeholder_test",
                figsize=(2, 1.5),
                interval=10,
                fps=1,
            )
        ],
    )
    assert len(partial_anim_artifacts) == 1
    partial_anim_path = next(iter(partial_anim_artifacts.values()))
    assert partial_anim_path.exists()
    assert partial_anim_path.stem.startswith("partial_placeholder_test__")
    placeholder_anim = hlp.animate_lfp_sweep(partial_anim_sweep, interval=10)
    placeholder_anim._draw_was_started = True
    hlp.plt.close("all")
    print("Sweep animations tolerate partial sweep slots: OK")

    # --- Sweep animation filenames should encode rendering settings ---
    naming_anim_sweep = {
        "path": "gaba_gmax",
        "sweep_dir": tmp / "naming-animation-sweep",
        "items": [{"label": "item_000", "value": 0.0, "result": {"y": 1.0}}],
    }
    naming_anim_sweep["sweep_dir"].mkdir(parents=True, exist_ok=True)

    def _scaled_sweep_plot(result, scale=1.0):
        fig, ax = hlp.plt.subplots(figsize=(2, 1.5))
        ax.plot([0, 1], [0, result["y"] * scale])
        return fig

    named_artifacts = hlp.animate_sweep_plots(
        naming_anim_sweep,
        [
            hlp.make_sweep_plot_spec(
                _scaled_sweep_plot,
                name="settings_sensitive",
                plot_kwargs={"scale": 1.0},
                interval=10,
                fps=1,
            ),
            hlp.make_sweep_plot_spec(
                _scaled_sweep_plot,
                name="settings_sensitive",
                plot_kwargs={"scale": 2.0},
                interval=10,
                fps=1,
            ),
        ],
    )
    named_paths = list(named_artifacts.values())
    assert len(named_artifacts) == 2
    assert len({path.name for path in named_paths}) == 2
    assert all("scale-" in path.stem for path in named_paths)
    assert all(path.exists() for path in named_paths)
    print("Sweep animation filenames encode settings: OK")

print("\nAll tests passed.")
