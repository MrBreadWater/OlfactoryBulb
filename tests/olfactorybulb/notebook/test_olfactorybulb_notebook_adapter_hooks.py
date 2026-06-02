from pathlib import Path

from olfactorybulb.notebook_adapter_hooks import (
    build_local_run_hook_builder_hooks,
    build_local_run_payload_hooks,
    build_notebook_presentation_hooks,
    build_notebook_result_hooks,
    build_notebook_run_dispatch_hooks,
    build_notebook_sweep_dispatch_hooks,
    build_notebook_sweep_hooks,
    build_notebook_workflow_adapter_hooks,
    build_remote_run_payload_hooks,
    build_remote_run_workflow_builder_hooks,
    build_remote_sweep_payload_hooks,
    build_remote_sweep_workflow_builder_hooks,
)
from olfactorybulb.notebook_dispatch import (
    NotebookRunDispatchAdapterHooks,
    NotebookSweepDispatchAdapterHooks,
)
from olfactorybulb.notebook_local_runs import (
    LocalRunPayloadHooks,
    NotebookLocalRunHookBuilderHooks,
)
from olfactorybulb.notebook_presentations import NotebookPresentationHooks
from olfactorybulb.notebook_remote_runs import (
    NotebookRemoteRunWorkflowBuilderHooks,
    RemoteRunPayloadHooks,
)
from olfactorybulb.notebook_remote_sweeps import (
    NotebookRemoteSweepWorkflowBuilderHooks,
    RemoteSweepPayloadHooks,
)
from olfactorybulb.notebook_results import NotebookResultHooks
from olfactorybulb.notebook_sweeps import NotebookSweepHooks
from olfactorybulb.notebook_workflows import NotebookWorkflowAdapterHooks


def _sentinel(name: str):
    def _fn(*args, **kwargs):
        return (name, args, kwargs)

    _fn.__name__ = name
    return _fn


path_value = Path("/tmp/obgpu-results")
callable_a = _sentinel("a")
callable_b = _sentinel("b")
callable_c = _sentinel("c")
callable_d = _sentinel("d")
callable_e = _sentinel("e")
callable_f = _sentinel("f")
callable_g = _sentinel("g")
callable_h = _sentinel("h")
callable_i = _sentinel("i")
callable_j = _sentinel("j")
callable_k = _sentinel("k")
callable_l = _sentinel("l")
callable_m = _sentinel("m")
callable_n = _sentinel("n")
callable_o = _sentinel("o")
callable_p = _sentinel("p")
callable_q = _sentinel("q")
callable_r = _sentinel("r")
callable_s = _sentinel("s")
callable_t = _sentinel("t")
callable_u = _sentinel("u")
callable_v = _sentinel("v")
callable_w = _sentinel("w")
callable_x = _sentinel("x")
callable_y = _sentinel("y")
callable_z = _sentinel("z")

local_payload_hooks = build_local_run_payload_hooks(
    benchmark_param_overrides_payload_fn=callable_a,
    write_benchmark_overrides_file_fn=callable_b,
    build_run_command_fn=callable_c,
)
assert isinstance(local_payload_hooks, LocalRunPayloadHooks)
assert local_payload_hooks.build_run_command_fn is callable_c

local_builder_hooks = build_local_run_hook_builder_hooks(
    read_summary_fn=callable_d,
    write_run_info_fn=callable_e,
    build_param_overrides_fn=callable_f,
    run_record_factory_fn=callable_g,
)
assert isinstance(local_builder_hooks, NotebookLocalRunHookBuilderHooks)
assert local_builder_hooks.run_record_factory_fn is callable_g

workflow_hooks = build_notebook_workflow_adapter_hooks(
    load_run_record_fn=callable_h,
    load_result_fn=callable_i,
    run_simulation_fn=callable_j,
    merge_run_info_payload_fn=callable_k,
    save_sweep_fn=callable_l,
    sweep_item_runs_dir_fn=callable_m,
    sweep_dir_fn=callable_n,
)
assert isinstance(workflow_hooks, NotebookWorkflowAdapterHooks)
assert workflow_hooks.load_result_fn is callable_i

run_dispatch_hooks = build_notebook_run_dispatch_hooks(
    normalize_config_fn=callable_o,
    make_timestamp_fn=callable_p,
    make_label_fn=callable_q,
    build_local_run_payload_fn=callable_r,
    local_run_payload_hooks_fn=callable_s,
    build_local_run_hooks_fn=callable_t,
    local_run_hook_builder_hooks_fn=callable_u,
    execute_local_run_fn=callable_v,
    execute_remote_run_fn=callable_w,
    default_results_base=path_value,
)
assert isinstance(run_dispatch_hooks, NotebookRunDispatchAdapterHooks)
assert run_dispatch_hooks.default_results_base == path_value
assert run_dispatch_hooks.execute_remote_run_fn is callable_w

sweep_dispatch_hooks = build_notebook_sweep_dispatch_hooks(
    prepare_sweep_plan_fn=callable_x,
    uses_remote_batch_engine_fn=callable_y,
    build_local_sweep_hooks_fn=callable_z,
    notebook_workflow_adapter_hooks_fn=callable_a,
    execute_local_sweep_plan_fn=callable_b,
    execute_remote_sweep_fn=callable_c,
)
assert isinstance(sweep_dispatch_hooks, NotebookSweepDispatchAdapterHooks)
assert sweep_dispatch_hooks.execute_remote_sweep_fn is callable_c

result_view_hooks = object()
artifact_loading_hooks = object()
result_hooks = build_notebook_result_hooks(
    find_soma_trace_artifact_fn=callable_d,
    preferred_soma_trace_artifact_name_fn=callable_e,
    soma_trace_artifact_candidates_fn=callable_f,
    result_view_hooks=result_view_hooks,
    artifact_loading_hooks=artifact_loading_hooks,
)
assert isinstance(result_hooks, NotebookResultHooks)
assert result_hooks.result_view_hooks is result_view_hooks
assert result_hooks.artifact_loading_hooks is artifact_loading_hooks

sweep_hooks = build_notebook_sweep_hooks(
    sweeps_base=path_value / "sweeps",
    default_results_base=path_value,
    make_timestamp_fn=callable_g,
    safe_name_fn=callable_h,
    json_ready_fn=callable_i,
    resolve_git_head_fn=callable_j,
    load_result_fn=callable_k,
    save_sweep_fn=callable_l,
    load_sweep_fn=callable_m,
    list_sweeps_fn=callable_n,
    save_animation_fn=callable_o,
    save_sweep_animation_stream_fn=callable_p,
    animate_sweep_plots_fn=callable_q,
    build_sweep_plot_callable_fn=callable_r,
    normalize_sweep_plot_spec_fn=callable_s,
    is_deprecated_sweep_animation_spec_fn=callable_t,
    deprecated_plot_names=("old_plot",),
    progress_factory_fn=callable_u,
    progress_write_fn=callable_v,
    run_parameter_sweep_fn=callable_w,
    run_grid_sweep_fn=callable_x,
)
assert isinstance(sweep_hooks, NotebookSweepHooks)
assert sweep_hooks.deprecated_plot_names == ("old_plot",)

presentation_hooks = build_notebook_presentation_hooks(
    default_results_base=path_value,
    make_timestamp_fn=callable_y,
    safe_name_fn=callable_z,
    plt_module=object(),
    save_figure_fn=callable_a,
    plot_input_overview_fn=callable_b,
    plot_voltage_traces_fn=callable_c,
    plot_spike_raster_fn=callable_d,
    plot_gc_output_overview_fn=callable_e,
    plot_lfp_overview_fn=callable_f,
    plot_spectrogram_fn=callable_g,
    plot_wavelet_fn=callable_h,
    plot_wavelet_band_power_fn=callable_i,
    result_overview_fn=callable_j,
    build_run_config_fn=callable_k,
    resolve_effective_params_fn=callable_l,
    resolve_paramset_defaults_fn=callable_m,
    diff_values_fn=callable_n,
    extract_runtime_control_snapshot_fn=callable_o,
    print_diff_section_fn=callable_p,
    write_fn=callable_q,
)
assert isinstance(presentation_hooks, NotebookPresentationHooks)
assert presentation_hooks.plot_lfp_overview_fn is callable_f

remote_run_payload_hooks = build_remote_run_payload_hooks(
    build_run_command_fn=callable_r,
    build_remote_submit_command_fn=callable_s,
    require_remote_host_fn=callable_t,
    default_remote_mpi_exec_fn=callable_u,
)
assert isinstance(remote_run_payload_hooks, RemoteRunPayloadHooks)
assert remote_run_payload_hooks.require_remote_host_fn is callable_t

remote_run_workflow_hooks = build_remote_run_workflow_builder_hooks(
    remote_job_session_hooks_fn=callable_v,
    remote_job_submit_hooks_fn=callable_w,
    remote_run_monitor_hooks_fn=callable_x,
    remote_run_artifact_hooks_fn=callable_y,
    build_remote_run_payload_fn=callable_z,
    upload_remote_text_file_fn=callable_a,
    json_ready_fn=callable_b,
    remote_fast_sync_files_fn=callable_c,
    preferred_soma_trace_artifact_name_fn=callable_d,
    write_run_info_fn=callable_e,
    summarize_submit_response_fn=callable_f,
    summarize_status_fn=callable_g,
    timing_summary_text_fn=callable_h,
    build_return_value_fn=callable_i,
    shell_join_fn=callable_j,
    progress_write=callable_k,
    record_timing_fn=callable_l,
    perf_counter_fn=callable_m,
)
assert isinstance(remote_run_workflow_hooks, NotebookRemoteRunWorkflowBuilderHooks)
assert remote_run_workflow_hooks.build_return_value_fn is callable_i

remote_sweep_payload_hooks = build_remote_sweep_payload_hooks(
    json_ready_fn=callable_n,
    benchmark_param_overrides_payload_fn=callable_o,
    build_run_command_fn=callable_p,
    remote_sweep_parallelism_fn=callable_q,
    require_remote_host_fn=callable_r,
    default_remote_mpi_exec_fn=callable_s,
)
assert isinstance(remote_sweep_payload_hooks, RemoteSweepPayloadHooks)
assert remote_sweep_payload_hooks.remote_sweep_parallelism_fn is callable_q

remote_sweep_workflow_hooks = build_remote_sweep_workflow_builder_hooks(
    remote_job_session_hooks_fn=callable_t,
    remote_job_submit_hooks_fn=callable_u,
    remote_sweep_monitor_hooks_fn=callable_v,
    remote_sweep_artifact_hooks_fn=callable_w,
    build_remote_submit_command_fn=callable_x,
    upload_remote_text_file_fn=callable_y,
    refresh_remote_heartbeat_fn=callable_z,
    should_sync_remote_sweep_finished_items_fn=callable_a,
    sync_remote_result_dir_fn=callable_b,
    remote_sweep_item_sync_files_fn=callable_c,
    local_sync_artifact_is_usable_fn=callable_d,
    synthesize_partial_sync_summary_fn=callable_e,
    persist_sweep_fn=callable_f,
    merge_sweep_info_payload_fn=callable_g,
    summarize_status_fn=callable_h,
    timing_summary_text_fn=callable_i,
    write_run_info_fn=callable_j,
    load_run_record_fn=callable_k,
    load_result_fn=callable_l,
    resolve_local_sweep_item_dir_fn=callable_m,
    json_ready_fn=callable_n,
    read_json_if_present_fn=callable_o,
    progress_write=callable_p,
    record_timing_fn=callable_q,
    perf_counter_fn=callable_r,
    default_remote_mpi_exec_fn=callable_s,
)
assert isinstance(remote_sweep_workflow_hooks, NotebookRemoteSweepWorkflowBuilderHooks)
assert remote_sweep_workflow_hooks.persist_sweep_fn is callable_f

print("olfactorybulb notebook adapter hooks: OK")
