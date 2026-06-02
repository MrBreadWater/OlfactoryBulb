"""Concrete olfactory-bulb notebook hook-assembly helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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


def build_remote_sweep_payload_hooks(
    *,
    json_ready_fn: Callable[[Any], Any],
    benchmark_param_overrides_payload_fn: Callable[[dict[str, Any]], tuple[dict[str, Any], str | None]],
    build_run_command_fn: Callable[..., list[str]],
    remote_sweep_parallelism_fn: Callable[[dict[str, Any], int], int],
    require_remote_host_fn: Callable[[dict[str, Any]], str],
    default_remote_mpi_exec_fn: Callable[[], str],
) -> RemoteSweepPayloadHooks:
    """Build the concrete olfactory-bulb remote sweep payload hooks."""
    return RemoteSweepPayloadHooks(
        json_ready_fn=json_ready_fn,
        benchmark_param_overrides_payload_fn=benchmark_param_overrides_payload_fn,
        build_run_command_fn=build_run_command_fn,
        remote_sweep_parallelism_fn=remote_sweep_parallelism_fn,
        require_remote_host_fn=require_remote_host_fn,
        default_remote_mpi_exec_fn=default_remote_mpi_exec_fn,
    )


def build_remote_run_payload_hooks(
    *,
    build_run_command_fn: Callable[..., list[str]],
    build_remote_submit_command_fn: Callable[..., str],
    require_remote_host_fn: Callable[[dict[str, Any]], str],
    default_remote_mpi_exec_fn: Callable[[], str],
) -> RemoteRunPayloadHooks:
    """Build the concrete olfactory-bulb remote single-run payload hooks."""
    return RemoteRunPayloadHooks(
        build_run_command_fn=build_run_command_fn,
        build_remote_submit_command_fn=build_remote_submit_command_fn,
        require_remote_host_fn=require_remote_host_fn,
        default_remote_mpi_exec_fn=default_remote_mpi_exec_fn,
    )


def build_remote_run_workflow_builder_hooks(
    *,
    remote_job_session_hooks_fn: Callable[[dict[str, float]], Any],
    remote_job_submit_hooks_fn: Callable[[dict[str, float]], Any],
    remote_run_monitor_hooks_fn: Callable[..., Any],
    remote_run_artifact_hooks_fn: Callable[[dict[str, float]], Any],
    build_remote_run_payload_fn: Callable[..., tuple[list[str], dict[str, Any], str]],
    upload_remote_text_file_fn: Callable[..., Any],
    json_ready_fn: Callable[[Any], Any],
    remote_fast_sync_files_fn: Callable[[dict[str, Any]], tuple[str, ...]],
    preferred_soma_trace_artifact_name_fn: Callable[[], str],
    write_run_info_fn: Callable[..., Any],
    summarize_submit_response_fn: Callable[[dict[str, Any]], Any],
    summarize_status_fn: Callable[[dict[str, Any] | None], Any],
    timing_summary_text_fn: Callable[[dict[str, float]], str],
    build_return_value_fn: Callable[..., Any],
    shell_join_fn: Callable[[list[str]], str],
    progress_write: Callable[[str], None],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    perf_counter_fn: Callable[[], float],
) -> NotebookRemoteRunWorkflowBuilderHooks:
    """Build the concrete olfactory-bulb remote single-run workflow hooks."""
    return NotebookRemoteRunWorkflowBuilderHooks(
        remote_job_session_hooks_fn=remote_job_session_hooks_fn,
        remote_job_submit_hooks_fn=remote_job_submit_hooks_fn,
        remote_run_monitor_hooks_fn=remote_run_monitor_hooks_fn,
        remote_run_artifact_hooks_fn=remote_run_artifact_hooks_fn,
        build_remote_run_payload_fn=build_remote_run_payload_fn,
        upload_remote_text_file_fn=upload_remote_text_file_fn,
        json_ready_fn=json_ready_fn,
        remote_fast_sync_files_fn=remote_fast_sync_files_fn,
        preferred_soma_trace_artifact_name_fn=preferred_soma_trace_artifact_name_fn,
        write_run_info_fn=write_run_info_fn,
        summarize_submit_response_fn=summarize_submit_response_fn,
        summarize_status_fn=summarize_status_fn,
        timing_summary_text_fn=timing_summary_text_fn,
        build_return_value_fn=build_return_value_fn,
        shell_join_fn=shell_join_fn,
        progress_write=progress_write,
        record_timing_fn=record_timing_fn,
        perf_counter_fn=perf_counter_fn,
    )


def build_remote_sweep_workflow_builder_hooks(
    *,
    remote_job_session_hooks_fn: Callable[[dict[str, float]], Any],
    remote_job_submit_hooks_fn: Callable[[dict[str, float]], Any],
    remote_sweep_monitor_hooks_fn: Callable[..., Any],
    remote_sweep_artifact_hooks_fn: Callable[..., Any],
    build_remote_submit_command_fn: Callable[..., str],
    upload_remote_text_file_fn: Callable[..., Any],
    refresh_remote_heartbeat_fn: Callable[..., Any],
    should_sync_remote_sweep_finished_items_fn: Callable[..., bool],
    sync_remote_result_dir_fn: Callable[..., Any],
    remote_sweep_item_sync_files_fn: Callable[[dict[str, Any]], tuple[str, ...]],
    local_sync_artifact_is_usable_fn: Callable[[str | Path], bool],
    synthesize_partial_sync_summary_fn: Callable[..., dict[str, Any]],
    persist_sweep_fn: Callable[..., Any],
    merge_sweep_info_payload_fn: Callable[..., Any],
    summarize_status_fn: Callable[[dict[str, Any] | None], Any],
    timing_summary_text_fn: Callable[[dict[str, float]], str],
    write_run_info_fn: Callable[..., Any],
    load_run_record_fn: Callable[[str | Path], Any],
    load_result_fn: Callable[[Any], Any],
    resolve_local_sweep_item_dir_fn: Callable[[str | Path, str], Path | None],
    json_ready_fn: Callable[[Any], Any],
    read_json_if_present_fn: Callable[[str | Path], Any],
    progress_write: Callable[[str], None],
    record_timing_fn: Callable[[dict[str, float], str, float], Any],
    perf_counter_fn: Callable[[], float],
    default_remote_mpi_exec_fn: Callable[[], str],
) -> NotebookRemoteSweepWorkflowBuilderHooks:
    """Build the concrete olfactory-bulb remote sweep workflow hooks."""
    return NotebookRemoteSweepWorkflowBuilderHooks(
        remote_job_session_hooks_fn=remote_job_session_hooks_fn,
        remote_job_submit_hooks_fn=remote_job_submit_hooks_fn,
        remote_sweep_monitor_hooks_fn=remote_sweep_monitor_hooks_fn,
        remote_sweep_artifact_hooks_fn=remote_sweep_artifact_hooks_fn,
        build_remote_submit_command_fn=build_remote_submit_command_fn,
        upload_remote_text_file_fn=upload_remote_text_file_fn,
        refresh_remote_heartbeat_fn=refresh_remote_heartbeat_fn,
        should_sync_remote_sweep_finished_items_fn=should_sync_remote_sweep_finished_items_fn,
        sync_remote_result_dir_fn=sync_remote_result_dir_fn,
        remote_sweep_item_sync_files_fn=remote_sweep_item_sync_files_fn,
        local_sync_artifact_is_usable_fn=local_sync_artifact_is_usable_fn,
        synthesize_partial_sync_summary_fn=synthesize_partial_sync_summary_fn,
        persist_sweep_fn=persist_sweep_fn,
        merge_sweep_info_payload_fn=merge_sweep_info_payload_fn,
        summarize_status_fn=summarize_status_fn,
        timing_summary_text_fn=timing_summary_text_fn,
        write_run_info_fn=write_run_info_fn,
        load_run_record_fn=load_run_record_fn,
        load_result_fn=load_result_fn,
        resolve_local_sweep_item_dir_fn=resolve_local_sweep_item_dir_fn,
        json_ready_fn=json_ready_fn,
        read_json_if_present_fn=read_json_if_present_fn,
        progress_write=progress_write,
        record_timing_fn=record_timing_fn,
        perf_counter_fn=perf_counter_fn,
        default_remote_mpi_exec_fn=default_remote_mpi_exec_fn,
    )


def build_local_run_payload_hooks(
    *,
    benchmark_param_overrides_payload_fn: Callable[[dict[str, Any]], tuple[dict[str, Any], str | None]],
    write_benchmark_overrides_file_fn: Callable[[str | Path, dict[str, Any]], Any],
    build_run_command_fn: Callable[..., list[str]],
) -> LocalRunPayloadHooks:
    """Build the concrete olfactory-bulb local run payload hooks."""
    return LocalRunPayloadHooks(
        benchmark_param_overrides_payload_fn=benchmark_param_overrides_payload_fn,
        write_benchmark_overrides_file_fn=write_benchmark_overrides_file_fn,
        build_run_command_fn=build_run_command_fn,
    )


def build_local_run_hook_builder_hooks(
    *,
    read_summary_fn: Callable[[Path], dict[str, Any]],
    write_run_info_fn: Callable[..., Any],
    build_param_overrides_fn: Callable[[dict[str, Any]], dict[str, Any]],
    run_record_factory_fn: Callable[..., Any],
) -> NotebookLocalRunHookBuilderHooks:
    """Build the concrete olfactory-bulb local run builder hooks."""
    return NotebookLocalRunHookBuilderHooks(
        read_summary_fn=read_summary_fn,
        write_run_info_fn=write_run_info_fn,
        build_param_overrides_fn=build_param_overrides_fn,
        run_record_factory_fn=run_record_factory_fn,
    )


def build_notebook_workflow_adapter_hooks(
    *,
    load_run_record_fn: Callable[..., Any],
    load_result_fn: Callable[[Any], Any],
    run_simulation_fn: Callable[..., Any],
    merge_run_info_payload_fn: Callable[[str | Path, dict[str, Any]], Any],
    save_sweep_fn: Callable[..., Path],
    sweep_item_runs_dir_fn: Callable[[dict[str, Any], str], str | Path],
    sweep_dir_fn: Callable[[dict[str, Any], str], Path],
) -> NotebookWorkflowAdapterHooks:
    """Build the concrete olfactory-bulb notebook workflow hooks."""
    return NotebookWorkflowAdapterHooks(
        load_run_record_fn=load_run_record_fn,
        load_result_fn=load_result_fn,
        run_simulation_fn=run_simulation_fn,
        merge_run_info_payload_fn=merge_run_info_payload_fn,
        save_sweep_fn=save_sweep_fn,
        sweep_item_runs_dir_fn=sweep_item_runs_dir_fn,
        sweep_dir_fn=sweep_dir_fn,
    )


def build_notebook_run_dispatch_hooks(
    *,
    normalize_config_fn: Callable[[dict[str, Any] | None], dict[str, Any]],
    make_timestamp_fn: Callable[[], str],
    make_label_fn: Callable[[dict[str, Any], str], str],
    build_local_run_payload_fn: Callable[..., Any],
    local_run_payload_hooks_fn: Callable[[], Any],
    build_local_run_hooks_fn: Callable[[Any], Any],
    local_run_hook_builder_hooks_fn: Callable[[], Any],
    execute_local_run_fn: Callable[..., Any],
    execute_remote_run_fn: Callable[..., Any],
    default_results_base: str | Path,
) -> NotebookRunDispatchAdapterHooks:
    """Build the concrete olfactory-bulb notebook run-dispatch hooks."""
    return NotebookRunDispatchAdapterHooks(
        normalize_config_fn=normalize_config_fn,
        make_timestamp_fn=make_timestamp_fn,
        make_label_fn=make_label_fn,
        build_local_run_payload_fn=build_local_run_payload_fn,
        local_run_payload_hooks_fn=local_run_payload_hooks_fn,
        build_local_run_hooks_fn=build_local_run_hooks_fn,
        local_run_hook_builder_hooks_fn=local_run_hook_builder_hooks_fn,
        execute_local_run_fn=execute_local_run_fn,
        execute_remote_run_fn=execute_remote_run_fn,
        default_results_base=default_results_base,
    )


def build_notebook_sweep_dispatch_hooks(
    *,
    prepare_sweep_plan_fn: Callable[..., dict[str, Any]],
    uses_remote_batch_engine_fn: Callable[[dict[str, Any]], bool],
    build_local_sweep_hooks_fn: Callable[[Any], Any],
    notebook_workflow_adapter_hooks_fn: Callable[[], Any],
    execute_local_sweep_plan_fn: Callable[[Any, dict[str, Any]], dict[str, Any]],
    execute_remote_sweep_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> NotebookSweepDispatchAdapterHooks:
    """Build the concrete olfactory-bulb notebook sweep-dispatch hooks."""
    return NotebookSweepDispatchAdapterHooks(
        prepare_sweep_plan_fn=prepare_sweep_plan_fn,
        uses_remote_batch_engine_fn=uses_remote_batch_engine_fn,
        build_local_sweep_hooks_fn=build_local_sweep_hooks_fn,
        notebook_workflow_adapter_hooks_fn=notebook_workflow_adapter_hooks_fn,
        execute_local_sweep_plan_fn=execute_local_sweep_plan_fn,
        execute_remote_sweep_fn=execute_remote_sweep_fn,
    )


def build_notebook_result_hooks(
    *,
    find_soma_trace_artifact_fn: Callable[[str | Path], Path | None],
    preferred_soma_trace_artifact_name_fn: Callable[[], str],
    soma_trace_artifact_candidates_fn: Callable[[], tuple[str, ...]],
    result_view_hooks: Any,
    artifact_loading_hooks: Any,
) -> NotebookResultHooks:
    """Build the concrete olfactory-bulb notebook result hooks."""
    return NotebookResultHooks(
        find_soma_trace_artifact_fn=find_soma_trace_artifact_fn,
        preferred_soma_trace_artifact_name_fn=preferred_soma_trace_artifact_name_fn,
        soma_trace_artifact_candidates_fn=soma_trace_artifact_candidates_fn,
        result_view_hooks=result_view_hooks,
        artifact_loading_hooks=artifact_loading_hooks,
    )


def build_notebook_sweep_hooks(
    *,
    sweeps_base: str | Path,
    default_results_base: str | Path,
    make_timestamp_fn: Callable[[], str],
    safe_name_fn: Callable[[Any], str],
    json_ready_fn: Callable[[Any], Any],
    resolve_git_head_fn: Callable[[], str | None],
    load_result_fn: Callable[..., Any],
    save_sweep_fn: Callable[..., Path],
    load_sweep_fn: Callable[..., dict[str, Any]],
    list_sweeps_fn: Callable[..., list[Path]],
    save_animation_fn: Callable[..., Path],
    save_sweep_animation_stream_fn: Callable[..., Path],
    animate_sweep_plots_fn: Callable[..., dict[str, Path]],
    build_sweep_plot_callable_fn: Callable[[Any], tuple[Any, str]],
    normalize_sweep_plot_spec_fn: Callable[[Any], Any],
    is_deprecated_sweep_animation_spec_fn: Callable[[Any], bool],
    deprecated_plot_names: tuple[str, ...],
    progress_factory_fn: Callable[[int, str], Any | None],
    progress_write_fn: Callable[[str], None],
    run_parameter_sweep_fn: Callable[..., dict[str, Any]],
    run_grid_sweep_fn: Callable[..., dict[str, Any]],
) -> NotebookSweepHooks:
    """Build the concrete olfactory-bulb notebook sweep hooks."""
    return NotebookSweepHooks(
        sweeps_base=sweeps_base,
        default_results_base=default_results_base,
        make_timestamp_fn=make_timestamp_fn,
        safe_name_fn=safe_name_fn,
        json_ready_fn=json_ready_fn,
        resolve_git_head_fn=resolve_git_head_fn,
        load_result_fn=load_result_fn,
        save_sweep_fn=save_sweep_fn,
        load_sweep_fn=load_sweep_fn,
        list_sweeps_fn=list_sweeps_fn,
        save_animation_fn=save_animation_fn,
        save_sweep_animation_stream_fn=save_sweep_animation_stream_fn,
        animate_sweep_plots_fn=animate_sweep_plots_fn,
        build_sweep_plot_callable_fn=build_sweep_plot_callable_fn,
        normalize_sweep_plot_spec_fn=normalize_sweep_plot_spec_fn,
        is_deprecated_sweep_animation_spec_fn=is_deprecated_sweep_animation_spec_fn,
        deprecated_plot_names=deprecated_plot_names,
        progress_factory_fn=progress_factory_fn,
        progress_write_fn=progress_write_fn,
        run_parameter_sweep_fn=run_parameter_sweep_fn,
        run_grid_sweep_fn=run_grid_sweep_fn,
    )


def build_notebook_presentation_hooks(
    *,
    default_results_base: str | Path,
    make_timestamp_fn: Callable[[], str],
    safe_name_fn: Callable[[Any], str],
    plt_module: Any,
    save_figure_fn: Callable[..., Path],
    plot_input_overview_fn: Callable[..., Any],
    plot_voltage_traces_fn: Callable[..., Any],
    plot_spike_raster_fn: Callable[..., Any],
    plot_gc_output_overview_fn: Callable[..., Any],
    plot_lfp_overview_fn: Callable[..., Any],
    plot_spectrogram_fn: Callable[..., Any],
    plot_wavelet_fn: Callable[..., Any],
    plot_wavelet_band_power_fn: Callable[..., Any],
    result_overview_fn: Callable[[dict[str, Any]], dict[str, Any]],
    build_run_config_fn: Callable[..., dict[str, Any]],
    resolve_effective_params_fn: Callable[[dict[str, Any]], dict[str, Any]],
    resolve_paramset_defaults_fn: Callable[[str], dict[str, Any]],
    diff_values_fn: Callable[[Any, Any], list[dict[str, Any]]],
    extract_runtime_control_snapshot_fn: Callable[[dict[str, Any]], dict[str, Any]],
    print_diff_section_fn: Callable[[str, list[dict[str, Any]], int | None], None],
    write_fn: Callable[[str], None] = print,
) -> NotebookPresentationHooks:
    """Build the concrete olfactory-bulb notebook presentation hooks."""
    return NotebookPresentationHooks(
        default_results_base=default_results_base,
        make_timestamp_fn=make_timestamp_fn,
        safe_name_fn=safe_name_fn,
        plt_module=plt_module,
        save_figure_fn=save_figure_fn,
        plot_input_overview_fn=plot_input_overview_fn,
        plot_voltage_traces_fn=plot_voltage_traces_fn,
        plot_spike_raster_fn=plot_spike_raster_fn,
        plot_gc_output_overview_fn=plot_gc_output_overview_fn,
        plot_lfp_overview_fn=plot_lfp_overview_fn,
        plot_spectrogram_fn=plot_spectrogram_fn,
        plot_wavelet_fn=plot_wavelet_fn,
        plot_wavelet_band_power_fn=plot_wavelet_band_power_fn,
        result_overview_fn=result_overview_fn,
        build_run_config_fn=build_run_config_fn,
        resolve_effective_params_fn=resolve_effective_params_fn,
        resolve_paramset_defaults_fn=resolve_paramset_defaults_fn,
        diff_values_fn=diff_values_fn,
        extract_runtime_control_snapshot_fn=extract_runtime_control_snapshot_fn,
        print_diff_section_fn=print_diff_section_fn,
        write_fn=write_fn,
    )
