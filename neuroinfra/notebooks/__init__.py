"""Reusable notebook-facing run metadata helpers."""

from .runs import (
    RunRecord,
    list_run_dirs,
    load_run_config,
    load_run_record,
    read_json_if_present,
    resolve_run_dir,
)
from .config_store import (
    json_ready,
    list_json_configs,
    load_json_config,
    save_json_config,
)
from .local_runs import (
    DEFAULT_COMMAND_FILENAME,
    LocalRunHooks,
    execute_local_run,
)
from .remote_jobs import (
    RemoteJobSession,
    RemoteJobSessionHooks,
    RemoteJobSubmission,
    RemoteJobSubmitHooks,
    prepare_remote_job_session,
    submit_remote_json_job,
)
from .remote_runs import (
    RemoteRunWorkflowHooks,
    execute_remote_run_workflow,
)
from .run_info import (
    RunInfoHooks,
    build_run_info_payload,
    env_subset,
    load_run_info_payload,
    merge_run_info_payload,
    persist_run_info,
)
from .reporting import (
    diff_values,
    flatten_for_diff,
    format_diff_value,
    print_diff_section,
    save_figure,
)
from .sweeps import (
    SweepPlanHooks,
    prepare_sweep_plan,
    set_nested_value,
    split_path_parts,
)
from .workflows import (
    LoadRunPairHooks,
    LocalSweepHooks,
    RunAndLoadHooks,
    load_run_pair,
    run_and_load,
    run_local_sweep_plan,
)

__all__ = [
    "diff_values",
    "flatten_for_diff",
    "format_diff_value",
    "RunRecord",
    "json_ready",
    "list_json_configs",
    "list_run_dirs",
    "LocalRunHooks",
    "RemoteJobSession",
    "RemoteJobSessionHooks",
    "RemoteJobSubmission",
    "RemoteJobSubmitHooks",
    "RemoteRunWorkflowHooks",
    "load_json_config",
    "LoadRunPairHooks",
    "load_run_config",
    "load_run_record",
    "RunInfoHooks",
    "build_run_info_payload",
    "DEFAULT_COMMAND_FILENAME",
    "env_subset",
    "execute_local_run",
    "load_run_pair",
    "load_run_info_payload",
    "merge_run_info_payload",
    "persist_run_info",
    "print_diff_section",
    "prepare_remote_job_session",
    "read_json_if_present",
    "resolve_run_dir",
    "execute_remote_run_workflow",
    "save_json_config",
    "save_figure",
    "SweepPlanHooks",
    "prepare_sweep_plan",
    "LocalSweepHooks",
    "RunAndLoadHooks",
    "run_and_load",
    "run_local_sweep_plan",
    "set_nested_value",
    "split_path_parts",
    "submit_remote_json_job",
]
