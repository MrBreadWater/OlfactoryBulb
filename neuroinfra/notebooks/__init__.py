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

__all__ = [
    "diff_values",
    "flatten_for_diff",
    "format_diff_value",
    "RunRecord",
    "json_ready",
    "list_json_configs",
    "list_run_dirs",
    "LocalRunHooks",
    "load_json_config",
    "load_run_config",
    "load_run_record",
    "RunInfoHooks",
    "build_run_info_payload",
    "DEFAULT_COMMAND_FILENAME",
    "env_subset",
    "execute_local_run",
    "load_run_info_payload",
    "merge_run_info_payload",
    "persist_run_info",
    "print_diff_section",
    "read_json_if_present",
    "resolve_run_dir",
    "save_json_config",
    "save_figure",
    "SweepPlanHooks",
    "prepare_sweep_plan",
    "set_nested_value",
    "split_path_parts",
]
