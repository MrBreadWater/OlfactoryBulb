"""Reusable helpers for notebook-managed remote allocation caching."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha1
import json
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


DEFAULT_RUNTIME_CONFIG_KEYS: tuple[str, ...] = (
    "remote_host",
    "remote_results_root",
    "remote_heartbeat_timeout_s",
    "runner_backend",
    "ssh_options",
    "ssh_transport",
    "ssh_keepalive_s",
    "remote_preserve_paramiko_session",
)


def allocation_runtime_config(
    config: Mapping[str, Any],
    *,
    keys: Iterable[str] = DEFAULT_RUNTIME_CONFIG_KEYS,
) -> dict[str, Any]:
    """Return the config subset needed to rediscover or cancel one allocation."""
    return {key: deepcopy(config.get(key)) for key in keys if key in config}


def allocation_signature(
    *,
    connection_key: str,
    results_root: PurePosixPath,
    partition: str | None = None,
    account: str | None = None,
    time_limit: str = "",
    gpus: int | None = None,
    cpus_per_task: int | None = None,
    mem: str | None = None,
    extra_args: Iterable[str] = (),
    remote_conda_activate_cmd: str = "",
    remote_runtime_profiles: Any = (),
    remote_fallback_conda_activate_cmd: str = "",
    remote_fast_node_feature: str = "",
    remote_mechanism_profile: str = "default",
    remote_fallback_mechanism_profile: str = "portable",
    name: str = "obgpu_notebook_alloc",
) -> dict[str, Any]:
    """Return the cache signature for one reusable remote Slurm allocation."""
    return {
        "remote_host": str(connection_key),
        "remote_results_root": results_root.as_posix(),
        "partition": None if partition in (None, "") else str(partition),
        "account": None if account in (None, "") else str(account),
        "time": str(time_limit or ""),
        "gpus": None if gpus in (None, "") else int(gpus),
        "cpus_per_task": None if cpus_per_task in (None, "") else int(cpus_per_task),
        "mem": None if mem in (None, "") else str(mem),
        "extra_args": [str(arg) for arg in extra_args],
        "remote_conda_activate_cmd": str(remote_conda_activate_cmd or ""),
        "remote_runtime_profiles": deepcopy(remote_runtime_profiles),
        "remote_fallback_conda_activate_cmd": str(remote_fallback_conda_activate_cmd or ""),
        "remote_fast_node_feature": str(remote_fast_node_feature or ""),
        "remote_mechanism_profile": str(remote_mechanism_profile or "default"),
        "remote_fallback_mechanism_profile": str(remote_fallback_mechanism_profile or "portable"),
        "name": str(name or "obgpu_notebook_alloc"),
    }


def allocation_cache_key(signature: Mapping[str, Any]) -> str:
    """Return the runtime cache key for one reusable remote Slurm allocation."""
    payload = json.dumps(dict(signature), sort_keys=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def allocation_record(
    *,
    job_id: str,
    cache_key: str,
    allocation_root: str,
    batch_script: str,
    heartbeat_path: str,
    heartbeat_timeout_s: Any,
    slurm_log_pattern: str,
    name: str,
    cached: bool,
    manual: bool,
    config: Mapping[str, Any] | None = None,
    state: str = "",
    reason: str = "",
    location: str = "",
) -> dict[str, Any]:
    """Return one normalized notebook allocation record."""
    return {
        "job_id": str(job_id),
        "cache_key": str(cache_key),
        "allocation_root": str(allocation_root),
        "batch_script": str(batch_script),
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_timeout_s": heartbeat_timeout_s,
        "slurm_log_pattern": str(slurm_log_pattern),
        "name": str(name),
        "cached": bool(cached),
        "manual": bool(manual),
        "config": None if config is None else dict(config),
        "state": str(state),
        "reason": str(reason),
        "location": str(location),
    }


def manual_allocation_record(job_id: str) -> dict[str, Any]:
    """Return one normalized record for a manually supplied allocation."""
    return {
        "job_id": str(job_id),
        "cached": False,
        "manual": True,
        "state": "",
        "reason": "",
        "location": "",
    }


def disabled_allocation_record() -> dict[str, Any]:
    """Return one normalized record for configs that do not reuse allocations."""
    return {
        "job_id": None,
        "cached": False,
        "manual": False,
        "state": "",
        "reason": "",
        "location": "",
    }
