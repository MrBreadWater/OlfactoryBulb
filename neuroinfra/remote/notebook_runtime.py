"""Reusable notebook-runtime and Paramiko session-policy helpers."""

from __future__ import annotations

from typing import Any, MutableMapping


def ensure_notebook_remote_runtime(storage: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Populate the standard remote-runtime keys on one notebook-shared store."""
    storage.setdefault("paramiko_connections", {})
    storage.setdefault("paramiko_authenticated_keys", set())
    storage.setdefault("paramiko_prompt_cache", {})
    storage.setdefault("slurm_allocations", {})
    storage.setdefault("remote_git_refs", {})
    storage.setdefault("remote_helper_caches", {})
    storage.setdefault("remote_preflight", {})
    storage.setdefault("remote_stale_cleanup", {})
    storage.setdefault("slurm_allocation_atexit_registered", False)
    return storage


def transport_is_usable(transport: Any) -> bool:
    """Return whether one cached Paramiko transport still looks usable."""
    if transport is None:
        return False
    try:
        return bool(transport.is_active() and transport.is_authenticated())
    except Exception:
        return False


def prompt_key(prompt_text: str) -> str:
    """Normalize one interactive-auth prompt into a stable cache key."""
    return " ".join(str(prompt_text or "").strip().split())


def cached_prompt_responses(
    prompt_cache: MutableMapping[str, dict[str, str]],
    connection_key: str,
) -> dict[str, str]:
    """Return the mutable prompt-response cache for one endpoint."""
    cached = prompt_cache.get(connection_key)
    if cached is None:
        cached = {}
        prompt_cache[connection_key] = cached
    return cached


def get_cached_prompt_response(
    prompt_cache: MutableMapping[str, dict[str, str]],
    connection_key: str,
    prompt_text: str,
) -> str | None:
    """Return one remembered auth response for this endpoint, if available."""
    return cached_prompt_responses(prompt_cache, connection_key).get(prompt_key(prompt_text))


def cache_prompt_response(
    prompt_cache: MutableMapping[str, dict[str, str]],
    connection_key: str,
    prompt_text: str,
    response: str,
) -> None:
    """Remember one auth response for later noninteractive reconnects."""
    cached_prompt_responses(prompt_cache, connection_key)[prompt_key(prompt_text)] = str(response)


def has_cached_auth(
    prompt_cache: MutableMapping[str, dict[str, str]],
    connection_key: str,
) -> bool:
    """Return whether one endpoint has cached auth responses for silent reconnect."""
    return bool(prompt_cache.get(connection_key))


def can_reconnect(
    *,
    connection_key: str,
    preserve_session: bool,
    allow_reauth: bool,
    authenticated_keys: set[str],
    prompt_cache: MutableMapping[str, dict[str, str]],
) -> bool:
    """Return whether one dead transport may be recovered automatically."""
    if not preserve_session or connection_key not in authenticated_keys:
        return True
    return allow_reauth or has_cached_auth(prompt_cache, connection_key)


def midrun_reauth_error(connection_key: str) -> str:
    """Explain why a fresh Paramiko login is being refused mid-run."""
    return (
        "The cached Paramiko SSH session is no longer usable, and "
        "remote_preserve_paramiko_session=True is preventing an automatic re-login.\n"
        f"Endpoint: {connection_key}\n"
        "This is intentional so notebook runs fail closed instead of prompting for password/2FA mid-run."
    )
