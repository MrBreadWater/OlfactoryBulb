"""Smoke tests for standardized notebook-runtime and session-policy helpers."""

from __future__ import annotations

from copy import deepcopy

import neuroinfra.remote.notebook_runtime as notebook_runtime
import obgpu_experiment_helpers as hlp


class _LiveTransport:
    def is_active(self) -> bool:
        return True

    def is_authenticated(self) -> bool:
        return True


class _DeadTransport:
    def is_active(self) -> bool:
        return False

    def is_authenticated(self) -> bool:
        return False


def main() -> None:
    runtime_store: dict[str, object] = {}
    ensured = notebook_runtime.ensure_notebook_remote_runtime(runtime_store)
    assert ensured is runtime_store
    assert isinstance(runtime_store["paramiko_connections"], dict)
    assert isinstance(runtime_store["paramiko_authenticated_keys"], set)
    assert isinstance(runtime_store["paramiko_prompt_cache"], dict)
    assert runtime_store["slurm_allocation_atexit_registered"] is False

    assert notebook_runtime.transport_is_usable(_LiveTransport()) is True
    assert notebook_runtime.transport_is_usable(_DeadTransport()) is False
    assert notebook_runtime.transport_is_usable(None) is False

    prompt_cache: dict[str, dict[str, str]] = {}
    connection_key = "user@cluster:22"
    assert notebook_runtime.prompt_key(" Password   for host: ") == "Password for host:"
    assert notebook_runtime.get_cached_prompt_response(prompt_cache, connection_key, "Password:") is None
    notebook_runtime.cache_prompt_response(prompt_cache, connection_key, "Password:", "secret")
    assert notebook_runtime.get_cached_prompt_response(prompt_cache, connection_key, "Password:") == "secret"
    assert notebook_runtime.has_cached_auth(prompt_cache, connection_key) is True
    assert notebook_runtime.can_reconnect(
        connection_key=connection_key,
        preserve_session=True,
        allow_reauth=False,
        authenticated_keys={connection_key},
        prompt_cache=prompt_cache,
    ) is True
    assert notebook_runtime.can_reconnect(
        connection_key=connection_key,
        preserve_session=True,
        allow_reauth=False,
        authenticated_keys=set(),
        prompt_cache={},
    ) is True
    assert notebook_runtime.can_reconnect(
        connection_key=connection_key,
        preserve_session=True,
        allow_reauth=False,
        authenticated_keys={connection_key},
        prompt_cache={},
    ) is False
    assert "remote_preserve_paramiko_session=True" in notebook_runtime.midrun_reauth_error(connection_key)
    assert connection_key in notebook_runtime.midrun_reauth_error(connection_key)

    saved_prompt_cache = deepcopy(hlp._LIVE_PARAMIKO_PROMPT_CACHE)
    saved_authenticated = set(hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS)
    try:
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.clear()
        cfg = {"remote_host": "user@cluster", "ssh_options": []}
        cache_key = hlp._paramiko_connection_key(cfg)
        assert hlp._paramiko_transport_is_usable(_LiveTransport()) is True
        assert hlp._paramiko_transport_is_usable(_DeadTransport()) is False
        assert hlp._paramiko_prompt_key(" Password : ") == notebook_runtime.prompt_key(" Password : ")
        assert hlp._get_cached_paramiko_prompt_response(cfg, "Password:") is None
        hlp._cache_paramiko_prompt_response(cfg, "Password:", "secret")
        assert hlp._get_cached_paramiko_prompt_response(cfg, "Password:") == "secret"
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.add(cache_key)
        assert hlp._paramiko_has_cached_auth(cfg) is True
        assert hlp._paramiko_can_reconnect(cfg) is True
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        assert hlp._paramiko_can_reconnect(cfg) is False
        assert cache_key in hlp._paramiko_midrun_reauth_error(cfg)
    finally:
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.clear()
        hlp._LIVE_PARAMIKO_PROMPT_CACHE.update(saved_prompt_cache)
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.clear()
        hlp._LIVE_PARAMIKO_AUTHENTICATED_KEYS.update(saved_authenticated)

    print("neuroinfra remote notebook runtime smoke test: OK")


if __name__ == "__main__":
    main()
