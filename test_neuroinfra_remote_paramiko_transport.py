"""Smoke tests for the extracted notebook Paramiko transport helpers."""

from __future__ import annotations

from types import SimpleNamespace

from neuroinfra.remote.paramiko_transport import (
    ParamikoTransportContext,
    connect_error_is_retryable,
)


class _AuthenticationException(Exception):
    pass


class _SSHException(Exception):
    pass


class _BadAuthenticationType(Exception):
    def __init__(self, allowed_types: list[str]):
        super().__init__("bad auth type")
        self.allowed_types = allowed_types


class _PartialAuthentication(Exception):
    def __init__(self, allowed_types: list[str]):
        super().__init__("partial auth")
        self.allowed_types = allowed_types


class _FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeSFTPHandle:
    def close(self) -> None:
        pass


class _FakeSFTPClient:
    calls = 0

    @classmethod
    def from_transport(cls, _transport):
        cls.calls += 1
        return _FakeSFTPHandle()


class _FakeChannel:
    def __init__(self, stdout: bytes = b"ok\n", stderr: bytes = b"", exit_status: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_status = exit_status
        self.exec_calls: list[str] = []
        self.closed = False

    def settimeout(self, _timeout: float) -> None:
        return None

    def exec_command(self, command: str) -> None:
        self.exec_calls.append(command)

    def recv_ready(self) -> bool:
        return bool(self._stdout)

    def recv(self, _count: int) -> bytes:
        payload = self._stdout
        self._stdout = b""
        return payload

    def recv_stderr_ready(self) -> bool:
        return bool(self._stderr)

    def recv_stderr(self, _count: int) -> bytes:
        payload = self._stderr
        self._stderr = b""
        return payload

    def exit_status_ready(self) -> bool:
        return bool(self.exec_calls)

    def recv_exit_status(self) -> int:
        return self._exit_status

    def close(self) -> None:
        self.closed = True


class _FakeTransport:
    auth_methods = ["password"]
    last_channel: _FakeChannel | None = None

    def __init__(self, raw_sock) -> None:
        self.raw_sock = raw_sock
        self._active = True
        self._authenticated = False
        self.keepalive = None

    def start_client(self, timeout: float) -> None:
        self.start_timeout = timeout

    def set_keepalive(self, seconds: int) -> None:
        self.keepalive = seconds

    def auth_none(self, _username: str) -> None:
        raise _BadAuthenticationType(list(self.auth_methods))

    def auth_password(self, _username: str, password: str) -> None:
        if password != "cached-secret":
            raise _AuthenticationException("bad password")
        self._authenticated = True

    def auth_interactive(self, _username: str, _handler) -> None:
        self._authenticated = True

    def is_authenticated(self) -> bool:
        return self._authenticated

    def is_active(self) -> bool:
        return self._active

    def open_session(self) -> _FakeChannel:
        channel = _FakeChannel(stdout=b"stdout\n", stderr=b"stderr\n", exit_status=0)
        _FakeTransport.last_channel = channel
        return channel

    def close(self) -> None:
        self._active = False


def _connection_key(config: dict[str, object]) -> str:
    return f"{config.get('remote_host', 'user@host')}:{config.get('remote_port', 22)}"


def _prompt_get(cache: dict[str, dict[str, str]], config: dict[str, object], prompt: str) -> str | None:
    return cache.setdefault(_connection_key(config), {}).get(prompt)


def _prompt_set(cache: dict[str, dict[str, str]], config: dict[str, object], prompt: str, response: str) -> None:
    cache.setdefault(_connection_key(config), {})[prompt] = response


def _build_context(
    *,
    config: dict[str, object] | None = None,
    live_connections=None,
    authenticated_keys=None,
    prompt_cache=None,
    can_reconnect=lambda _cfg: True,
    socket_attempts=None,
) -> ParamikoTransportContext:
    cfg = dict(config or {"remote_host": "user@host", "remote_port": 22, "remote_preserve_paramiko_session": True})
    live_connections = {} if live_connections is None else live_connections
    authenticated_keys = set() if authenticated_keys is None else authenticated_keys
    prompt_cache = {} if prompt_cache is None else prompt_cache
    progress: list[str] = []
    attempts = [] if socket_attempts is None else socket_attempts

    def _socket_create_connection(_target, timeout: float):
        if attempts:
            next_value = attempts.pop(0)
            if isinstance(next_value, BaseException):
                raise next_value
            return next_value
        return _FakeSocket()

    paramiko_module = SimpleNamespace(
        Transport=_FakeTransport,
        SFTPClient=_FakeSFTPClient,
        SSHException=_SSHException,
        AuthenticationException=_AuthenticationException,
        BadAuthenticationType=_BadAuthenticationType,
        ssh_exception=SimpleNamespace(PartialAuthentication=_PartialAuthentication),
    )
    _prompt_set(prompt_cache, cfg, "Password for user@host:", "cached-secret")
    context = ParamikoTransportContext(
        config=cfg,
        paramiko_module=paramiko_module,
        live_connections=live_connections,
        authenticated_keys=authenticated_keys,
        progress_write=progress.append,
        connection_key_fn=_connection_key,
        can_reconnect_fn=can_reconnect,
        midrun_reauth_error_fn=lambda _cfg: "midrun reauth refused",
        remote_endpoint_fn=lambda _cfg: ("host", 22, "user"),
        connect_retry_count_fn=lambda _cfg: 2,
        connect_retry_backoff_s_fn=lambda _cfg: 0.0,
        transport_is_usable_fn=lambda transport: bool(transport and transport.is_active() and transport.is_authenticated()),
        get_cached_prompt_response_fn=lambda cfg_inner, prompt: _prompt_get(prompt_cache, cfg_inner, prompt),
        cache_prompt_response_fn=lambda cfg_inner, prompt, response: _prompt_set(prompt_cache, cfg_inner, prompt, response),
        ssh_command_timeout_s_fn=lambda _cfg: 2.0,
        ssh_exec_timeout_s_fn=lambda _cfg: 2.0,
        socket_create_connection_fn=_socket_create_connection,
    )
    context._test_progress = progress  # type: ignore[attr-defined]
    return context


def main() -> None:
    paramiko_module = SimpleNamespace(
        SSHException=_SSHException,
        AuthenticationException=_AuthenticationException,
    )
    assert connect_error_is_retryable(EOFError(), paramiko_module=paramiko_module) is True
    assert connect_error_is_retryable(OSError("boom"), paramiko_module=paramiko_module) is True
    assert connect_error_is_retryable(_SSHException("banner"), paramiko_module=paramiko_module) is True
    assert connect_error_is_retryable(_AuthenticationException("nope"), paramiko_module=paramiko_module) is False

    live_connections: dict[str, object] = {}
    authenticated_keys: set[str] = set()
    prompt_cache: dict[str, dict[str, str]] = {}
    context = _build_context(
        live_connections=live_connections,
        authenticated_keys=authenticated_keys,
        prompt_cache=prompt_cache,
        socket_attempts=[OSError("transient banner"), _FakeSocket()],
    )
    connection = context.connect()
    assert connection["username"] == "user"
    assert _connection_key(context.config) in authenticated_keys
    assert any("retrying" in message for message in context._test_progress)  # type: ignore[attr-defined]
    assert context.connect() is connection

    _FakeSFTPClient.calls = 0
    sftp_1 = context.get_sftp()
    sftp_2 = context.get_sftp()
    assert sftp_1 is sftp_2
    assert _FakeSFTPClient.calls == 1
    context.close_sftp()
    assert connection["sftp"] is None

    result = context.run_shell("echo hi")
    assert result.returncode == 0
    assert result.stdout == "stdout\n"
    assert result.stderr == "stderr\n"
    assert _FakeTransport.last_channel is not None
    assert _FakeTransport.last_channel.exec_calls == ["bash -lc 'echo hi'"]

    stale_transport = _FakeTransport(_FakeSocket())
    live_connections = {
        "user@host:22": {"transport": stale_transport, "sftp": None, "hostname": "host", "port": 22, "username": "user"}
    }
    authenticated_keys = {"user@host:22"}
    context_blocked = _build_context(
        live_connections=live_connections,
        authenticated_keys=authenticated_keys,
        prompt_cache={"user@host:22": {"Password for user@host:": "cached-secret"}},
        can_reconnect=lambda _cfg: False,
    )
    try:
        context_blocked.connect()
        raise AssertionError("Expected mid-run reauth refusal")
    except RuntimeError as exc:
        assert "midrun reauth refused" in str(exc)

    print("neuroinfra remote paramiko transport smoke test: OK")


if __name__ == "__main__":
    main()
