"""Reusable Paramiko-backed notebook transport helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from getpass import getpass as _default_getpass
import shlex
import socket
import subprocess
import threading
import time
from typing import Any, Callable, MutableMapping


try:  # pragma: no cover - optional notebook integration
    from IPython import get_ipython as _default_ipython_getter
except Exception:  # pragma: no cover - optional notebook integration
    _default_ipython_getter = None


class SSHCommandTimeoutError(TimeoutError):
    """Raised when one notebook-managed remote shell command exceeds its budget."""


def partial_auth_exceptions(paramiko_module: Any) -> tuple[type[BaseException], ...]:
    """Return the Paramiko partial-auth exception types available in this runtime."""
    if paramiko_module is None:
        return ()
    return tuple(
        exc
        for exc in (
            getattr(paramiko_module, "PartialAuthentication", None),
            getattr(getattr(paramiko_module, "ssh_exception", None), "PartialAuthentication", None),
        )
        if exc is not None
    )


def connect_error_is_retryable(exc: BaseException, *, paramiko_module: Any) -> bool:
    """Return whether one fresh Paramiko connect failure is transient enough to retry."""
    if isinstance(exc, EOFError):
        return True
    if isinstance(exc, (socket.timeout, TimeoutError, OSError)):
        return True
    if paramiko_module is not None and isinstance(exc, getattr(paramiko_module, "AuthenticationException", tuple())):
        return False
    if paramiko_module is not None and isinstance(exc, getattr(paramiko_module, "SSHException", tuple())):
        return True
    return False


@dataclass
class ParamikoTransportContext:
    """State and callbacks for one notebook-managed Paramiko transport surface."""

    config: dict[str, Any]
    paramiko_module: Any
    live_connections: MutableMapping[str, Any]
    authenticated_keys: set[str]
    progress_write: Callable[[str], None]
    connection_key_fn: Callable[[dict[str, Any]], str]
    can_reconnect_fn: Callable[[dict[str, Any]], bool]
    midrun_reauth_error_fn: Callable[[dict[str, Any]], str]
    remote_endpoint_fn: Callable[[dict[str, Any]], tuple[str, int, str]]
    connect_retry_count_fn: Callable[[dict[str, Any]], int]
    connect_retry_backoff_s_fn: Callable[[dict[str, Any]], float]
    transport_is_usable_fn: Callable[[Any], bool]
    get_cached_prompt_response_fn: Callable[[dict[str, Any], str], str | None]
    cache_prompt_response_fn: Callable[[dict[str, Any], str, str], None]
    ssh_command_timeout_s_fn: Callable[[dict[str, Any]], float | None]
    ssh_exec_timeout_s_fn: Callable[[dict[str, Any]], float | None]
    socket_create_connection_fn: Callable[..., Any] = socket.create_connection
    sleep_fn: Callable[[float], None] = time.sleep
    getpass_fn: Callable[[str], str] = _default_getpass
    input_fn: Callable[[str], str] = input
    ipython_getter: Callable[[], Any] | None = _default_ipython_getter
    thread_cls: type[threading.Thread] = threading.Thread
    _partial_auth_exceptions: tuple[type[BaseException], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._partial_auth_exceptions = partial_auth_exceptions(self.paramiko_module)

    def prompt_response(self, prompt_text: str, *, config: dict[str, Any] | None = None) -> str:
        """Prompt the notebook user for one interactive SSH auth field."""
        cfg = self.config if config is None else config
        prompt = prompt_text.strip() or "SSH authentication:"
        if config is not None:
            cached = self.get_cached_prompt_response_fn(cfg, prompt)
            if cached is not None:
                return cached
        lowered = prompt.lower()
        kernel_prompt_exc: Exception | None = None
        if self.ipython_getter is not None:
            shell = self.ipython_getter()
            kernel = getattr(shell, "kernel", None)
            if kernel is not None:
                try:
                    if "password" in lowered or "passphrase" in lowered:
                        response = kernel.getpass(prompt + " ")
                    else:
                        response = kernel.raw_input(prompt + " ")
                except EOFError as exc:
                    kernel_prompt_exc = exc
                except Exception as exc:  # pragma: no cover - frontend-dependent
                    kernel_prompt_exc = exc
                else:
                    if config is not None:
                        self.cache_prompt_response_fn(cfg, prompt, response)
                    return response
        try:
            if "password" in lowered or "passphrase" in lowered:
                response = self.getpass_fn(prompt + " ")
            else:
                response = self.input_fn(prompt + " ")
        except EOFError as exc:
            endpoint = self.connection_key_fn(cfg) if isinstance(cfg, dict) and cfg else "<unknown>"
            frontend_note = ""
            if kernel_prompt_exc is not None:
                frontend_note = f"\nKernel input request error: {kernel_prompt_exc}"
            raise RuntimeError(
                "Paramiko authentication could not read notebook input.\n"
                f"Endpoint: {endpoint}\n"
                "This usually means the live notebook kernel cannot service an interactive getpass/input prompt. "
                "Run `paramiko_auth_probe(REMOTE_CONFIG)` in the active kernel to refresh auth, "
                "or rely on cached auth responses for unattended reconnects."
                f"{frontend_note}"
            ) from exc
        if config is not None:
            self.cache_prompt_response_fn(cfg, prompt, response)
        return response

    def drop_connection(self, *, config: dict[str, Any] | None = None) -> None:
        """Close and forget one cached Paramiko connection."""
        cfg = self.config if config is None else config
        cached = self.live_connections.pop(self.connection_key_fn(cfg), None)
        if cached is None:
            return
        sftp = cached.get("sftp")
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        transport = cached.get("transport")
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass

    def close_sftp(self, *, config: dict[str, Any] | None = None) -> None:
        """Close the cached Paramiko SFTP channel while keeping the SSH transport alive."""
        cfg = self.config if config is None else config
        cached = self.live_connections.get(self.connection_key_fn(cfg))
        if cached is None:
            return
        sftp = cached.get("sftp")
        if sftp is None:
            return
        cached["sftp"] = None
        try:
            sftp.close()
        except Exception:
            pass

    def get_sftp(self, *, config: dict[str, Any] | None = None) -> Any:
        """Return the cached Paramiko SFTP client, opening it only when needed."""
        cfg = self.config if config is None else config
        connection = self.connect(config=cfg)
        sftp = connection.get("sftp")
        if sftp is not None:
            return sftp
        self.progress_write("[Sol remote] Opening SFTP channel...")
        try:
            sftp = self.paramiko_module.SFTPClient.from_transport(connection["transport"])
        except Exception:
            connection["sftp"] = None
            if not self.transport_is_usable_fn(connection.get("transport")):
                self.drop_connection(config=cfg)
            raise
        connection["sftp"] = sftp
        return sftp

    def connect(self, *, config: dict[str, Any] | None = None) -> Any:
        """Open or reuse one persistent Paramiko transport."""
        cfg = self.config if config is None else config
        if self.paramiko_module is None:
            raise RuntimeError("Paramiko transport requested but the 'paramiko' package is not installed.")

        cache_key = self.connection_key_fn(cfg)
        preserve_session = bool(cfg.get("remote_preserve_paramiko_session", True))
        cached = self.live_connections.get(cache_key)
        if cached is not None:
            transport = cached.get("transport")
            if self.transport_is_usable_fn(transport):
                return cached
            self.live_connections.pop(cache_key, None)
            if preserve_session and cache_key in self.authenticated_keys and not self.can_reconnect_fn(cfg):
                raise RuntimeError(self.midrun_reauth_error_fn(cfg))
        elif preserve_session and cache_key in self.authenticated_keys and not self.can_reconnect_fn(cfg):
            raise RuntimeError(self.midrun_reauth_error_fn(cfg))

        hostname, port, username = self.remote_endpoint_fn(cfg)
        connect_retries = self.connect_retry_count_fn(cfg)
        backoff_s = self.connect_retry_backoff_s_fn(cfg)
        last_exc: Exception | None = None
        for attempt in range(connect_retries):
            raw_sock = None
            transport = None
            try:
                self.progress_write(f"[Sol remote] Opening SSH session to {username}@{hostname}:{port}...")
                raw_sock = self.socket_create_connection_fn((hostname, port), timeout=30.0)
                transport = self.paramiko_module.Transport(raw_sock)
                transport.start_client(timeout=30.0)
                keepalive_seconds = int(cfg.get("ssh_keepalive_s", 30) or 0)
                if keepalive_seconds > 0:
                    transport.set_keepalive(keepalive_seconds)

                auth_methods: list[str] = []
                try:
                    transport.auth_none(username)
                except self.paramiko_module.BadAuthenticationType as exc:
                    auth_methods = list(exc.allowed_types)
                except self._partial_auth_exceptions as exc:  # pragma: no cover - defensive
                    auth_methods = list(exc.allowed_types)
                except self.paramiko_module.AuthenticationException:
                    auth_methods = []

                authenticated = False
                if "keyboard-interactive" in auth_methods or not auth_methods:
                    self.progress_write(f"[Sol remote] Waiting for interactive SSH authentication...")

                    def handler(title: str, instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
                        responses: list[str] = []
                        if title:
                            print(title)
                        if instructions:
                            print(instructions)
                        for prompt_text, _echo in prompt_list:
                            responses.append(self.prompt_response(prompt_text, config=cfg))
                        return responses

                    try:
                        transport.auth_interactive(username, handler)
                        authenticated = transport.is_authenticated()
                    except self.paramiko_module.AuthenticationException:
                        authenticated = False

                if not authenticated and "password" in auth_methods:
                    try:
                        self.progress_write(f"[Sol remote] Waiting for password authentication...")
                        transport.auth_password(
                            username,
                            self.prompt_response(f"Password for {username}@{hostname}:", config=cfg),
                        )
                        authenticated = transport.is_authenticated()
                    except self._partial_auth_exceptions as exc:
                        auth_methods = list(exc.allowed_types)
                        authenticated = False

                if not authenticated and "keyboard-interactive" in auth_methods:
                    self.progress_write(f"[Sol remote] Waiting for interactive SSH authentication...")

                    def handler(title: str, instructions: str, prompt_list: list[tuple[str, bool]]) -> list[str]:
                        responses: list[str] = []
                        if title:
                            print(title)
                        if instructions:
                            print(instructions)
                        for prompt_text, _echo in prompt_list:
                            responses.append(self.prompt_response(prompt_text, config=cfg))
                        return responses

                    transport.auth_interactive(username, handler)
                    authenticated = transport.is_authenticated()

                if not authenticated:
                    raise RuntimeError(
                        "Paramiko could not authenticate to the Sol backend.\n"
                        f"Host: {username}@{hostname}:{port}\n"
                        f"Auth methods: {auth_methods}"
                    )

                self.progress_write("[Sol remote] SSH authentication complete.")
                connection = {
                    "transport": transport,
                    "sftp": None,
                    "hostname": hostname,
                    "port": port,
                    "username": username,
                }
                self.live_connections[cache_key] = connection
                self.authenticated_keys.add(cache_key)
                self.progress_write(f"[Sol remote] SSH session ready for {username}@{hostname}:{port}.")
                return connection
            except Exception as exc:
                last_exc = exc
                if transport is not None:
                    try:
                        transport.close()
                    except Exception:
                        pass
                if raw_sock is not None:
                    try:
                        raw_sock.close()
                    except Exception:
                        pass
                if attempt + 1 >= connect_retries or not connect_error_is_retryable(exc, paramiko_module=self.paramiko_module):
                    raise
                retry_sleep_s = min(backoff_s * float(attempt + 1), 5.0)
                self.progress_write(
                    "[Sol remote] Fresh SSH connect failed; retrying "
                    f"({attempt + 1}/{connect_retries - 1} retries used). "
                    f"Reason: {exc}"
                )
                if retry_sleep_s > 0:
                    self.sleep_fn(retry_sleep_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Paramiko connect failed without an exception")

    def run_shell(
        self,
        remote_shell_command: str,
        *,
        config: dict[str, Any] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one shell command over a persistent Paramiko transport."""
        cfg = self.config if config is None else config
        last_exc: Exception | None = None
        command_timeout_s = self.ssh_command_timeout_s_fn(cfg)
        exec_timeout_s = self.ssh_exec_timeout_s_fn(cfg)
        for attempt in range(2):
            connection = self.connect(config=cfg)
            transport = connection["transport"]
            channel = None
            try:
                channel = transport.open_session()
                channel.settimeout(1.0)
                exec_error: list[BaseException] = []

                def exec_target() -> None:
                    try:
                        channel.exec_command(f"bash -lc {shlex.quote(remote_shell_command)}")
                    except BaseException as exc:  # pragma: no cover - defensive thread bridge
                        exec_error.append(exc)

                exec_thread = self.thread_cls(target=exec_target, name="obgpu-paramiko-exec", daemon=True)
                exec_thread.start()
                exec_thread.join(exec_timeout_s)
                if exec_thread.is_alive():
                    try:
                        channel.close()
                    except Exception:
                        pass
                    raise SSHCommandTimeoutError(
                        "Paramiko exec_command acknowledgement timed out after "
                        f"{exec_timeout_s:.1f}s.\n"
                        f"Command: {remote_shell_command}"
                    )
                if exec_error:
                    raise exec_error[0]
                stdout_chunks: list[bytes] = []
                stderr_chunks: list[bytes] = []
                deadline = None if command_timeout_s is None else time.monotonic() + command_timeout_s
                while True:
                    while channel.recv_ready():
                        stdout_chunks.append(channel.recv(65536))
                    while channel.recv_stderr_ready():
                        stderr_chunks.append(channel.recv_stderr(65536))
                    if channel.exit_status_ready():
                        while channel.recv_ready():
                            stdout_chunks.append(channel.recv(65536))
                        while channel.recv_stderr_ready():
                            stderr_chunks.append(channel.recv_stderr(65536))
                        returncode = channel.recv_exit_status()
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        stdout_data = b"".join(stdout_chunks).decode("utf-8", errors="replace")
                        stderr_data = b"".join(stderr_chunks).decode("utf-8", errors="replace")
                        try:
                            channel.close()
                        except Exception:
                            pass
                        raise SSHCommandTimeoutError(
                            "Paramiko shell command timed out after "
                            f"{command_timeout_s:.1f}s.\n"
                            f"Command: {remote_shell_command}\n"
                            f"Stdout tail:\n{stdout_data[-2000:]}\n\n"
                            f"Stderr tail:\n{stderr_data[-2000:]}"
                        )
                    self.sleep_fn(0.05)
                stdout_data = b"".join(stdout_chunks).decode("utf-8", errors="replace")
                stderr_data = b"".join(stderr_chunks).decode("utf-8", errors="replace")
                return subprocess.CompletedProcess(
                    args=["paramiko", connection["hostname"], remote_shell_command],
                    returncode=returncode,
                    stdout=stdout_data,
                    stderr=stderr_data,
                )
            except SSHCommandTimeoutError:
                self.close_sftp(config=cfg)
                raise
            except Exception as exc:
                last_exc = exc
                self.close_sftp(config=cfg)
                if not self.transport_is_usable_fn(transport):
                    if (
                        bool(cfg.get("remote_preserve_paramiko_session", True))
                        and self.connection_key_fn(cfg) in self.authenticated_keys
                        and not self.can_reconnect_fn(cfg)
                    ):
                        raise RuntimeError(self.midrun_reauth_error_fn(cfg) + f"\nOriginal error: {exc}") from exc
                    self.drop_connection(config=cfg)
                if attempt == 0:
                    continue
                raise
            finally:
                if channel is not None:
                    channel.close()
        raise RuntimeError(f"Paramiko shell command failed unexpectedly: {last_exc}")
