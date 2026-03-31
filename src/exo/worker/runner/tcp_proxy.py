"""Local TCP proxy for runner subprocesses.

On macOS, subprocesses spawned via ``posix_spawn`` (the ``"spawn"``
multiprocessing start method) cannot reach LAN devices — macOS returns
EHOSTUNREACH.  This proxy runs as daemon threads in the **main** exo
process (which CAN access the LAN) and relays connections from the
subprocess via a Unix domain socket.

Protocol:
  1. Subprocess connects to the Unix socket
  2. Subprocess sends ``host:port\\n`` as the first line
  3. Proxy connects to ``(host, port)`` via TCP
  4. Bidirectional relay until either side closes
"""

from __future__ import annotations

import os
import socket
import threading
import uuid

from loguru import logger

_RELAY_BUF = 256 * 1024  # 256 KB relay buffer


def _relay(src: socket.socket, dst: socket.socket, label: str) -> None:
    """Copy data from *src* to *dst* until EOF or error."""
    try:
        while True:
            data = src.recv(_RELAY_BUF)
            if not data:
                break
            dst.sendall(data)
    except (OSError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class TcpProxyServer:
    """Unix-domain-socket proxy that relays to TCP endpoints."""

    def __init__(self) -> None:
        self._sock_path = f"/tmp/exo-proxy-{uuid.uuid4().hex[:12]}.sock"
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def socket_path(self) -> str:
        return self._sock_path

    def start(self) -> None:
        """Start the proxy accept loop as a daemon thread."""
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self._sock_path)
        self._server.listen(8)
        self._server.settimeout(1.0)  # allow periodic stop checks

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info(f"TCP proxy started at {self._sock_path}")

    def stop(self) -> None:
        """Stop the proxy and clean up the socket file."""
        self._stop.set()
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(3)
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_client, args=(client,), daemon=True
            ).start()

    def _handle_client(self, client: socket.socket) -> None:
        """Read target header, connect to remote, relay bidirectionally."""
        remote: socket.socket | None = None
        try:
            # Read header: "host:port\n"
            header_buf = b""
            while b"\n" not in header_buf:
                chunk = client.recv(1024)
                if not chunk:
                    return
                header_buf += chunk

            header_line, remainder = header_buf.split(b"\n", 1)
            target = header_line.decode("utf-8").strip()
            host, port_str = target.rsplit(":", 1)
            port = int(port_str)

            logger.info(f"TCP proxy: connecting to {host}:{port}")
            remote = socket.create_connection((host, port), timeout=30)
            remote.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            remote.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

            # Forward any data that arrived after the header
            if remainder:
                remote.sendall(remainder)

            # Bidirectional relay
            t1 = threading.Thread(
                target=_relay, args=(client, remote, "client→remote"), daemon=True
            )
            t2 = threading.Thread(
                target=_relay, args=(remote, client, "remote→client"), daemon=True
            )
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        except Exception as e:
            logger.warning(f"TCP proxy error: {e}")
        finally:
            try:
                client.close()
            except OSError:
                pass
            if remote:
                try:
                    remote.close()
                except OSError:
                    pass
