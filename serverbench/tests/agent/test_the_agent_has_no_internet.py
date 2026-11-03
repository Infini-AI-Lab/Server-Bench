"""The agent's container has no route to the internet, only to the broker.

The agent needs its model endpoint, and nothing else outside the container.
`egress_broker.py` sits on the host and tunnels only to allowed hosts; the
container's allowlist names ONE address -- the docker bridge gateway, which is
the host and not the internet -- and `restrict-network.sh` REJECTs everything
else.

What is checked here is the broker's own behaviour, which needs no container and
no GPU: that it tunnels to an allowed host, refuses everything else, refuses to
be used as anything but a tunnel, and writes down every attempt. Whether the
firewall then admits only the broker is a property of a running container and is
asserted by the egress probes on a real run.
"""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading

import pytest
from paths import SERVERBENCH

sys.path.insert(0, str(SERVERBENCH / "scripts" / "agent"))

import egress_broker as eb  # noqa: E402


@pytest.fixture
def upstream():
    """A stand-in for the model endpoint: accepts, echoes, closes."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    seen: list[bytes] = []

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            data = conn.recv(1024)
            seen.append(data)
            conn.sendall(b"pong:" + data)
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()[1], seen
    server.close()


@pytest.fixture
def broker(tmp_path, upstream, monkeypatch):
    """The broker, with the allowlist pointed at the local stand-in."""
    port, _ = upstream
    monkeypatch.setattr(eb, "ALLOWED", frozenset({"localhost"}))
    # The port policy is 443 in production; the stand-in listens on an
    # ephemeral one, so the POLICY is patched rather than the check weakened.
    monkeypatch.setattr(eb, "ALLOWED_PORTS", frozenset({port}))
    log = tmp_path / "egress.jsonl"
    server = eb.serve("127.0.0.1", 0, log)
    yield server.server_address[1], log, port
    server.shutdown()


def _speak(port: int, request: bytes, then: bytes = b"") -> bytes:
    sock = socket.create_connection(("127.0.0.1", port), 5)
    sock.settimeout(5)
    sock.sendall(request)
    head = sock.recv(4096)
    if then and b"200" in head:
        sock.sendall(then)
        with contextlib.suppress(TimeoutError, OSError):
            head += sock.recv(4096)
    sock.close()
    return head


def _lines(log, *, expect=1, timeout=5.0):
    """The broker records from its handler thread, which finishes after the
    client socket closes -- so a reader that does not wait sees a short file."""
    import time
    deadline = time.time() + timeout
    while True:
        if log.exists():
            found = [json.loads(line) for line in
                     log.read_text("utf-8").splitlines() if line.strip()]
            if len(found) >= expect or time.time() > deadline:
                return found
        elif time.time() > deadline:
            return []
        time.sleep(0.05)


def test_an_allowed_host_is_tunnelled(broker, upstream):
    port, _, up = broker
    got = _speak(port, f"CONNECT localhost:{up} HTTP/1.1\r\n\r\n".encode(),
                 then=b"hello")
    assert b"200 Connection established" in got
    assert b"pong:hello" in got, "the tunnel did not carry the payload"


@pytest.mark.parametrize("host", ["pypi.org", "github.com", "huggingface.co",
                                  "download.pytorch.org", "registry.npmjs.org",
                                  "evil.example.com"])
def test_a_host_that_is_not_allowed_is_refused(broker, host):
    port, log, _ = broker
    got = _speak(port, f"CONNECT {host}:443 HTTP/1.1\r\n\r\n".encode())
    assert b"403" in got, f"{host} was not refused"
    assert any(e["host"] == host and e["allowed"] is False for e in _lines(log))


def test_it_will_not_fetch_a_url_for_anybody(broker):
    """A proxy that will GET is a proxy that will fetch a wheel."""
    port, _, _ = broker
    got = _speak(port, b"GET https://pypi.org/simple/vllm/ HTTP/1.1\r\n\r\n")
    assert b"405" in got, "the broker answered a non-CONNECT verb"


def test_a_port_the_policy_does_not_name_is_refused(broker):
    """A tunnel to port 22 on an allowed host is still somewhere nobody
    authorised."""
    port, _, _ = broker
    got = _speak(port, b"CONNECT localhost:22 HTTP/1.1\r\n\r\n")
    assert b"403" in got


def test_production_policy_is_https_only():
    assert frozenset({443}) == eb.ALLOWED_PORTS


def test_every_attempt_is_written_down(broker, upstream):
    """A rule that is never exercised looks identical to one never needed."""
    port, log, up = broker
    _speak(port, f"CONNECT localhost:{up} HTTP/1.1\r\n\r\n".encode(), then=b"hi")
    _speak(port, b"CONNECT pypi.org:443 HTTP/1.1\r\n\r\n")
    entries = _lines(log, expect=2)
    assert {e["allowed"] for e in entries} == {True, False}
    allowed = next(e for e in entries if e["allowed"])
    assert allowed["sent"] > 0 and allowed["received"] > 0, (
        "an allowed tunnel must record how much crossed it")


def test_the_allowlist_is_small_and_is_the_model_endpoint():
    """No wildcard, no suffix match, nothing addable from the environment."""
    assert "api.anthropic.com" in eb.ALLOWED
    assert all("*" not in host for host in eb.ALLOWED)
    for banned in ("pypi.org", "github.com", "huggingface.co",
                   "download.pytorch.org"):
        assert banned not in eb.ALLOWED


def test_codex_has_its_own_small_model_and_auth_allowlist():
    allowed = eb.PROVIDER_HOSTS["codex"]
    assert {"chatgpt.com", "auth.openai.com", "api.openai.com"} <= allowed
    assert "api.anthropic.com" not in allowed
    assert all("*" not in host for host in allowed)


def test_a_giant_request_head_does_not_hang_the_broker(broker):
    """The head is attacker-controlled; an unbounded read is a way to make the
    broker the thing that falls over."""
    port, _, _ = broker
    sock = socket.create_connection(("127.0.0.1", port), 5)
    sock.settimeout(10)
    sock.sendall(b"CONNECT " + b"a" * 20000)
    with contextlib.suppress(TimeoutError, OSError):
        sock.recv(100)
    sock.close()
    # Still serving afterwards.
    assert b"403" in _speak(port, b"CONNECT pypi.org:443 HTTP/1.1\r\n\r\n")
