#!/usr/bin/env python3
"""Minimal Docker CLI replacement that talks to Docker Engine API over TCP/Unix.

Used on HPC where no docker/podman binary is installed, only a TCP tunnel
to a remote Docker host (DOCKER_HOST=tcp://127.0.0.1:2375).

Only implements commands needed by mini-swe-agent's DockerEnvironment:
  run  -d --name NAME -w DIR [--rm] [-e K=V]... IMAGE CMD...
  exec [-w DIR] [-e K=V]... CONTAINER CMD...
  stop CONTAINER
  rm   [-f] CONTAINER

Environment:
  DOCKER_HOST  tcp://host:port  or  unix:///path/to/sock
"""

import http.client
import json
import os
import socket
import struct
import sys
import urllib.parse

API = "/v1.41"
_TIMEOUT = 600  # seconds; subprocess.run() in mini-swe-agent enforces its own timeout


# ── connection ──────────────────────────────────────────────────────────────

class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection over a Unix domain socket."""

    def __init__(self, socket_path, timeout=_TIMEOUT):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._socket_path)


def _conn():
    h = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
    if h.startswith("tcp://"):
        hp = h[6:].split(":")
        return http.client.HTTPConnection(hp[0], int(hp[1]) if len(hp) > 1 else 2375, timeout=_TIMEOUT)
    if h.startswith("unix://"):
        return _UnixHTTPConnection(h[7:])
    _die(f"Unsupported DOCKER_HOST={h}")


def _api(method, path, body=None):
    """Send a request to the Docker Engine API and return (status, bytes)."""
    c = _conn()
    hdrs = {}
    raw = None
    if body is not None:
        raw = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    c.request(method, API + path, body=raw, headers=hdrs)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def _die(msg, code=1):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


# ── Docker stream demux ────────────────────────────────────────────────────

def _demux(raw: bytes) -> str:
    """Parse Docker multiplexed stream (8-byte header per frame)."""
    parts = []
    i = 0
    while i + 8 <= len(raw):
        size = struct.unpack(">I", raw[i + 4 : i + 8])[0]
        i += 8
        end = min(i + size, len(raw))
        parts.append(raw[i:end].decode("utf-8", errors="replace"))
        i = end
    if not parts:
        # No valid frames — return raw (TTY mode or empty)
        return raw.decode("utf-8", errors="replace")
    return "".join(parts)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_run(args):
    """docker run -d --name NAME -w DIR [--rm] [-e K=V]... IMAGE CMD..."""
    name = workdir = None
    auto_rm = False
    envs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-d":
            i += 1
        elif a == "--name" and i + 1 < len(args):
            name = args[i + 1]; i += 2
        elif a == "-w" and i + 1 < len(args):
            workdir = args[i + 1]; i += 2
        elif a == "--rm":
            auto_rm = True; i += 1
        elif a == "-e" and i + 1 < len(args):
            envs.append(args[i + 1]); i += 2
        elif a.startswith("-"):
            i += 1  # skip unknown flag
        else:
            break
    if i >= len(args):
        _die("docker run: missing IMAGE")
    image = args[i]
    command = args[i + 1 :]

    body = {
        "Image": image,
        "Cmd": command,
        "Tty": False,
        "OpenStdin": False,
        "HostConfig": {"AutoRemove": auto_rm},
    }
    if workdir:
        body["WorkingDir"] = workdir
    if envs:
        body["Env"] = envs

    qs = f"?name={urllib.parse.quote(name)}" if name else ""
    st, data = _api("POST", f"/containers/create{qs}", body)

    # Image not found locally — pull it, then retry create
    if st == 404:
        _pull(image)
        st, data = _api("POST", f"/containers/create{qs}", body)

    if st not in (200, 201):
        _die(f"create failed ({st}): {data.decode(errors='replace')}")

    cid = json.loads(data)["Id"]

    st2, d2 = _api("POST", f"/containers/{cid}/start")
    if st2 not in (200, 204, 304):
        _die(f"start failed ({st2}): {d2.decode(errors='replace')}")

    # docker run -d prints container ID
    sys.stdout.write(cid + "\n")


def _pull(image):
    """Pull an image from registry (blocking)."""
    if ":" in image.rsplit("/", 1)[-1]:
        from_img, tag = image.rsplit(":", 1)
    else:
        from_img, tag = image, "latest"

    sys.stderr.write(f"Pulling {from_img}:{tag} ...\n")
    qs = f"?fromImage={urllib.parse.quote(from_img)}&tag={urllib.parse.quote(tag)}"
    st, data = _api("POST", f"/images/create{qs}")
    if st != 200:
        _die(f"pull failed ({st}): {data.decode(errors='replace')}")
    sys.stderr.write("Pull complete.\n")


def cmd_exec(args):
    """docker exec [-w DIR] [-e K=V]... CONTAINER CMD..."""
    workdir = None
    envs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-w" and i + 1 < len(args):
            workdir = args[i + 1]; i += 2
        elif a == "-e" and i + 1 < len(args):
            envs.append(args[i + 1]); i += 2
        elif a.startswith("-"):
            i += 1
        else:
            break
    if i >= len(args):
        _die("docker exec: missing CONTAINER")
    container = args[i]
    command = args[i + 1 :]

    body = {"AttachStdout": True, "AttachStderr": True, "Tty": False, "Cmd": command}
    if workdir:
        body["WorkingDir"] = workdir
    if envs:
        body["Env"] = envs

    st, data = _api("POST", f"/containers/{container}/exec", body)
    if st not in (200, 201):
        _die(f"exec create failed ({st}): {data.decode(errors='replace')}")
    eid = json.loads(data)["Id"]

    st2, raw = _api("POST", f"/exec/{eid}/start", {"Detach": False, "Tty": False})
    sys.stdout.write(_demux(raw))
    sys.stdout.flush()

    # Retrieve exit code of the executed command
    st3, info = _api("GET", f"/exec/{eid}/json")
    if st3 == 200:
        ec = json.loads(info).get("ExitCode", 0)
        if ec:
            sys.exit(ec)


def cmd_stop(args):
    """docker stop CONTAINER"""
    cid = args[-1]
    _api("POST", f"/containers/{cid}/stop")


def cmd_rm(args):
    """docker rm [-f] CONTAINER"""
    force = "-f" in args
    cid = args[-1]
    qs = "?force=true" if force else ""
    _api("DELETE", f"/containers/{cid}{qs}")


# ── main ────────────────────────────────────────────────────────────────────

_CMDS = {"run": cmd_run, "exec": cmd_exec, "stop": cmd_stop, "rm": cmd_rm}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        _die(f"Usage: {sys.argv[0]} <{'|'.join(_CMDS)}> ...")
    try:
        _CMDS[sys.argv[1]](sys.argv[2:])
    except ConnectionRefusedError:
        _die(f"Cannot connect to Docker daemon at {os.environ.get('DOCKER_HOST', '(unset)')}")
    except Exception as e:
        _die(f"{type(e).__name__}: {e}")
