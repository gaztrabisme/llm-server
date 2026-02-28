"""Docker lifecycle management — start, stop, health check, logs."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from typing import Any

import click
import requests


_active_container: str | None = None


def _cleanup_handler():
    """Remove active container on exit."""
    if _active_container:
        subprocess.run(
            ["docker", "rm", "-f", _active_container],
            capture_output=True,
        )


atexit.register(_cleanup_handler)


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    _cleanup_handler()
    sys.exit(128 + signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def start_container(
    image: str,
    server_flags: list[str],
    port: int = 8080,
    model_dir: str = "./models",
    container_prefix: str = "llm-bench",
    extra_volumes: list[str] | None = None,
    verbose: bool = False,
) -> str:
    """Start a llama-server Docker container.

    Returns the container name.
    """
    global _active_container

    container_name = f"{container_prefix}-{os.getpid()}"

    # Remove any existing container with same name
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )

    model_dir = os.path.abspath(model_dir)

    cmd = [
        "docker", "run", "-d",
        "--gpus", "all",
        "--ipc", "host",
        "--name", container_name,
        "-p", f"{port}:{port}",
        "-v", f"{model_dir}:/models",
    ]

    if extra_volumes:
        for vol in extra_volumes:
            cmd.extend(["-v", vol])

    cmd.append(image)
    cmd.extend(server_flags)

    if verbose:
        click.echo(f"  Docker command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr.strip()}")

    _active_container = container_name
    return container_name


def stop_container(container_name: str) -> None:
    """Stop and remove a container."""
    global _active_container
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )
    if _active_container == container_name:
        _active_container = None


def wait_for_health(
    port: int = 8080,
    timeout: int = 600,
    interval: int = 5,
    verbose: bool = False,
) -> bool:
    """Poll /health endpoint until ready or timeout."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                if verbose:
                    click.echo(f"  Server healthy after {attempt * interval}s")
                return True
        except requests.ConnectionError:
            pass
        if verbose and attempt % 6 == 0:
            elapsed = attempt * interval
            click.echo(f"  Waiting for server... ({elapsed}s)")
        time.sleep(interval)

    return False


def check_port_available(port: int) -> bool:
    """Check if a port is available (not in use)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def get_container_logs(container_name: str, tail: int = 50) -> str:
    """Get recent logs from a container."""
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_name],
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def run_docker_command(
    image: str,
    entrypoint: str,
    args: list[str],
    model_dir: str = "./models",
    extra_volumes: list[str] | None = None,
    verbose: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a one-shot Docker command (e.g., llama-perplexity).

    Returns CompletedProcess with stdout/stderr.
    """
    model_dir = os.path.abspath(model_dir)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--ipc", "host",
        "--entrypoint", entrypoint,
        "-v", f"{model_dir}:/models",
    ]

    if extra_volumes:
        for vol in extra_volumes:
            cmd.extend(["-v", vol])

    cmd.append(image)
    cmd.extend(args)

    if verbose:
        click.echo(f"  Docker command: {' '.join(cmd)}")

    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd, text=True)
