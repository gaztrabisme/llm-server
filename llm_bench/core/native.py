"""Native subprocess lifecycle — start, stop, health check for llama-server."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from typing import Any

import click


_active_process: subprocess.Popen | None = None


def _cleanup_handler():
    """Terminate active native server on exit."""
    global _active_process
    if _active_process and _active_process.poll() is None:
        _active_process.terminate()
        try:
            _active_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _active_process.kill()
            _active_process.wait(timeout=5)
        _active_process = None


atexit.register(_cleanup_handler)


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    _cleanup_handler()
    sys.exit(128 + signum)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def start_native_server(
    server_bin: str,
    server_flags: list[str],
    verbose: bool = False,
) -> subprocess.Popen:
    """Start llama-server as a native subprocess.

    Returns the Popen handle.
    """
    global _active_process

    cmd = [server_bin] + server_flags

    if verbose:
        click.echo(f"  Native command: {' '.join(cmd)}")

    # Send stdout/stderr to devnull unless verbose
    if verbose:
        proc = subprocess.Popen(cmd)
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Check it didn't immediately fail
    time.sleep(0.5)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Native server exited immediately (code {proc.returncode}). "
            f"Command: {' '.join(cmd)}"
        )

    _active_process = proc
    return proc


def stop_native_server(proc: subprocess.Popen) -> None:
    """Stop a native server subprocess."""
    global _active_process

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    if _active_process is proc:
        _active_process = None


def run_native_command(
    binary: str,
    args: list[str],
    verbose: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a one-shot native command (e.g., llama-perplexity).

    Returns CompletedProcess with stdout/stderr.
    """
    cmd = [binary] + args

    if verbose:
        click.echo(f"  Native command: {' '.join(cmd)}")

    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd, text=True)


def native_server_flags(
    server_args: dict[str, Any],
    model_path: str | None = None,
) -> list[str]:
    """Convert server_args to native CLI flags.

    Unlike Docker mode, native mode uses real filesystem paths directly
    (no /models/ prefix). If model_path is given, it overrides the model
    arg in server_args with the real path.
    """
    from .config import server_args_to_flags

    # Make a copy so we don't mutate the original
    args = dict(server_args)
    if model_path is not None:
        args["model"] = model_path

    return server_args_to_flags(args)
