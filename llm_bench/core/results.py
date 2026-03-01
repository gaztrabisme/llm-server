"""Result JSON I/O and schema — backward compatible with bench-matrix.sh output."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def detect_hardware() -> dict[str, Any]:
    """Detect hardware fingerprint for embedding in result JSON.

    Returns a dict like:
        {"gpu": "NVIDIA GeForce RTX 5080", "gpu_vram_mb": 16303, "ram_gb": 128, "cpu_cores": 32}

    Any field that can't be detected is omitted.
    """
    from .gpu import detect_gpu
    from .hardware import detect_cpu, detect_ram

    hw: dict[str, Any] = {}

    gpu = detect_gpu()
    if gpu is not None:
        hw["gpu"] = gpu.name
        hw["gpu_vram_mb"] = gpu.vram_total_mb

    ram = detect_ram()
    if ram.total_mb > 0:
        hw["ram_gb"] = ram.total_mb // 1024

    cpu = detect_cpu()
    if cpu.cores > 0:
        hw["cpu_cores"] = cpu.cores

    return hw


def create_result(
    label: str,
    matrix: dict[str, Any],
    config_file: str = "",
    server_args: str = "",
    docker_image: str = "",
    thread_count: int = 20,
) -> dict[str, Any]:
    """Create a new result dict with metadata."""
    result: dict[str, Any] = {
        "label": label,
        "matrix": matrix,
        "config_file": config_file,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "server_args": server_args,
        "docker_image": docker_image,
        "thread_count": thread_count,
        "gpu_stats": {},
        "pp_benchmarks": {},
        "tg_benchmarks": {},
        "version": "llm-bench-0.1.0",
    }

    hw = detect_hardware()
    if hw:
        result["hardware"] = hw

    return result


def save_result(result: dict[str, Any], output_dir: str = "./benchmarks/matrix") -> str:
    """Save result to JSON file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    label = result.get("label", "benchmark")
    timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d-%H%M%S"))
    filename = f"{label}-{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    return filepath


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a single result JSON file."""
    with open(path) as f:
        return json.load(f)


def load_all(directory: str = "./benchmarks/matrix") -> list[dict[str, Any]]:
    """Load all result JSON files from a directory."""
    results = []
    dirpath = Path(directory)
    if not dirpath.exists():
        return results
    for path in sorted(dirpath.glob("*.json")):
        try:
            results.append(load_result(path))
        except (json.JSONDecodeError, KeyError):
            continue
    return results
