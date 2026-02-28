"""Result JSON I/O and schema — backward compatible with bench-matrix.sh output."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def create_result(
    label: str,
    matrix: dict[str, Any],
    config_file: str = "",
    server_args: str = "",
    docker_image: str = "",
    thread_count: int = 20,
) -> dict[str, Any]:
    """Create a new result dict with metadata."""
    return {
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
