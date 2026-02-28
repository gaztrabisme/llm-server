"""Read and normalize old bench.sh JSON format to matrix format."""

from __future__ import annotations

import os
import re
from typing import Any


def is_legacy_format(data: dict[str, Any]) -> bool:
    """Check if a result dict is in old bench.sh format (not matrix format)."""
    return "matrix" not in data and "engine" in data


def normalize_legacy(data: dict[str, Any]) -> dict[str, Any]:
    """Convert old bench.sh format to matrix format."""
    label = data.get("label", data.get("config", "unknown"))
    # Clean up label
    for prefix in ("llama-cpp-", "ik-llama-"):
        if label.startswith(prefix):
            label = label[len(prefix):]
    # Strip timestamp suffix
    label = re.sub(r"-\d{8}-\d{6}$", "", label)

    # Extract matrix axes from label/config
    quant = _guess_quant(label, data)
    kv = _guess_kv(data)

    # Parse VRAM
    vram = _parse_legacy_vram(data)

    # Build normalized result
    result: dict[str, Any] = {
        "label": label,
        "matrix": {
            "quant": quant,
            "kv": kv,
            "batch": "none",
            "ctx": data.get("context_size", 65536),
        },
        "config_file": data.get("config", ""),
        "timestamp": data.get("timestamp", ""),
        "server_args": data.get("server_args", ""),
        "docker_image": data.get("docker_image", data.get("engine", "")),
        "thread_count": data.get("thread_count", 20),
        "gpu_stats": vram,
        "pp_benchmarks": {},
        "tg_benchmarks": {},
    }

    # Convert benchmarks
    if "benchmarks" in data:
        for bench in data["benchmarks"]:
            _convert_benchmark(result, bench)

    return result


def _guess_quant(label: str, data: dict) -> str:
    """Guess quantization from label or config."""
    upper = label.upper()
    if "UD-Q4_K_XL" in upper or "UD_Q4_K_XL" in upper:
        return "UD-Q4_K_XL"
    if "MXFP4" in upper:
        return "MXFP4_MOE"
    if "Q4_K_L" in upper:
        return "Q4_K_L"
    if "Q4_K_M" in upper:
        return "Q4_K_M"
    if "Q8_0" in upper:
        return "Q8_0"
    return "unknown"


def _guess_kv(data: dict) -> str:
    """Guess KV config from server args."""
    args = data.get("server_args", "")
    if "-ctk q8_0" in args:
        return "q8_0"
    if "-ctk q4_0" in args:
        if "-ctv q8_0" in args:
            return "asym"
        return "q4_0"
    return "f16"


def _parse_legacy_vram(data: dict) -> dict:
    """Parse VRAM from legacy format (CSV strings)."""
    stats: dict[str, Any] = {}
    for phase in ("after_load", "after_bench"):
        raw = data.get(f"vram_{phase}", data.get("gpu_stats", {}).get(phase))
        if isinstance(raw, str):
            try:
                parts = [p.strip() for p in raw.split(",")]
                stats[phase] = {
                    "used_mb": int(float(parts[0])),
                    "total_mb": int(float(parts[1])),
                    "utilization_pct": int(float(parts[2])) if len(parts) > 2 else 0,
                }
            except (ValueError, IndexError):
                pass
        elif isinstance(raw, dict):
            stats[phase] = raw
    return stats


def _convert_benchmark(result: dict, bench: dict) -> None:
    """Convert a single old-format benchmark entry to matrix format."""
    btype = bench.get("type", "")
    name = bench.get("name", "")

    if btype == "pp" or "prompt_processing" in name.lower():
        # PP benchmark
        tokens = bench.get("prompt_tokens", 0)
        key = str(tokens) if tokens else name
        result["pp_benchmarks"][key] = {
            "mean_tps": bench.get("mean_tps", bench.get("tokens_per_second", 0)),
            "stddev_tps": bench.get("stddev_tps", 0),
        }
    elif btype == "tg" or "generation" in name.lower():
        key = _normalize_workload_name(name)
        result["tg_benchmarks"][key] = {
            "mean_tps": bench.get("mean_tps", bench.get("tokens_per_second", 0)),
            "stddev_tps": bench.get("stddev_tps", 0),
        }


def _normalize_workload_name(name: str) -> str:
    """Normalize workload name to match matrix format keys."""
    lower = name.lower()
    if "short" in lower:
        return "short_prompt"
    if "medium" in lower:
        return "medium_prompt"
    if "long" in lower:
        return "long_prompt"
    if "multi" in lower:
        return "multi_turn"
    return name
