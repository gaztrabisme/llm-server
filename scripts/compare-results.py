#!/usr/bin/env python3
"""Parse and compare benchmark results from benchmarks/ directory.

Usage:
    python3 scripts/compare-results.py [benchmarks_dir]

Reads all JSON files from the benchmarks directory and produces a comparison
table showing: engine x config x workload → tok/s, VRAM usage.
"""

import json
import os
import sys
from pathlib import Path


def load_results(benchmarks_dir: str) -> list[dict]:
    """Load all JSON result files from the benchmarks directory."""
    results = []
    bdir = Path(benchmarks_dir)
    for f in sorted(bdir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
                data["_file"] = f.name
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)
    return results


def format_table(results: list[dict]) -> str:
    """Format results into a comparison table."""
    if not results:
        return "No benchmark results found."

    # Collect all workload names
    workloads = set()
    for r in results:
        workloads.update(r.get("workloads", {}).keys())
    workloads = sorted(workloads)

    # Header
    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK COMPARISON")
    lines.append("=" * 100)
    lines.append("")

    # Summary table
    header = f"{'Engine':<15} {'Config':<12} {'Threads':<8}"
    for w in workloads:
        header += f" {w:<20}"
    header += f" {'VRAM (MB)':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    best_per_workload: dict[str, tuple[float, str]] = {}

    for r in results:
        engine = r.get("engine", "?")
        config = r.get("config", "?")
        threads = r.get("thread_count", "?")
        vram = r.get("gpu_stats", {}).get("vram_info", "N/A")

        row = f"{engine:<15} {config:<12} {str(threads):<8}"

        for w in workloads:
            wdata = r.get("workloads", {}).get(w, {})
            mean = wdata.get("mean_tps", 0)
            std = wdata.get("stddev_tps", 0)
            if mean > 0:
                cell = f"{mean:.1f} ± {std:.1f}"
                key = w
                if key not in best_per_workload or mean > best_per_workload[key][0]:
                    best_per_workload[key] = (mean, f"{engine}/{config}")
            else:
                cell = "N/A"
            row += f" {cell:<20}"

        # Parse VRAM from CSV format "used, total, util%"
        vram_str = vram.split(",")[0].strip() if vram != "N/A" else "N/A"
        row += f" {vram_str:<12}"

        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("WINNERS (by mean tok/s, excluding warmup run)")
    lines.append("=" * 100)

    for w in workloads:
        if w in best_per_workload:
            tps, label = best_per_workload[w]
            lines.append(f"  {w:<20} → {label} ({tps:.1f} tok/s)")

    lines.append("")

    # Detailed per-run data
    lines.append("=" * 100)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 100)

    for r in results:
        engine = r.get("engine", "?")
        config = r.get("config", "?")
        lines.append(f"\n--- {engine} / {config} (threads={r.get('thread_count', '?')}) ---")
        lines.append(f"    File: {r.get('_file', '?')}")
        lines.append(f"    GPU: {r.get('gpu_stats', {}).get('vram_info', 'N/A')}")

        for w in workloads:
            wdata = r.get("workloads", {}).get(w, {})
            runs = wdata.get("runs", [])
            if not runs:
                continue
            lines.append(f"    {w}:")
            for run in runs:
                marker = " (warmup)" if run.get("run") == 1 else ""
                lines.append(
                    f"      Run {run.get('run', '?')}: "
                    f"{run.get('tokens_per_sec', 0):.1f} tok/s, "
                    f"{run.get('wall_ms', 0)}ms"
                    f"{marker}"
                )

    return "\n".join(lines)


def main():
    benchmarks_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks"

    if not os.path.isdir(benchmarks_dir):
        print(f"Directory not found: {benchmarks_dir}")
        sys.exit(1)

    results = load_results(benchmarks_dir)
    print(format_table(results))


if __name__ == "__main__":
    main()
