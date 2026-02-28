"""llm-bench report — generate markdown benchmark report."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import click

from ..core.gpu import detect_gpu
from ..core.hardware import detect_cpu, detect_ram
from ..core.results import load_all


@click.command()
@click.option("--dir", "result_dir", default=None,
              help="Results directory (default: config output.dir)")
@click.option("--output", default=None,
              help="Output markdown path (default: stdout)")
@click.option("--title", default=None, help="Report title")
def report(result_dir, output, title):
    """Generate a markdown benchmark report from result files."""

    # 1. Load results
    if result_dir is None:
        result_dir = "./benchmarks/matrix"

    results = load_all(result_dir)
    if not results:
        raise click.ClickException(
            f"No result JSON files found in {result_dir}"
        )

    click.echo(f"Loaded {len(results)} result(s) from {result_dir}")

    # 2. Detect hardware
    gpu = detect_gpu()
    cpu = detect_cpu()
    ram = detect_ram()

    # 3. Generate markdown
    if title is None:
        title = "LLM Benchmark Report"

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # -- Hardware section --
    lines.append("## Hardware")
    lines.append("")
    lines.append("| Component | Details |")
    lines.append("|-----------|---------|")
    if gpu:
        lines.append(f"| GPU | {gpu.name} ({gpu.vram_total_mb} MB) |")
    else:
        lines.append("| GPU | Not detected |")
    lines.append(f"| CPU | {cpu.cores} cores ({cpu.model_name}) |")
    lines.append(f"| RAM | {ram.total_mb // 1024} GB |")
    lines.append("")

    # -- Speed results section --
    speed_results = [r for r in results if r.get("tg_benchmarks") or r.get("pp_benchmarks")]
    if speed_results:
        lines.append("## Speed Results")
        lines.append("")

        # Collect all PP lengths and TG workload keys across results
        all_pp_keys = _collect_keys(speed_results, "pp_benchmarks")
        all_tg_keys = _collect_keys(speed_results, "tg_benchmarks")

        # Build header
        header_parts = ["Config"]
        for pk in all_pp_keys:
            header_parts.append(f"PP-{pk}")
        for tk in all_tg_keys:
            header_parts.append(_format_tg_header(tk))
        header_parts.append("VRAM")

        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join(["--------"] * len(header_parts)) + "|")

        # Build rows
        for r in speed_results:
            label = r.get("label", "unknown")
            row = [label]

            for pk in all_pp_keys:
                val = r.get("pp_benchmarks", {}).get(pk, {})
                tps = val.get("mean_tps", 0)
                row.append(f"{tps:.1f}" if tps else "-")

            for tk in all_tg_keys:
                val = r.get("tg_benchmarks", {}).get(tk, {})
                tps = val.get("mean_tps", 0)
                row.append(f"{tps:.1f}" if tps else "-")

            vram = _extract_vram(r)
            row.append(f"{vram} MB" if vram else "-")

            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # -- Quality results section --
    quality_results = [r for r in results if _has_quality_data(r)]
    if quality_results:
        lines.append("## Quality Results")
        lines.append("")
        lines.append("| Model | PPL | KLD Mean | Same-top-p |")
        lines.append("|-------|-----|----------|------------|")

        for r in quality_results:
            label = r.get("label", "unknown")
            q = r.get("quality", {})
            ppl = q.get("ppl", q.get("perplexity"))
            kld = q.get("kld_mean", q.get("kld"))
            stp = q.get("same_top_p")

            ppl_str = f"{ppl:.4f}" if ppl is not None else "-"
            kld_str = f"{kld:.4f}" if kld is not None else "-"
            stp_str = f"{stp:.2f}%" if stp is not None else "-"

            lines.append(f"| {label} | {ppl_str} | {kld_str} | {stp_str} |")

        lines.append("")

    # -- Key findings section --
    lines.append("## Key Findings")
    lines.append("")

    if speed_results:
        best_tg = _find_best(speed_results, "tg_benchmarks")
        best_pp = _find_best(speed_results, "pp_benchmarks")
        lowest_vram = _find_lowest_vram(speed_results)

        if best_tg:
            lines.append(
                f"- Best TG throughput: {best_tg['label']} at "
                f"{best_tg['tps']:.1f} tok/s"
            )
        if best_pp:
            lines.append(
                f"- Best PP throughput: {best_pp['label']} at "
                f"{best_pp['tps']:.1f} tok/s"
            )
        if lowest_vram:
            lines.append(
                f"- Lowest VRAM: {lowest_vram['label']} at "
                f"{lowest_vram['vram']} MB"
            )
    else:
        lines.append("- No speed benchmark data found.")

    lines.append("")

    # 4. Write output
    md_text = "\n".join(lines)

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            f.write(md_text)
        click.echo(f"Report written to {output}")
    else:
        click.echo(md_text)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _collect_keys(results: list[dict], section: str) -> list[str]:
    """Collect all unique keys from a benchmark section, in sorted order."""
    keys: set[str] = set()
    for r in results:
        keys.update(r.get(section, {}).keys())
    # Sort PP keys numerically if possible, TG keys alphabetically
    def _sort_key(k: str):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)
    return sorted(keys, key=_sort_key)


def _format_tg_header(key: str) -> str:
    """Format a TG benchmark key for the table header."""
    mapping = {
        "short_prompt": "TG-Short",
        "medium_prompt": "TG-Med",
        "long_prompt": "TG-Long",
        "multi_turn": "TG-Multi",
        "short": "TG-Short",
        "medium": "TG-Med",
        "long": "TG-Long",
    }
    return mapping.get(key, f"TG-{key}")


def _extract_vram(result: dict) -> int | None:
    """Extract peak VRAM (MB) from a result dict."""
    gpu_stats = result.get("gpu_stats", {})
    # Prefer after_bench, fall back to after_load
    for phase in ("after_bench", "after_load"):
        snap = gpu_stats.get(phase)
        if isinstance(snap, dict) and "used_mb" in snap:
            return snap["used_mb"]
    return None


def _has_quality_data(result: dict) -> bool:
    """Check if a result contains quality (PPL/KLD) data."""
    q = result.get("quality", {})
    return bool(q.get("ppl") or q.get("perplexity") or q.get("kld_mean") or q.get("kld"))


def _find_best(results: list[dict], section: str) -> dict[str, Any] | None:
    """Find the result with the highest mean TPS in a section."""
    best_label = None
    best_tps = 0.0
    for r in results:
        benchmarks = r.get(section, {})
        for key, val in benchmarks.items():
            tps = val.get("mean_tps", 0)
            if tps > best_tps:
                best_tps = tps
                best_label = r.get("label", "unknown")
    if best_label is None:
        return None
    return {"label": best_label, "tps": best_tps}


def _find_lowest_vram(results: list[dict]) -> dict[str, Any] | None:
    """Find the result with the lowest VRAM usage."""
    best_label = None
    best_vram = float("inf")
    for r in results:
        vram = _extract_vram(r)
        if vram is not None and vram < best_vram:
            best_vram = vram
            best_label = r.get("label", "unknown")
    if best_label is None:
        return None
    return {"label": best_label, "vram": int(best_vram)}
