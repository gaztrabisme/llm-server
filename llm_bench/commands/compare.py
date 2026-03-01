"""Compare benchmark results — port of compare-matrix.py to Click CLI."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from ..core.results import load_result, load_all
from ..compat.legacy_results import is_legacy_format, normalize_legacy


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

PP_KEYS = ["512", "1024", "4096", "16384"]
TG_SOURCE_KEYS = ["short_prompt", "medium_prompt", "long_prompt", "multi_turn"]
TG_DISPLAY_KEYS = ["tg_short", "tg_medium", "tg_long", "tg_multi"]

# Map from internal source key to display key
_TG_KEY_MAP = {
    "short_prompt": "tg_short",
    "medium_prompt": "tg_medium",
    "long_prompt": "tg_long",
    "multi_turn": "tg_multi",
}

COLUMN_NAMES = {
    "pp_512": "PP-512",
    "pp_1024": "PP-1024",
    "pp_4096": "PP-4096",
    "pp_16384": "PP-16k",
    "tg_short": "TG-Short",
    "tg_medium": "TG-Med",
    "tg_long": "TG-Long",
    "tg_multi": "TG-Multi",
}

COMPACT_COLUMN_NAMES = {
    "pp_512": "PP5",
    "pp_1024": "PP1k",
    "pp_4096": "PP4k",
    "pp_16384": "PP16k",
    "tg_short": "TGs",
    "tg_medium": "TGm",
    "tg_long": "TGl",
    "tg_multi": "TGmt",
}


# ---------------------------------------------------------------------------
# Normalize raw result data into flat comparison records
# ---------------------------------------------------------------------------

def _normalize_record(data: dict[str, Any], source_path: str = "") -> dict[str, Any]:
    """Normalize a raw result dict (matrix or legacy) into a flat comparison record."""
    label = data.get("label", "")
    matrix = data.get("matrix", {})

    record: dict[str, Any] = {
        "label": label,
        "quant": matrix.get("quant", "-"),
        "kv": matrix.get("kv", "-"),
        "batch": matrix.get("batch", "-"),
        "ctx": matrix.get("ctx", 0),
        "pp_512": None,
        "pp_1024": None,
        "pp_4096": None,
        "pp_16384": None,
        "tg_short": None,
        "tg_medium": None,
        "tg_long": None,
        "tg_multi": None,
        "tg_mean": 0.0,
        "vram_used": None,
        "vram_total": None,
        "_source": source_path,
        "_hardware_gpu": None,
    }

    # Hardware fingerprint (optional, may be absent in old results)
    hardware = data.get("hardware", {})
    if isinstance(hardware, dict) and hardware.get("gpu"):
        record["_hardware_gpu"] = hardware["gpu"]

    # PP benchmarks
    pp_benchmarks = data.get("pp_benchmarks", {})
    for pk in PP_KEYS:
        bench = pp_benchmarks.get(pk)
        if bench and bench.get("mean_tps") is not None:
            record[f"pp_{pk}"] = {
                "mean": bench["mean_tps"],
                "stddev": bench.get("stddev_tps"),
            }

    # TG benchmarks
    tg_benchmarks = data.get("tg_benchmarks", {})
    tg_means = []
    for src_key, disp_key in _TG_KEY_MAP.items():
        bench = tg_benchmarks.get(src_key)
        if bench and bench.get("mean_tps") is not None:
            record[disp_key] = {
                "mean": bench["mean_tps"],
                "stddev": bench.get("stddev_tps"),
            }
            tg_means.append(bench["mean_tps"])

    record["tg_mean"] = sum(tg_means) / len(tg_means) if tg_means else 0.0

    # VRAM
    gpu_stats = data.get("gpu_stats", {})
    after_load = gpu_stats.get("after_load", {})
    if isinstance(after_load, dict):
        if "used_mb" in after_load:
            record["vram_used"] = after_load["used_mb"]
        if "total_mb" in after_load:
            record["vram_total"] = after_load["total_mb"]

    return record


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_and_normalize(filepath: Path) -> dict[str, Any] | None:
    """Load a JSON file, detect format, normalize if legacy, and return a record."""
    try:
        data = load_result(filepath)
    except (json.JSONDecodeError, IOError, KeyError) as exc:
        click.echo(f"Warning: skipping {filepath.name}: {exc}", err=True)
        return None

    # Skip comparison output files
    if filepath.name.startswith("comparison-"):
        return None

    if is_legacy_format(data):
        data = normalize_legacy(data)

    return _normalize_record(data, str(filepath))


def _load_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Load specific files and return normalized records."""
    records = []
    for p in paths:
        rec = _load_and_normalize(p)
        if rec is not None:
            records.append(rec)
    return records


def _load_dir(directory: Path, include_legacy: bool) -> list[dict[str, Any]]:
    """Load all JSON files from the results directory (and optionally legacy dir)."""
    records = []

    # Matrix-format results
    if directory.is_dir():
        for f in sorted(directory.glob("*.json")):
            rec = _load_and_normalize(f)
            if rec is not None:
                records.append(rec)

    # Legacy results from benchmarks/ (parent of matrix/)
    if include_legacy:
        legacy_dir = directory.parent if directory.name == "matrix" else directory
        if legacy_dir.is_dir() and legacy_dir != directory:
            for f in sorted(legacy_dir.glob("*.json")):
                rec = _load_and_normalize(f)
                if rec is not None:
                    records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def _sort_key(record: dict[str, Any], sort_by: str):
    """Return a sort key for the given record and sort field."""
    if sort_by == "label":
        return record["label"]
    elif sort_by == "tg_mean":
        return -(record.get("tg_mean") or 0.0)
    elif sort_by == "pp_512":
        pp = record.get("pp_512")
        val = pp["mean"] if pp else 0.0
        return -val
    elif sort_by == "pp_1024":
        pp = record.get("pp_1024")
        val = pp["mean"] if pp else 0.0
        return -val
    elif sort_by == "vram":
        return record.get("vram_used") or 99999
    else:
        return record["label"]


# ---------------------------------------------------------------------------
# Winner computation
# ---------------------------------------------------------------------------

# All metric columns where highest is best
_HIGHER_IS_BETTER = (
    [f"pp_{k}" for k in PP_KEYS] + TG_DISPLAY_KEYS
)


def _compute_winners(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the best value per metric across all records.

    PP and TG: highest mean wins.
    VRAM: lowest used wins.
    """
    winners: dict[str, Any] = {}

    for col in _HIGHER_IS_BETTER:
        best_val = None
        best_label = None
        for rec in records:
            entry = rec.get(col)
            if entry is None:
                continue
            val = entry["mean"]
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_label = rec["label"]
        if best_val is not None:
            winners[col] = {"label": best_label, "value": best_val}

    # VRAM: lowest is best
    best_vram = None
    best_label = None
    for rec in records:
        val = rec.get("vram_used")
        if val is not None and (best_vram is None or val < best_vram):
            best_vram = val
            best_label = rec["label"]
    if best_vram is not None:
        winners["vram"] = {"label": best_label, "value": best_vram}

    return winners


# ---------------------------------------------------------------------------
# Value formatting helpers
# ---------------------------------------------------------------------------

def _fmt_val(entry: dict[str, Any] | None, is_winner: bool = False) -> str:
    """Format a metric entry as 'mean (+-stddev)' with optional '*' marker."""
    if entry is None:
        return "-"
    mean = entry.get("mean")
    if mean is None:
        return "-"
    stddev = entry.get("stddev")
    suffix = "*" if is_winner else ""
    if stddev is not None and stddev > 0:
        return f"{mean:.1f} (\u00b1{stddev:.1f}){suffix}"
    return f"{mean:.1f}{suffix}"


def _fmt_val_csv(entry: dict[str, Any] | None) -> str:
    """Format a metric entry for CSV (mean only)."""
    if entry is None:
        return ""
    mean = entry.get("mean")
    if mean is None:
        return ""
    return f"{mean:.1f}"


# ---------------------------------------------------------------------------
# Determine visible columns
# ---------------------------------------------------------------------------

def _visible_metric_cols(term_width: int, pp_detail: bool = False) -> list[str]:
    """Return the list of metric column keys to display based on terminal width.

    Default: PP-512 + all TG columns.
    --pp-detail: all PP lengths + all TG columns.
    """
    tg_cols = list(TG_DISPLAY_KEYS)

    if pp_detail:
        pp_cols = [f"pp_{k}" for k in PP_KEYS]
        return pp_cols + tg_cols

    if term_width < 100:
        # Very narrow: TG only
        return tg_cols

    # Default: PP-512 + TG
    return ["pp_512"] + tg_cols


def _col_name(key: str, compact: bool) -> str:
    """Return column display name."""
    if compact:
        return COMPACT_COLUMN_NAMES.get(key, key)
    return COLUMN_NAMES.get(key, key)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _has_mixed_hardware(records: list[dict[str, Any]]) -> bool:
    """Check if records come from different machines (different GPU names)."""
    gpus = {r["_hardware_gpu"] for r in records if r.get("_hardware_gpu")}
    return len(gpus) > 1


def _short_gpu_name(name: str | None) -> str:
    """Shorten GPU name for display, e.g. 'NVIDIA GeForce RTX 5080' -> 'RTX 5080'."""
    if not name:
        return "-"
    # Strip common prefixes
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _format_table(records: list[dict[str, Any]], winners: dict[str, Any],
                  group_by: str | None, pp_detail: bool = False) -> str:
    """Format records into an ASCII comparison table."""
    if not records:
        return "No benchmark results found."

    if group_by:
        return _format_grouped_tables(records, winners, group_by, pp_detail=pp_detail)

    return _format_single_table(records, winners, omit_axis=None, pp_detail=pp_detail)


def _format_single_table(records: list[dict[str, Any]], winners: dict[str, Any],
                          omit_axis: str | None, pp_detail: bool = False) -> str:
    """Render one ASCII table."""
    term_width = shutil.get_terminal_size((120, 40)).columns
    compact = term_width < 160
    metric_cols = _visible_metric_cols(term_width, pp_detail=pp_detail)
    show_hw = _has_mixed_hardware(records)

    # Matrix axis columns (omit the grouped one)
    axis_defs = [("quant", "Quant"), ("kv", "KV"), ("batch", "Batch"), ("ctx", "Ctx")]
    axis_cols = [(key, name) for key, name in axis_defs if key != omit_axis]

    # Build header
    header: list[str] = ["Label"]
    if show_hw:
        header.append("Hardware")
    for _, name in axis_cols:
        header.append(name)
    for mc in metric_cols:
        header.append(_col_name(mc, compact))
    header.append("VRAM")

    # Build rows
    rows: list[list[str]] = []
    for rec in records:
        row = [rec["label"]]
        if show_hw:
            row.append(_short_gpu_name(rec.get("_hardware_gpu")))
        for key, _ in axis_cols:
            row.append(str(rec.get(key, "-")))
        for mc in metric_cols:
            is_winner = (mc in winners and winners[mc]["label"] == rec["label"]
                         and rec.get(mc) is not None)
            row.append(_fmt_val(rec.get(mc), is_winner))
        # VRAM column
        vram = rec.get("vram_used")
        vram_winner = ("vram" in winners and winners["vram"]["label"] == rec["label"]
                       and vram is not None)
        if vram is not None:
            row.append(f"{vram}{'*' if vram_winner else ''}")
        else:
            row.append("-")
        rows.append(row)

    # Compute column widths
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    # Render
    lines: list[str] = []
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    sep_line = "\u2500".join("\u2500" * (widths[i] + 2) for i in range(len(header)))
    # Replace first and last padding
    sep_parts = ["\u2500" * widths[i] for i in range(len(header))]
    sep_line = "\u2500\u253c\u2500".join(sep_parts)

    lines.append(header_line)
    lines.append(sep_line)

    for row in rows:
        line = " | ".join(row[i].ljust(widths[i]) for i in range(len(header)))
        lines.append(line)

    # Winners footer
    lines.append("")
    winner_parts = []
    for mc in metric_cols:
        if mc in winners:
            w = winners[mc]
            winner_parts.append(
                f"{_col_name(mc, compact)}: {w['label']} ({w['value']:.1f})"
            )
    if "vram" in winners:
        w = winners["vram"]
        winner_parts.append(f"VRAM: {w['label']} ({w['value']} MB)")

    if winner_parts:
        lines.append("WINNERS: " + " | ".join(winner_parts))

    return "\n".join(lines)


def _format_grouped_tables(records: list[dict[str, Any]], winners: dict[str, Any],
                           group_by: str, pp_detail: bool = False) -> str:
    """Format records as multiple tables grouped by a matrix axis."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = str(rec.get(group_by, "-"))
        groups.setdefault(key, []).append(rec)

    sections: list[str] = []
    for group_val in sorted(groups.keys()):
        group_records = groups[group_val]
        group_winners = _compute_winners(group_records)
        header = f"=== {group_by.capitalize()}: {group_val} ==="
        table = _format_single_table(group_records, group_winners, omit_axis=group_by,
                                     pp_detail=pp_detail)
        sections.append(f"{header}\n{table}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------

def _format_csv(records: list[dict[str, Any]], pp_detail: bool = False) -> str:
    """Format records as CSV with mean values only."""
    if pp_detail:
        metric_cols = [f"pp_{k}" for k in PP_KEYS] + list(TG_DISPLAY_KEYS)
    else:
        metric_cols = ["pp_512"] + list(TG_DISPLAY_KEYS)
    show_hw = _has_mixed_hardware(records)

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header
    header = ["Label"]
    if show_hw:
        header.append("Hardware")
    header.extend(["Quant", "KV", "Batch", "Ctx"])
    for mc in metric_cols:
        header.append(COLUMN_NAMES.get(mc, mc))
    header.append("VRAM")
    writer.writerow(header)

    # Rows
    for rec in records:
        row = [rec["label"]]
        if show_hw:
            row.append(_short_gpu_name(rec.get("_hardware_gpu")))
        row.extend([
            rec.get("quant", "-"),
            rec.get("kv", "-"),
            rec.get("batch", "-"),
            str(rec.get("ctx", "-")),
        ])
        for mc in metric_cols:
            row.append(_fmt_val_csv(rec.get(mc)))
        vram = rec.get("vram_used")
        row.append(str(vram) if vram is not None else "")
        writer.writerow(row)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# JSON formatting
# ---------------------------------------------------------------------------

def _format_json(records: list[dict[str, Any]], winners: dict[str, Any]) -> dict[str, Any]:
    """Build structured JSON output."""
    cleaned = []
    for rec in records:
        entry: dict[str, Any] = {
            "label": rec["label"],
            "quant": rec.get("quant", "-"),
            "kv": rec.get("kv", "-"),
            "batch": rec.get("batch", "-"),
            "ctx": rec.get("ctx", 0),
        }
        # PP metrics
        for pk in PP_KEYS:
            key = f"pp_{pk}"
            entry[key] = rec.get(key)
        # TG metrics
        for dk in TG_DISPLAY_KEYS:
            entry[dk] = rec.get(dk)
        entry["tg_mean"] = rec.get("tg_mean", 0.0)
        entry["vram_used"] = rec.get("vram_used")
        entry["vram_total"] = rec.get("vram_total")
        cleaned.append(entry)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results": cleaned,
        "winners": winners,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _format_markdown(records: list[dict[str, Any]], winners: dict[str, Any],
                     group_by: str | None, pp_detail: bool = False) -> str:
    """Format records as a markdown table."""
    if not records:
        return "No benchmark results found."

    if group_by:
        return _format_grouped_markdown(records, winners, group_by, pp_detail=pp_detail)

    return _format_single_markdown(records, winners, omit_axis=None, pp_detail=pp_detail)


def _format_single_markdown(records: list[dict[str, Any]], winners: dict[str, Any],
                             omit_axis: str | None, pp_detail: bool = False) -> str:
    """Render one markdown table."""
    if pp_detail:
        metric_cols = [f"pp_{k}" for k in PP_KEYS] + list(TG_DISPLAY_KEYS)
    else:
        metric_cols = ["pp_512"] + list(TG_DISPLAY_KEYS)
    show_hw = _has_mixed_hardware(records)

    # Axis columns
    axis_defs = [("quant", "Quant"), ("kv", "KV"), ("batch", "Batch"), ("ctx", "Ctx")]
    axis_cols = [(key, name) for key, name in axis_defs if key != omit_axis]

    # Header
    header_cells = ["Label"]
    if show_hw:
        header_cells.append("Hardware")
    for _, name in axis_cols:
        header_cells.append(name)
    for mc in metric_cols:
        header_cells.append(COLUMN_NAMES.get(mc, mc))
    header_cells.append("VRAM")

    lines: list[str] = []
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join("---" for _ in header_cells) + " |")

    # Data rows
    for rec in records:
        cells = [rec["label"]]
        if show_hw:
            cells.append(_short_gpu_name(rec.get("_hardware_gpu")))
        for key, _ in axis_cols:
            cells.append(str(rec.get(key, "-")))
        for mc in metric_cols:
            is_winner = (mc in winners and winners[mc]["label"] == rec["label"]
                         and rec.get(mc) is not None)
            cells.append(_fmt_val(rec.get(mc), is_winner))
        vram = rec.get("vram_used")
        vram_winner = ("vram" in winners and winners["vram"]["label"] == rec["label"]
                       and vram is not None)
        if vram is not None:
            cells.append(f"{vram}{'*' if vram_winner else ''}")
        else:
            cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")

    # Winners footer
    lines.append("")
    winner_parts = []
    for mc in metric_cols:
        if mc in winners:
            w = winners[mc]
            winner_parts.append(
                f"**{COLUMN_NAMES.get(mc, mc)}**: {w['label']} ({w['value']:.1f})"
            )
    if "vram" in winners:
        w = winners["vram"]
        winner_parts.append(f"**VRAM**: {w['label']} ({w['value']} MB)")

    if winner_parts:
        lines.append("**WINNERS:** " + " | ".join(winner_parts))

    return "\n".join(lines)


def _format_grouped_markdown(records: list[dict[str, Any]], winners: dict[str, Any],
                              group_by: str, pp_detail: bool = False) -> str:
    """Format records as grouped markdown tables."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = str(rec.get(group_by, "-"))
        groups.setdefault(key, []).append(rec)

    sections: list[str] = []
    for group_val in sorted(groups.keys()):
        group_records = groups[group_val]
        group_winners = _compute_winners(group_records)
        header = f"### {group_by.capitalize()}: {group_val}"
        table = _format_single_markdown(group_records, group_winners, omit_axis=group_by,
                                        pp_detail=pp_detail)
        sections.append(f"{header}\n\n{table}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--dir", "result_dir", default=None,
              help="Results directory (default: benchmarks/matrix/)")
@click.option("--filter", "filter_str", default=None,
              help="Filter by label substring")
@click.option("--group-by", type=click.Choice(["quant", "kv", "batch", "ctx"]),
              default=None, help="Group results by matrix axis")
@click.option("--sort-by",
              type=click.Choice(["label", "tg_mean", "pp_512", "pp_1024", "vram"]),
              default="label", help="Sort results (default: label)")
@click.option("--format", "output_format",
              type=click.Choice(["table", "csv", "json", "markdown"]),
              default="table", help="Output format (default: table)")
@click.option("--pp-detail", is_flag=True,
              help="Show all PP lengths (512/1024/4096/16384) instead of just PP-512")
@click.option("--include-legacy", is_flag=True,
              help="Also scan old-format results in benchmarks/")
@click.option("--output", "output_path", default=None,
              help="Save comparison JSON to path")
def compare(
    files: tuple[str, ...],
    result_dir: str | None,
    filter_str: str | None,
    group_by: str | None,
    sort_by: str,
    output_format: str,
    pp_detail: bool,
    include_legacy: bool,
    output_path: str | None,
) -> None:
    """Compare benchmark results side-by-side.

    Reads matrix benchmark JSON files and produces comparison tables showing
    PP throughput, TG speed, and VRAM usage across configurations.

    Pass specific FILES as arguments, or use --dir to scan a directory.

    \b
    Examples:
        llm-bench compare
        llm-bench compare --pp-detail
        llm-bench compare --filter s007-e1 --group-by ctx
        llm-bench compare --sort-by tg_mean --format markdown
        llm-bench compare result1.json result2.json
        llm-bench compare --include-legacy --sort-by vram
    """
    # 1. Load results
    if files:
        records = _load_files([Path(f) for f in files])
    else:
        directory = Path(result_dir) if result_dir else Path("./benchmarks/matrix")
        records = _load_dir(directory, include_legacy)

    if not records:
        source = ", ".join(files) if files else (result_dir or "./benchmarks/matrix/")
        click.echo(f"No benchmark results found in {source}", err=True)
        raise SystemExit(0)

    # 2. Filter
    if filter_str:
        filter_lower = filter_str.lower()
        records = [r for r in records if filter_lower in r["label"].lower()]
        if not records:
            click.echo(f"No results match filter '{filter_str}'", err=True)
            raise SystemExit(0)

    # 3. Sort
    records.sort(key=lambda r: _sort_key(r, sort_by))

    # 4. Compute winners
    winners = _compute_winners(records)

    # 5. Format and output
    if output_format == "table":
        output = _format_table(records, winners, group_by, pp_detail=pp_detail)
        click.echo(output)

    elif output_format == "csv":
        output = _format_csv(records, pp_detail=pp_detail)
        click.echo(output, nl=False)

    elif output_format == "json":
        json_data = _format_json(records, winners)
        click.echo(json.dumps(json_data, indent=2))

    elif output_format == "markdown":
        output = _format_markdown(records, winners, group_by, pp_detail=pp_detail)
        click.echo(output)

    # 6. Save comparison JSON if requested
    if output_path:
        json_data = _format_json(records, winners)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(json_data, f, indent=2)
        click.echo(f"Comparison JSON saved to: {output_path}", err=True)
