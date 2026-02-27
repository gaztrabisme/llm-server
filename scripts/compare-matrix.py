#!/usr/bin/env python3
"""Matrix benchmark comparison tool for llm-server Session 007+.

Reads matrix benchmark results from benchmarks/matrix/ and produces comparison
tables sliced by any matrix axis.

Usage:
    python3 scripts/compare-matrix.py [options]

Options:
    --dir <path>          Results directory (default: benchmarks/matrix/)
    --group-by <axis>     Group results by axis: quant | kv | batch | ctx (default: show all)
    --filter <pattern>    Filter results by label pattern (e.g., "s007-e1" for E1 only)
    --metric <type>       Focus metric: tg | pp | vram | all (default: all)
    --format <type>       Output format: table | csv | json (default: table)
    --output <file>       Write detailed JSON to file (always writes, this overrides path)
    --sort-by <field>     Sort rows by: tg_mean | pp_512 | pp_1024 | vram | label (default: label)
    --help                Show this help

Output modes:
    table (default):
        Prints a compact ASCII table to stdout showing PP and TG speed for
        every tested combination. Uses fixed-width columns for alignment.

    csv:
        Prints CSV to stdout (pipe to file or use --output).

    json:
        Prints full structured JSON to stdout.

Examples:
    # Show all results in a table
    python3 scripts/compare-matrix.py

    # Group by KV config (one table per KV type)
    python3 scripts/compare-matrix.py --group-by kv

    # Show only E1 results, grouped by context length
    python3 scripts/compare-matrix.py --filter s007-e1 --group-by ctx

    # Export E2 results as CSV
    python3 scripts/compare-matrix.py --filter s007-e2 --format csv

    # Sort by TG speed (fastest first)
    python3 scripts/compare-matrix.py --sort-by tg_mean

SPEC — Implementation notes:

1. DATA LOADING
   - Scan benchmarks/matrix/*.json
   - Parse each file, extract:
     - label, matrix dict (quant, kv, batch, ctx)
     - pp_benchmarks dict (keyed by prompt length)
     - tg_benchmarks dict (keyed by workload name)
     - gpu_stats
   - Apply --filter as substring match on label
   - Sort by --sort-by field

2. TABLE FORMAT (default, --format table)
   - Header row with column names
   - One row per benchmark result
   - Columns:
     | Label | Quant | KV | Batch | Ctx | PP@512 | PP@1024 | PP@4096 | PP@16k | TG Short | TG Med | TG Long | TG Multi | VRAM |
   - Values: mean_tps with stddev in parentheses, e.g., "74.7 (1.2)"
   - Missing values shown as "-"
   - Best value per column highlighted with "*" suffix
   - Footer: WINNER row showing best config per metric

3. GROUP-BY MODE
   - When --group-by is specified, split results into groups
   - Print one table per group, with group header
   - E.g., --group-by kv prints:
     === KV: f16 ===
     [table of all f16 results]
     === KV: q8_0 ===
     [table of all q8_0 results]
     ...
   - Within each group, columns for the grouping axis are omitted (redundant)

4. CSV FORMAT (--format csv)
   - Standard CSV with header row
   - Same columns as table format
   - Values: just the mean (no stddev)
   - Suitable for import into spreadsheets

5. JSON FORMAT (--format json)
   - Full structured output:
     {
       "generated_at": "2026-03-01T14:30:00",
       "results_dir": "benchmarks/matrix/",
       "filter": "s007-e1",
       "group_by": "kv",
       "results": [
         {
           "label": "s007-e1-kvq8-ctx32k",
           "matrix": {"quant": "Q4_K_M", "kv": "q8_0", "batch": "none", "ctx": 32768},
           "pp": {"512": {"mean": 1200.5, "std": 15.2}, ...},
           "tg": {"short_prompt": {"mean": 74.7, "std": 1.2}, ...},
           "vram_mb": 14559
         },
         ...
       ],
       "winners": {
         "tg_short": {"label": "...", "value": 76.1},
         "pp_512": {"label": "...", "value": 1500.0},
         ...
       }
     }

6. DETAILED JSON OUTPUT (always)
   - Regardless of --format, write a detailed JSON file to:
     benchmarks/matrix/comparison-{timestamp}.json
   - Override path with --output
   - Contains everything from JSON format above, plus per-run data

7. CONTEXT-CONSCIOUS OUTPUT
   - Detect terminal width (shutil.get_terminal_size)
   - If width < 160: use compact column names (PP5, PP1k, TGs, TGm, etc.)
   - If width < 100: omit PP columns, show TG only
   - Always show full data in JSON output file

8. COMPARISON WITH OLD RESULTS
   - Can also read from benchmarks/*.json (old bench.sh format)
   - When old-format files detected, extract TG data and map to matrix format
   - Label derived from filename (e.g., "s006-e4-fit-nobatch")
   - PP data not available from old format (shown as "-")
"""

import argparse
import csv
import io
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

PP_KEYS = ["512", "1024", "4096", "16384"]
TG_KEYS = ["short_prompt", "medium_prompt", "long_prompt", "multi_turn"]

FULL_COL_NAMES = {
    "pp_512": "PP@512",
    "pp_1024": "PP@1024",
    "pp_4096": "PP@4096",
    "pp_16384": "PP@16k",
    "tg_short_prompt": "TG Short",
    "tg_medium_prompt": "TG Med",
    "tg_long_prompt": "TG Long",
    "tg_multi_turn": "TG Multi",
}

COMPACT_COL_NAMES = {
    "pp_512": "PP5",
    "pp_1024": "PP1k",
    "pp_4096": "PP4k",
    "pp_16384": "PP16k",
    "tg_short_prompt": "TGs",
    "tg_medium_prompt": "TGm",
    "tg_long_prompt": "TGl",
    "tg_multi_turn": "TGmt",
}

MATRIX_AXES = ["quant", "kv", "batch", "ctx"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matrix_result(filepath: Path) -> dict | None:
    """Load a matrix-format JSON result file and return a normalized record."""
    try:
        with open(filepath) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: skipping {filepath.name}: {e}", file=sys.stderr)
        return None

    # Detect format: matrix-format has "matrix" key, old-format has "workloads"
    if "matrix" in data:
        return _normalize_matrix(data, filepath)
    elif "workloads" in data:
        return _normalize_old(data, filepath)
    else:
        print(f"Warning: skipping {filepath.name}: unrecognized format", file=sys.stderr)
        return None


def _normalize_matrix(data: dict, filepath: Path) -> dict:
    """Normalize a matrix-format result into the internal record structure."""
    label = data.get("label", filepath.stem)
    matrix = data.get("matrix", {})

    pp = {}
    for key, bench in data.get("pp_benchmarks", {}).items():
        pp[key] = {
            "mean": bench.get("mean_tps"),
            "std": bench.get("stddev_tps"),
            "runs": bench.get("runs", []),
        }

    tg = {}
    for key, bench in data.get("tg_benchmarks", {}).items():
        tg[key] = {
            "mean": bench.get("mean_tps"),
            "std": bench.get("stddev_tps"),
            "runs": bench.get("runs", []),
        }

    vram_mb = None
    gpu_stats = data.get("gpu_stats", {})
    after_load = gpu_stats.get("after_load", {})
    if after_load and "used_mb" in after_load:
        vram_mb = after_load["used_mb"]

    return {
        "label": label,
        "matrix": {
            "quant": matrix.get("quant", "-"),
            "kv": matrix.get("kv", "-"),
            "batch": matrix.get("batch", "-"),
            "ctx": matrix.get("ctx", "-"),
        },
        "pp": pp,
        "tg": tg,
        "vram_mb": vram_mb,
        "gpu_stats": gpu_stats,
        "_source": str(filepath),
    }


def _normalize_old(data: dict, filepath: Path) -> dict:
    """Normalize an old bench.sh format result into the internal record structure."""
    # Derive label from filename, stripping engine prefix and timestamp suffix
    # e.g., "llama-cpp-s006-e4-fit-nobatch-20260227-164812" -> "s006-e4-fit-nobatch"
    stem = filepath.stem
    label = stem
    # Strip engine prefix (e.g., "llama-cpp-" or "ik-llama-")
    for prefix in ("llama-cpp-", "ik-llama-"):
        if stem.startswith(prefix):
            label = stem[len(prefix):]
            break
    # Strip timestamp suffix (pattern: -YYYYMMDD-HHMMSS)
    parts = label.rsplit("-", 2)
    if len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
        if parts[-2].isdigit() and parts[-1].isdigit():
            label = "-".join(parts[:-2])

    tg = {}
    for key, wdata in data.get("workloads", {}).items():
        tg[key] = {
            "mean": wdata.get("mean_tps"),
            "std": wdata.get("stddev_tps"),
            "runs": wdata.get("runs", []),
        }

    # Parse VRAM from old CSV format: "used, total, util%"
    vram_mb = None
    gpu_stats_raw = data.get("gpu_stats", {})
    vram_info = gpu_stats_raw.get("vram_info", "")
    if vram_info:
        try:
            vram_mb = int(vram_info.split(",")[0].strip())
        except (ValueError, IndexError):
            pass

    # Build normalized gpu_stats from old format
    gpu_stats = {}
    if vram_info:
        try:
            parts = [p.strip() for p in vram_info.split(",")]
            gpu_stats["after_load"] = {
                "used_mb": int(parts[0]),
                "total_mb": int(parts[1]) if len(parts) > 1 else None,
                "utilization_pct": int(parts[2]) if len(parts) > 2 else None,
            }
        except (ValueError, IndexError):
            pass
    vram_after = gpu_stats_raw.get("vram_after_bench", "")
    if vram_after:
        try:
            parts = [p.strip() for p in vram_after.split(",")]
            gpu_stats["after_bench"] = {
                "used_mb": int(parts[0]),
                "total_mb": int(parts[1]) if len(parts) > 1 else None,
                "utilization_pct": int(parts[2]) if len(parts) > 2 else None,
            }
        except (ValueError, IndexError):
            pass

    return {
        "label": label,
        "matrix": {
            "quant": "-",
            "kv": "-",
            "batch": "-",
            "ctx": "-",
        },
        "pp": {},
        "tg": tg,
        "vram_mb": vram_mb,
        "gpu_stats": gpu_stats,
        "_source": str(filepath),
    }


def load_all_results(results_dir: str) -> list[dict]:
    """Load all results from the given directory, including old-format fallback."""
    records = []
    rdir = Path(results_dir)

    # Load matrix-format results from the specified directory
    if rdir.is_dir():
        for f in sorted(rdir.glob("*.json")):
            # Skip comparison output files
            if f.name.startswith("comparison-"):
                continue
            rec = load_matrix_result(f)
            if rec is not None:
                records.append(rec)

    # Also scan old-format results from benchmarks/ (parent of matrix/)
    old_dir = rdir.parent if rdir.name == "matrix" else rdir
    if old_dir.is_dir() and old_dir != rdir:
        for f in sorted(old_dir.glob("*.json")):
            rec = load_matrix_result(f)
            if rec is not None:
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_key(record: dict, sort_by: str):
    """Return a sort key for the given record and sort field."""
    if sort_by == "label":
        return record["label"]
    elif sort_by == "tg_mean":
        # Average of all available TG means; missing -> 0
        vals = [v["mean"] for v in record["tg"].values() if v.get("mean") is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        return -avg  # descending (fastest first)
    elif sort_by == "pp_512":
        pp = record["pp"].get("512", {})
        val = pp.get("mean") if pp else None
        return -(val or 0.0)
    elif sort_by == "pp_1024":
        pp = record["pp"].get("1024", {})
        val = pp.get("mean") if pp else None
        return -(val or 0.0)
    elif sort_by == "vram":
        return record.get("vram_mb") or 99999
    else:
        return record["label"]


# ---------------------------------------------------------------------------
# Winner computation
# ---------------------------------------------------------------------------

def compute_winners(records: list[dict]) -> dict:
    """Compute the best (highest) value per metric column across all records."""
    winners = {}

    # PP winners
    for pk in PP_KEYS:
        best_val = None
        best_label = None
        for rec in records:
            pp = rec["pp"].get(pk, {})
            val = pp.get("mean")
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_label = rec["label"]
        if best_val is not None:
            winners[f"pp_{pk}"] = {"label": best_label, "value": best_val}

    # TG winners
    for tk in TG_KEYS:
        best_val = None
        best_label = None
        for rec in records:
            tg = rec["tg"].get(tk, {})
            val = tg.get("mean")
            if val is not None and (best_val is None or val > best_val):
                best_val = val
                best_label = rec["label"]
        if best_val is not None:
            winners[f"tg_{tk}"] = {"label": best_label, "value": best_val}

    # VRAM winner (lowest is best)
    best_vram = None
    best_label = None
    for rec in records:
        val = rec.get("vram_mb")
        if val is not None and (best_vram is None or val < best_vram):
            best_vram = val
            best_label = rec["label"]
    if best_vram is not None:
        winners["vram"] = {"label": best_label, "value": best_vram}

    return winners


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _format_value(mean, std, is_best=False) -> str:
    """Format a metric value as 'mean (+-std)' with optional '*' for best."""
    if mean is None:
        return "-"
    std_str = f"\u00b1{std:.1f}" if std is not None else ""
    suffix = "*" if is_best else ""
    if std_str:
        return f"{mean:.1f} ({std_str}){suffix}"
    else:
        return f"{mean:.1f}{suffix}"


def _format_value_csv(mean) -> str:
    """Format a metric value for CSV (mean only, no stddev)."""
    if mean is None:
        return ""
    return f"{mean:.1f}"


def _determine_columns(metric: str, term_width: int) -> tuple[list[str], dict[str, str]]:
    """Determine which metric columns to show based on --metric and terminal width.

    Returns (column_keys, column_name_map).
    """
    # Start with all columns based on metric filter
    if metric == "pp":
        col_keys = [f"pp_{k}" for k in PP_KEYS]
    elif metric == "tg":
        col_keys = [f"tg_{k}" for k in TG_KEYS]
    elif metric == "vram":
        col_keys = []  # VRAM is always shown as a separate column
    else:
        # "all" — PP + TG
        col_keys = [f"pp_{k}" for k in PP_KEYS] + [f"tg_{k}" for k in TG_KEYS]

    # Context-conscious narrowing for table output
    if term_width < 100:
        # Omit PP columns, show TG only
        col_keys = [k for k in col_keys if k.startswith("tg_")]

    # Choose name map
    if term_width < 160:
        name_map = COMPACT_COL_NAMES
    else:
        name_map = FULL_COL_NAMES

    return col_keys, name_map


def _get_metric_value(record: dict, col_key: str) -> tuple:
    """Extract (mean, std) for a given metric column key from a record."""
    if col_key.startswith("pp_"):
        pk = col_key[3:]
        pp = record["pp"].get(pk, {})
        return (pp.get("mean"), pp.get("std"))
    elif col_key.startswith("tg_"):
        tk = col_key[3:]
        tg = record["tg"].get(tk, {})
        return (tg.get("mean"), tg.get("std"))
    return (None, None)


def format_table(records: list[dict], metric: str, group_by: str | None,
                 winners: dict, term_width: int) -> str:
    """Format records into an ASCII comparison table."""
    if not records:
        return "No benchmark results found."

    col_keys, name_map = _determine_columns(metric, term_width)

    if group_by:
        return _format_grouped_tables(records, col_keys, name_map, group_by, winners, metric)
    else:
        return _format_single_table(records, col_keys, name_map, None, winners, metric)


def _format_single_table(records: list[dict], col_keys: list[str],
                         name_map: dict, omit_axis: str | None,
                         winners: dict, metric: str) -> str:
    """Format a single ASCII table for a set of records."""
    lines = []

    # Determine matrix axis columns to show
    axis_cols = [a for a in MATRIX_AXES if a != omit_axis]

    # Build header
    header_parts = ["Label"]
    for ac in axis_cols:
        header_parts.append(ac.capitalize())
    for ck in col_keys:
        header_parts.append(name_map.get(ck, ck))
    header_parts.append("VRAM")

    # Compute column widths
    # Build all row data first, then compute widths
    row_data = []
    for rec in records:
        row = [rec["label"]]
        for ac in axis_cols:
            val = rec["matrix"].get(ac, "-")
            row.append(str(val))
        for ck in col_keys:
            mean, std = _get_metric_value(rec, ck)
            is_best = (ck in winners and winners[ck]["label"] == rec["label"]
                       and mean is not None)
            row.append(_format_value(mean, std, is_best))
        # VRAM column
        vram = rec.get("vram_mb")
        vram_is_best = ("vram" in winners and winners["vram"]["label"] == rec["label"]
                        and vram is not None)
        if vram is not None:
            vram_str = f"{vram}" + ("*" if vram_is_best else "")
        else:
            vram_str = "-"
        row.append(vram_str)
        row_data.append(row)

    # Calculate column widths
    col_widths = []
    for i, h in enumerate(header_parts):
        w = len(h)
        for row in row_data:
            if i < len(row):
                w = max(w, len(row[i]))
        col_widths.append(w)

    # Format header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header_parts))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(header_parts)))

    lines.append(header_line)
    lines.append(separator)

    # Format rows
    for row in row_data:
        line = " | ".join(row[i].ljust(col_widths[i]) for i in range(len(header_parts)))
        lines.append(line)

    # Footer: WINNER row per metric
    lines.append(separator)
    winner_parts = []
    for ck in col_keys:
        if ck in winners:
            w = winners[ck]
            winner_parts.append(f"{name_map.get(ck, ck)}: {w['label']} ({w['value']:.1f})")
    if "vram" in winners and (metric in ("all", "vram")):
        w = winners["vram"]
        winner_parts.append(f"VRAM: {w['label']} ({w['value']} MB)")

    if winner_parts:
        lines.append("WINNERS:")
        for wp in winner_parts:
            lines.append(f"  {wp}")

    return "\n".join(lines)


def _format_grouped_tables(records: list[dict], col_keys: list[str],
                           name_map: dict, group_by: str,
                           winners: dict, metric: str) -> str:
    """Format records as multiple tables, grouped by a matrix axis."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        key = str(rec["matrix"].get(group_by, "-"))
        groups.setdefault(key, []).append(rec)

    sections = []
    for group_val in sorted(groups.keys()):
        group_records = groups[group_val]
        group_winners = compute_winners(group_records)
        header = f"=== {group_by.capitalize()}: {group_val} ==="
        table = _format_single_table(group_records, col_keys, name_map,
                                     group_by, group_winners, metric)
        sections.append(f"{header}\n{table}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------

def format_csv(records: list[dict], metric: str) -> str:
    """Format records as CSV."""
    col_keys, _ = _determine_columns(metric, 999)  # CSV always uses full columns

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["Label", "Quant", "KV", "Batch", "Ctx"]
    for ck in col_keys:
        header.append(FULL_COL_NAMES.get(ck, ck))
    header.append("VRAM_MB")
    writer.writerow(header)

    # Rows
    for rec in records:
        row = [
            rec["label"],
            rec["matrix"].get("quant", "-"),
            rec["matrix"].get("kv", "-"),
            rec["matrix"].get("batch", "-"),
            str(rec["matrix"].get("ctx", "-")),
        ]
        for ck in col_keys:
            mean, _ = _get_metric_value(rec, ck)
            row.append(_format_value_csv(mean))
        row.append(str(rec.get("vram_mb", "")))
        writer.writerow(row)

    return output.getvalue()


# ---------------------------------------------------------------------------
# JSON formatting
# ---------------------------------------------------------------------------

def format_json_output(records: list[dict], args, winners: dict) -> dict:
    """Build the structured JSON output dictionary."""
    results = []
    for rec in records:
        pp_out = {}
        for pk in PP_KEYS:
            if pk in rec["pp"]:
                pp_out[pk] = {
                    "mean": rec["pp"][pk].get("mean"),
                    "std": rec["pp"][pk].get("std"),
                }

        tg_out = {}
        for tk in TG_KEYS:
            if tk in rec["tg"]:
                tg_out[tk] = {
                    "mean": rec["tg"][tk].get("mean"),
                    "std": rec["tg"][tk].get("std"),
                }

        results.append({
            "label": rec["label"],
            "matrix": rec["matrix"],
            "pp": pp_out,
            "tg": tg_out,
            "vram_mb": rec.get("vram_mb"),
        })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results_dir": args.dir,
        "filter": args.filter,
        "group_by": args.group_by,
        "results": results,
        "winners": winners,
    }


def build_detailed_json(records: list[dict], args, winners: dict) -> dict:
    """Build detailed JSON output including per-run data."""
    results = []
    for rec in records:
        pp_out = {}
        for pk in PP_KEYS:
            if pk in rec["pp"]:
                pp_out[pk] = {
                    "mean": rec["pp"][pk].get("mean"),
                    "std": rec["pp"][pk].get("std"),
                    "runs": rec["pp"][pk].get("runs", []),
                }

        tg_out = {}
        for tk in TG_KEYS:
            if tk in rec["tg"]:
                tg_out[tk] = {
                    "mean": rec["tg"][tk].get("mean"),
                    "std": rec["tg"][tk].get("std"),
                    "runs": rec["tg"][tk].get("runs", []),
                }

        results.append({
            "label": rec["label"],
            "matrix": rec["matrix"],
            "pp": pp_out,
            "tg": tg_out,
            "vram_mb": rec.get("vram_mb"),
            "gpu_stats": rec.get("gpu_stats", {}),
            "source_file": rec.get("_source"),
        })

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results_dir": args.dir,
        "filter": args.filter,
        "group_by": args.group_by,
        "sort_by": args.sort_by,
        "metric": args.metric,
        "format": args.format,
        "total_results": len(results),
        "results": results,
        "winners": winners,
    }


# ---------------------------------------------------------------------------
# Detailed JSON output (always written)
# ---------------------------------------------------------------------------

def write_detailed_json(detailed: dict, output_path: str | None, results_dir: str) -> str:
    """Write detailed JSON to file. Returns the path written to."""
    if output_path:
        path = Path(output_path)
    else:
        # Default: benchmarks/matrix/comparison-{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"comparison-{timestamp}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(detailed, fh, indent=2)

    return str(path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matrix benchmark comparison tool for llm-server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default="benchmarks/matrix/",
        help="Results directory (default: benchmarks/matrix/)",
    )
    parser.add_argument(
        "--group-by",
        choices=["quant", "kv", "batch", "ctx"],
        default=None,
        help="Group results by matrix axis",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter results by label substring",
    )
    parser.add_argument(
        "--metric",
        choices=["tg", "pp", "vram", "all"],
        default="all",
        help="Focus metric (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        dest="format",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override detailed JSON output path",
    )
    parser.add_argument(
        "--sort-by",
        choices=["tg_mean", "pp_512", "pp_1024", "vram", "label"],
        default="label",
        help="Sort rows by field (default: label)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load results
    records = load_all_results(args.dir)

    if not records:
        print(f"No benchmark results found in {args.dir}", file=sys.stderr)
        # Still write empty detailed JSON
        winners = {}
        detailed = build_detailed_json([], args, winners)
        json_path = write_detailed_json(detailed, args.output, args.dir)
        print(f"Detailed JSON written to: {json_path}", file=sys.stderr)
        sys.exit(0)

    # Apply filter
    if args.filter:
        records = [r for r in records if args.filter in r["label"]]
        if not records:
            print(f"No results match filter '{args.filter}'", file=sys.stderr)
            winners = {}
            detailed = build_detailed_json([], args, winners)
            json_path = write_detailed_json(detailed, args.output, args.dir)
            print(f"Detailed JSON written to: {json_path}", file=sys.stderr)
            sys.exit(0)

    # Sort
    records.sort(key=lambda r: sort_key(r, args.sort_by))

    # Compute winners across all (filtered) records
    winners = compute_winners(records)

    # Build and write detailed JSON (always)
    detailed = build_detailed_json(records, args, winners)
    json_path = write_detailed_json(detailed, args.output, args.dir)

    # Output in requested format
    if args.format == "table":
        term_width = shutil.get_terminal_size((120, 40)).columns
        output = format_table(records, args.metric, args.group_by, winners, term_width)
        print(output)
        print(f"\nDetailed JSON written to: {json_path}", file=sys.stderr)

    elif args.format == "csv":
        output = format_csv(records, args.metric)
        print(output, end="")
        print(f"Detailed JSON written to: {json_path}", file=sys.stderr)

    elif args.format == "json":
        json_out = format_json_output(records, args, winners)
        print(json.dumps(json_out, indent=2))
        print(f"Detailed JSON written to: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
