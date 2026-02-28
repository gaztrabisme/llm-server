"""Quality evaluation command — PPL and KL divergence measurement."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import click

from ..core.config import load_config, apply_overrides
from ..core.docker import run_docker_command


# --- Regex patterns for parsing llama-perplexity output ---

PPL_RE = re.compile(r"(?:Final estimate: PPL|Mean PPL\(Q\))\s*=\s*([\d.]+)")
KLD_MEAN_RE = re.compile(r"KLD mean:\s*([\d.]+)")
KLD_MAX_RE = re.compile(r"KLD max:\s*([\d.]+)")
DELTA_P_MEAN_RE = re.compile(r"delta-p mean:\s*([-\d.]+)")
DELTA_P_RMS_RE = re.compile(r"delta-p RMS:\s*([\d.]+)")
SAME_TOP_P_RE = re.compile(r"same-top-p:\s*([\d.]+)%")


def _parse_float(regex: re.Pattern, text: str) -> float | None:
    """Extract a float from text using a compiled regex, or return None."""
    m = regex.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _build_perplexity_args(
    model_path: str,
    corpus_path: str,
    ctx: int,
    threads: int,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build common llama-perplexity CLI arguments."""
    args = [
        "-m", model_path,
        "--file", corpus_path,
        "-ngl", "999",
        "-ot", "exps=CPU",
        "-fa", "on",
        "-t", str(threads),
        "--no-mmap",
        "-c", str(ctx),
    ]
    if extra_args:
        args.extend(extra_args)
    return args


def _run_perplexity(
    image: str,
    model_dir: str,
    corpus_dir: str,
    args: list[str],
    extra_volumes: list[str] | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> str:
    """Run llama-perplexity via Docker and return combined stdout+stderr.

    If dry_run is True, prints the command and returns empty string.
    """
    volumes = [f"{os.path.abspath(corpus_dir)}:/data"]
    if extra_volumes:
        volumes.extend(extra_volumes)

    if dry_run:
        # Build the command string for display
        vol_flags = " ".join(f"-v {v}" for v in volumes)
        arg_str = " ".join(args)
        click.echo(
            f"  docker run --rm --gpus all --ipc host "
            f"--entrypoint llama-perplexity "
            f"-v {os.path.abspath(model_dir)}:/models {vol_flags} "
            f"{image} {arg_str}"
        )
        return ""

    result = run_docker_command(
        image=image,
        entrypoint="llama-perplexity",
        args=args,
        model_dir=model_dir,
        extra_volumes=volumes,
        verbose=verbose,
        capture=True,
    )

    output = (result.stdout or "") + (result.stderr or "")

    if result.returncode != 0:
        raise click.ClickException(
            f"llama-perplexity failed (exit {result.returncode}).\n"
            f"Output:\n{output[-2000:]}"
        )

    return output


@click.command()
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="YAML config file path")
@click.option("--model", required=True, help="Model GGUF path to evaluate")
@click.option("--reference", default=None,
              help="Reference model path for KLD")
@click.option("--corpus", default=None, help="WikiText-2 corpus path")
@click.option("--ctx", type=int, default=None, help="Context size for perplexity")
@click.option("--image", default=None, help="Docker image override")
@click.option("--ppl-only", is_flag=True, help="Skip KLD, only measure PPL")
@click.option("--kld-only", is_flag=True,
              help="Skip PPL, use existing logits file")
@click.option("--logits-file", default=None,
              help="Path to existing logits file for KLD")
@click.option("--model-dir", default=None, help="Model directory")
@click.option("--dry-run", is_flag=True,
              help="Print Docker commands without executing")
@click.option("--verbose", is_flag=True, help="Show detailed output")
def quality(
    config_path: str | None,
    model: str,
    reference: str | None,
    corpus: str | None,
    ctx: int | None,
    image: str | None,
    ppl_only: bool,
    kld_only: bool,
    logits_file: str | None,
    model_dir: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Measure model quality via perplexity and KL divergence."""

    # --- Validate mutually exclusive flags ---
    if ppl_only and kld_only:
        raise click.UsageError("Cannot combine --ppl-only and --kld-only")

    if kld_only and logits_file is None:
        raise click.UsageError("--kld-only requires --logits-file")

    # --- Load config and apply CLI overrides ---
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, image=image, ctx=ctx)

    quality_cfg = cfg.get("quality", {})
    server_cfg = cfg.get("server", {})

    resolved_image = cfg["server"]["image"]
    resolved_ctx = cfg["server_args"].get("ctx_size", quality_cfg.get("ctx", 512))
    resolved_threads = cfg["server_args"].get("threads", 20)
    resolved_model_dir = model_dir or server_cfg.get("model_dir", "./models")
    resolved_corpus = corpus or quality_cfg.get(
        "corpus", "benchmarks/perplexity/wikitext-2-raw/wiki.test.raw"
    )

    # If --ctx was passed explicitly, use that instead of server_args default
    if ctx is not None:
        resolved_ctx = ctx

    # --- Resolve model filename for Docker ---
    # Model path: could be absolute, relative, or just a filename
    model_path = Path(model)
    if model_path.is_absolute():
        # Absolute path: model_dir is the parent, filename is the name
        resolved_model_dir = str(model_path.parent)
        model_filename = model_path.name
    else:
        # Strip leading directory components (e.g., "models/foo.gguf" → "foo.gguf")
        # Docker mount maps model_dir → /models, so we only need the filename
        model_filename = model_path.name

    # Docker paths
    docker_model = f"/models/{model_filename}"
    corpus_path = Path(resolved_corpus)
    corpus_dir = str(corpus_path.parent)
    corpus_filename = corpus_path.name
    docker_corpus = f"/data/{corpus_filename}"

    # --- Resolve reference model for KLD ---
    reference_model = reference
    if not ppl_only and reference_model is None:
        # Default: derive Q8_0 path from the model base name
        ref_quant = quality_cfg.get("reference_quant", "Q8_0")
        model_base = quality_cfg.get("model_base", "Qwen3.5-35B-A3B")
        reference_model = f"{model_base}-{ref_quant}.gguf"

    docker_reference = f"/models/{Path(reference_model).name}" if reference_model else None

    # --- Resolve logits file ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if logits_file is not None:
        resolved_logits = logits_file
    else:
        # Place logits file in model_dir so Docker can access via /models mount
        logits_basename = f"logits-ref-{timestamp}.bin"
        resolved_logits = os.path.join(resolved_model_dir, logits_basename)

    docker_logits = f"/models/{Path(resolved_logits).name}"

    # --- Display header ---
    model_label = Path(model_filename).stem
    click.echo(f"Quality evaluation: {model_label}")

    # --- Results accumulator ---
    result: dict = {
        "timestamp": timestamp,
        "model": model_filename,
        "corpus": corpus_filename,
        "ctx": resolved_ctx,
        "version": "llm-bench-0.1.0",
    }

    # =====================================================================
    # PPL measurement
    # =====================================================================
    if not kld_only:
        click.echo(f"  Measuring PPL (ctx={resolved_ctx})...")

        ppl_args = _build_perplexity_args(
            model_path=docker_model,
            corpus_path=docker_corpus,
            ctx=resolved_ctx,
            threads=resolved_threads,
        )

        output = _run_perplexity(
            image=resolved_image,
            model_dir=resolved_model_dir,
            corpus_dir=corpus_dir,
            args=ppl_args,
            verbose=verbose,
            dry_run=dry_run,
        )

        ppl_value = _parse_float(PPL_RE, output)
        if ppl_value is not None:
            click.echo(f"  PPL = {ppl_value}")
            result["ppl"] = ppl_value
        elif not dry_run:
            click.echo("  Warning: could not parse PPL from output")
            if verbose:
                click.echo(f"  Last 500 chars of output:\n{output[-500:]}")
            result["ppl"] = None

    # =====================================================================
    # KLD measurement
    # =====================================================================
    if not ppl_only:
        # --- Step 1: Generate reference logits (unless --kld-only) ---
        if not kld_only:
            click.echo(f"  Generating reference logits ({Path(reference_model).stem})...")

            ref_args = _build_perplexity_args(
                model_path=docker_reference,
                corpus_path=docker_corpus,
                ctx=resolved_ctx,
                threads=resolved_threads,
                extra_args=["--kl-divergence-base", docker_logits],
            )

            _run_perplexity(
                image=resolved_image,
                model_dir=resolved_model_dir,
                corpus_dir=corpus_dir,
                args=ref_args,
                verbose=verbose,
                dry_run=dry_run,
            )

            if not dry_run:
                click.echo(f"  Reference logits saved: {resolved_logits}")

        # --- Step 2: Compare test model against reference logits ---
        click.echo("  Comparing KLD...")

        kld_args = _build_perplexity_args(
            model_path=docker_model,
            corpus_path=docker_corpus,
            ctx=resolved_ctx,
            threads=resolved_threads,
            extra_args=[
                "--kl-divergence-base", docker_logits,
                "--kl-divergence",
            ],
        )

        kld_output = _run_perplexity(
            image=resolved_image,
            model_dir=resolved_model_dir,
            corpus_dir=corpus_dir,
            args=kld_args,
            verbose=verbose,
            dry_run=dry_run,
        )

        kld_mean = _parse_float(KLD_MEAN_RE, kld_output)
        kld_max = _parse_float(KLD_MAX_RE, kld_output)
        delta_p_mean = _parse_float(DELTA_P_MEAN_RE, kld_output)
        delta_p_rms = _parse_float(DELTA_P_RMS_RE, kld_output)
        same_top_p = _parse_float(SAME_TOP_P_RE, kld_output)

        if kld_mean is not None:
            click.echo(f"  KLD mean: {kld_mean}, same-top-p: {same_top_p}%")
        elif not dry_run:
            click.echo("  Warning: could not parse KLD from output")
            if verbose:
                click.echo(f"  Last 500 chars of output:\n{kld_output[-500:]}")

        result["kld"] = {
            "reference": reference_model,
            "kld_mean": kld_mean,
            "kld_max": kld_max,
            "delta_p_mean": delta_p_mean,
            "delta_p_rms": delta_p_rms,
            "same_top_p": same_top_p,
        }

    # =====================================================================
    # Save results
    # =====================================================================
    if dry_run:
        click.echo("  (dry run — no results saved)")
        return

    output_dir = cfg.get("output", {}).get("dir", "./benchmarks/matrix")
    quality_dir = os.path.join(os.path.dirname(output_dir), "quality")
    os.makedirs(quality_dir, exist_ok=True)

    out_filename = f"quality-{model_label}-{timestamp}.json"
    out_path = os.path.join(quality_dir, out_filename)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    click.echo(f"Results saved to {out_path}")

    # --- Clean up logits file if we generated it ---
    if not ppl_only and not kld_only and logits_file is None:
        if os.path.exists(resolved_logits):
            logits_size_gb = os.path.getsize(resolved_logits) / (1024 ** 3)
            click.echo(
                f"  Note: reference logits file ({logits_size_gb:.1f} GB) "
                f"at {resolved_logits}"
            )
            click.echo(
                "  Reuse with --kld-only --logits-file to skip regeneration, "
                "or delete to reclaim disk space."
            )
