"""llm-bench speed — PP/TG measurement command, ported from bench-matrix.sh."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import click
import requests

from ..compat.env_parser import env_label, parse_env_file
from ..core.config import (
    apply_overrides,
    extract_matrix,
    load_config,
    server_args_to_flags,
    server_args_to_string,
)
from ..core.docker import (
    check_port_available,
    start_container,
    stop_container,
    wait_for_health,
)
from ..core.gpu import capture_vram
from ..core.results import create_result, save_result
from ..core.stats import compute_stats
from ..workloads.defaults import (
    MULTI_TURN_FOLLOWUPS,
    SHORT_PROMPT,
    get_workload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_corpus_text(corpus_path: str, char_count: int) -> str:
    """Read the first *char_count* characters from the WikiText-2 corpus."""
    with open(corpus_path) as f:
        return f.read(char_count)


def _pp_request(text: str) -> dict[str, Any]:
    """Build a /completion request body for PP measurement."""
    return {
        "prompt": text,
        "n_predict": 0,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
    }


def _tg_request(
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Build a /v1/chat/completions request body for TG measurement."""
    return {
        "model": "benchmark",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


# ---------------------------------------------------------------------------
# PP measurement
# ---------------------------------------------------------------------------


def _run_pp_benchmarks(
    server_url: str,
    corpus_path: str,
    pp_lengths: list[int],
    pp_runs: int,
    warmup: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run prompt-processing benchmarks and return pp_benchmarks dict."""
    pp_benchmarks: dict[str, Any] = {}

    for pp_len in pp_lengths:
        char_count = pp_len * 5
        text = _load_corpus_text(corpus_path, char_count)
        body = _pp_request(text)

        runs: list[dict[str, Any]] = []

        for i in range(1, pp_runs + 1):
            try:
                resp = requests.post(
                    f"{server_url}/completion",
                    json=body,
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                click.echo(f"  PP: {pp_len} tokens [{i}/{pp_runs}] ... FAILED ({exc})")
                continue

            timings = data.get("timings", {})
            pp_sec = timings.get("prompt_per_second", 0.0)
            prompt_n = timings.get("prompt_n", 0)

            run_record = {
                "run": i,
                "prompt_tokens": prompt_n,
                "prompt_per_second": round(pp_sec, 1),
            }
            runs.append(run_record)

            tag = "(warmup) " if i <= warmup else ""
            click.echo(
                f"  PP: {pp_len} tokens [{i}/{pp_runs}] {tag}... "
                f"{run_record['prompt_per_second']} tok/s"
            )

        # Compute stats (skipping warmup)
        tps_values = [r["prompt_per_second"] for r in runs]
        stats = compute_stats(tps_values, warmup=warmup)
        pp_benchmarks[str(pp_len)] = {
            "runs": runs,
            "mean_tps": stats.to_dict()["mean_tps"],
            "stddev_tps": stats.to_dict()["stddev_tps"],
        }

        click.echo(
            f"  [PP {pp_len}] {stats.to_dict()['mean_tps']} tok/s "
            f"(+/-{stats.to_dict()['stddev_tps']})"
        )

    return pp_benchmarks


# ---------------------------------------------------------------------------
# TG measurement
# ---------------------------------------------------------------------------


def _run_single_tg_workload(
    server_url: str,
    workload_name: str,
    messages: list[dict[str, str]],
    tg_runs: int,
    max_tokens: int,
    temperature: float,
    warmup: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run a single TG workload and return its benchmark entry."""
    body = _tg_request(messages, max_tokens=max_tokens, temperature=temperature)
    runs: list[dict[str, Any]] = []

    for i in range(1, tg_runs + 1):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=body,
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            click.echo(f"  TG: {workload_name} [{i}/{tg_runs}] ... FAILED ({exc})")
            continue
        wall_s = time.time() - t0
        wall_ms = int(wall_s * 1000)

        usage = data.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)

        # Prefer server-side timings if available
        timings = data.get("timings", {})
        predicted_ps = timings.get("predicted_per_second", 0)

        if predicted_ps and predicted_ps > 0:
            tps = round(predicted_ps, 1)
        elif completion_tok > 0 and wall_ms > 0:
            tps = round(completion_tok / wall_s, 1)
        else:
            tps = 0.0

        run_record = {
            "run": i,
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "tokens_per_sec": tps,
            "wall_ms": wall_ms,
        }
        runs.append(run_record)

        tag = "(warmup) " if i <= warmup else ""
        click.echo(
            f"  TG: {workload_name} [{i}/{tg_runs}] {tag}... {tps} tok/s"
        )

    tps_values = [r["tokens_per_sec"] for r in runs]
    stats = compute_stats(tps_values, warmup=warmup)
    return {
        "runs": runs,
        "mean_tps": stats.to_dict()["mean_tps"],
        "stddev_tps": stats.to_dict()["stddev_tps"],
    }


def _run_multi_turn(
    server_url: str,
    tg_runs: int,
    max_tokens: int,
    temperature: float,
    turns: int,
    truncate: int,
    warmup: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run multi-turn TG workload and return its benchmark entry."""
    runs: list[dict[str, Any]] = []

    for run_idx in range(1, tg_runs + 1):
        # Start each run with the short prompt
        messages: list[dict[str, str]] = list(SHORT_PROMPT)
        total_tokens = 0
        total_ms = 0
        run_failed = False

        for turn in range(turns):
            body = _tg_request(
                messages, max_tokens=max_tokens, temperature=temperature,
            )
            t0 = time.time()
            try:
                resp = requests.post(
                    f"{server_url}/v1/chat/completions",
                    json=body,
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                click.echo(
                    f"  Multi-turn [{run_idx}/{tg_runs}] turn {turn + 1}: "
                    f"FAILED ({exc})"
                )
                run_failed = True
                break
            wall_ms = int((time.time() - t0) * 1000)
            total_ms += wall_ms

            usage = data.get("usage", {})
            completion_tok = usage.get("completion_tokens", 0)
            total_tokens += completion_tok

            # Append truncated assistant response + next follow-up
            try:
                assistant_content = data["choices"][0]["message"]["content"][
                    :truncate
                ]
            except (KeyError, IndexError):
                assistant_content = ""
            messages.append({"role": "assistant", "content": assistant_content})

            # Pick the follow-up (cycle through the list, last turn has no follow-up)
            if turn < turns - 1:
                followup_idx = turn % len(MULTI_TURN_FOLLOWUPS)
                messages.append(
                    {"role": "user", "content": MULTI_TURN_FOLLOWUPS[followup_idx]}
                )

        if run_failed:
            continue

        if total_tokens > 0 and total_ms > 0:
            tps = round(total_tokens / (total_ms / 1000), 1)
        else:
            tps = 0.0

        run_record = {
            "run": run_idx,
            "total_tokens": total_tokens,
            "tokens_per_sec": tps,
            "wall_ms": total_ms,
        }
        runs.append(run_record)

        tag = "(warmup) " if run_idx <= warmup else ""
        click.echo(
            f"  Multi-turn [{run_idx}/{tg_runs}] {tag}... {tps} tok/s"
        )

    tps_values = [r["tokens_per_sec"] for r in runs]
    stats = compute_stats(tps_values, warmup=warmup)
    return {
        "runs": runs,
        "mean_tps": stats.to_dict()["mean_tps"],
        "stddev_tps": stats.to_dict()["stddev_tps"],
    }


def _run_tg_benchmarks(
    server_url: str,
    cfg: dict[str, Any],
    quick: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run all TG workloads and return tg_benchmarks dict."""
    speed_cfg = cfg.get("speed", {})
    tg_runs = speed_cfg.get("tg_runs", 5)
    max_tokens = speed_cfg.get("tg_max_tokens", 256)
    temperature = speed_cfg.get("tg_temperature", 0.7)
    warmup = speed_cfg.get("warmup_runs", 1)
    multi_turn_turns = speed_cfg.get("multi_turn_turns", 5)
    multi_turn_truncate = speed_cfg.get("multi_turn_truncate", 200)

    if quick:
        tg_runs = 3

    workloads = speed_cfg.get("tg_workloads", ["short", "medium", "long", "multi_turn"])
    if quick:
        workloads = [w for w in workloads if w != "multi_turn"]

    tg_benchmarks: dict[str, Any] = {}

    for workload_name in workloads:
        if workload_name == "multi_turn":
            result = _run_multi_turn(
                server_url=server_url,
                tg_runs=tg_runs,
                max_tokens=max_tokens,
                temperature=temperature,
                turns=multi_turn_turns,
                truncate=multi_turn_truncate,
                warmup=warmup,
                verbose=verbose,
            )
            tg_benchmarks["multi_turn"] = result
            click.echo(
                f"  [TG multi_turn] {result['mean_tps']} tok/s "
                f"(+/-{result['stddev_tps']})"
            )
        else:
            messages = get_workload(workload_name)
            result = _run_single_tg_workload(
                server_url=server_url,
                workload_name=workload_name,
                messages=messages,
                tg_runs=tg_runs,
                max_tokens=max_tokens,
                temperature=temperature,
                warmup=warmup,
                verbose=verbose,
            )
            # Use shell-compatible key names: short → short_prompt
            key = f"{workload_name}_prompt" if workload_name != "multi_turn" else workload_name
            tg_benchmarks[key] = result
            click.echo(
                f"  [TG {key}] {result['mean_tps']} tok/s "
                f"(+/-{result['stddev_tps']})"
            )

    return tg_benchmarks


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(), default=None,
    help="YAML config file",
)
@click.option(
    "--env", "env_path", type=click.Path(exists=True), default=None,
    help="Legacy .env config",
)
@click.option("--model", default=None, help="Model path override")
@click.option("--image", default=None, help="Docker image override")
@click.option("--threads", type=int, default=None, help="Thread count override")
@click.option("--ctx", type=int, default=None, help="Context size override")
@click.option("--port", type=int, default=None, help="Port override")
@click.option("--kv-k", default=None, help="KV cache type K override")
@click.option("--kv-v", default=None, help="KV cache type V override")
@click.option("--quick", is_flag=True, help="Fewer runs, fewer PP lengths")
@click.option("--pp-only", is_flag=True, help="Only run prompt processing benchmarks")
@click.option("--tg-only", is_flag=True, help="Only run token generation benchmarks")
@click.option(
    "--pp-lengths", default=None,
    help="Comma-separated PP lengths (e.g. 512,1024,4096)",
)
@click.option("--label", default=None, help="Custom label for results")
@click.option(
    "--skip-server", is_flag=True,
    help="Assume server already running",
)
@click.option(
    "--server-url", default=None,
    help="Server URL (default: http://localhost:{port})",
)
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
@click.option(
    "--output", "output_path", default=None,
    help="Output JSON path (overrides output.dir)",
)
@click.option("--verbose", is_flag=True, help="Show detailed output")
def speed(
    config_path: str | None,
    env_path: str | None,
    model: str | None,
    image: str | None,
    threads: int | None,
    ctx: int | None,
    port: int | None,
    kv_k: str | None,
    kv_v: str | None,
    quick: bool,
    pp_only: bool,
    tg_only: bool,
    pp_lengths: str | None,
    label: str | None,
    skip_server: bool,
    server_url: str | None,
    dry_run: bool,
    output_path: str | None,
    verbose: bool,
) -> None:
    """Run PP and TG speed benchmarks against a llama.cpp server."""

    if pp_only and tg_only:
        raise click.UsageError("--pp-only and --tg-only are mutually exclusive")

    # ------------------------------------------------------------------
    # 1. Config loading
    # ------------------------------------------------------------------
    if env_path is not None:
        cfg = parse_env_file(env_path)
        auto_label = env_label(env_path)
        config_file_ref = str(env_path)
    elif config_path is not None:
        cfg = load_config(config_path)
        auto_label = Path(config_path).stem
        config_file_ref = str(config_path)
    else:
        # Try default YAML, then fall back to bare defaults
        default_yaml = Path("./llm-bench.yaml")
        if default_yaml.exists():
            cfg = load_config(default_yaml)
            auto_label = "default"
            config_file_ref = str(default_yaml)
        else:
            cfg = load_config(None)
            auto_label = "default"
            config_file_ref = ""

    # ------------------------------------------------------------------
    # 2. CLI overrides
    # ------------------------------------------------------------------
    cfg = apply_overrides(
        cfg,
        model=model,
        image=image,
        threads=threads,
        ctx=ctx,
        port=port,
        label=label,
        kv_k=kv_k,
        kv_v=kv_v,
    )

    if label is not None:
        result_label = label
    else:
        result_label = cfg.get("label", auto_label)

    # Resolve speed-specific settings
    speed_cfg = cfg.get("speed", {})
    effective_port = cfg["server"]["port"]
    if server_url is None:
        server_url = f"http://localhost:{effective_port}"

    # PP lengths resolution
    if pp_lengths is not None:
        effective_pp_lengths = [int(x.strip()) for x in pp_lengths.split(",")]
    elif quick:
        effective_pp_lengths = [512, 1024]
    else:
        effective_pp_lengths = speed_cfg.get("pp_lengths", [512, 1024, 4096, 16384])

    pp_runs = speed_cfg.get("pp_runs", 3)
    if quick:
        pp_runs = 2

    warmup = speed_cfg.get("warmup_runs", 1)
    corpus_path = speed_cfg.get(
        "corpus",
        cfg.get("quality", {}).get(
            "corpus",
            "benchmarks/perplexity/wikitext-2-raw/wiki.test.raw",
        ),
    )

    # Server flags and matrix
    server_args = cfg.get("server_args", {})
    flags = server_args_to_flags(server_args)
    flags_str = server_args_to_string(server_args)
    matrix = extract_matrix(cfg)

    docker_image = cfg["server"]["image"]
    model_dir = cfg["server"].get("model_dir", "./models")
    thread_count = server_args.get("threads", 20)

    # Output directory
    output_dir = cfg.get("output", {}).get("dir", "./benchmarks/matrix")
    if output_path is not None:
        # If output_path is a directory, use it as output_dir
        if os.path.isdir(output_path) or output_path.endswith("/"):
            output_dir = output_path
            output_path = None  # let save_result generate the filename

    # ------------------------------------------------------------------
    # 3. Banner
    # ------------------------------------------------------------------
    click.echo(
        f"Speed benchmark: {matrix['quant']} / kv={matrix['kv']} / ctx={matrix['ctx']}"
    )
    if verbose:
        click.echo(f"  Config: {config_file_ref}")
        click.echo(f"  Image:  {docker_image}")
        click.echo(f"  Flags:  {flags_str}")
        click.echo(f"  Label:  {result_label}")
    click.echo()

    # ------------------------------------------------------------------
    # 4. Dry run
    # ------------------------------------------------------------------
    if dry_run:
        click.echo("=== DRY RUN ===")
        click.echo(f"  Docker image: {docker_image}")
        click.echo(f"  Server flags: {flags_str}")
        click.echo(f"  Model dir:    {os.path.abspath(model_dir)}")
        click.echo(f"  Port:         {effective_port}")
        click.echo(f"  Server URL:   {server_url}")
        click.echo()

        docker_cmd = (
            f"docker run -d --gpus all --ipc host "
            f"-v {os.path.abspath(model_dir)}:/models "
            f"-p {effective_port}:{effective_port} "
            f"{docker_image} {flags_str}"
        )
        click.echo(f"  Docker command:\n    {docker_cmd}")
        click.echo()

        if not tg_only:
            click.echo(
                f"  PP workloads: {effective_pp_lengths} "
                f"({pp_runs} runs each, {warmup} warmup)"
            )
            click.echo(f"  PP corpus: {corpus_path}")
            click.echo(f"  PP API: POST {server_url}/completion")
            click.echo(
                f"  PP body: {{\"prompt\": \"<{effective_pp_lengths[0]}*5 chars>\", "
                f"\"n_predict\": 0, \"temperature\": 0.0, \"cache_prompt\": false}}"
            )
            click.echo()

        if not pp_only:
            tg_runs = speed_cfg.get("tg_runs", 5)
            if quick:
                tg_runs = 3
            workloads = speed_cfg.get(
                "tg_workloads", ["short", "medium", "long", "multi_turn"],
            )
            if quick:
                workloads = [w for w in workloads if w != "multi_turn"]
            click.echo(
                f"  TG workloads: {workloads} ({tg_runs} runs each, {warmup} warmup)"
            )
            click.echo(f"  TG API: POST {server_url}/v1/chat/completions")
            click.echo(
                f"  TG body: {{\"model\": \"benchmark\", \"messages\": [...], "
                f"\"max_tokens\": {speed_cfg.get('tg_max_tokens', 256)}, "
                f"\"temperature\": {speed_cfg.get('tg_temperature', 0.7)}}}"
            )

        return

    # ------------------------------------------------------------------
    # 5. Docker lifecycle (unless --skip-server)
    # ------------------------------------------------------------------
    container_name: str | None = None

    if not skip_server:
        if not check_port_available(effective_port):
            raise click.ClickException(
                f"Port {effective_port} is already in use. "
                f"Use --skip-server if the server is already running, "
                f"or --port to pick another port."
            )

        click.echo("Starting server container...")
        container_name = start_container(
            image=docker_image,
            server_flags=flags,
            port=effective_port,
            model_dir=model_dir,
            container_prefix=cfg["server"].get("container_prefix", "llm-bench"),
            verbose=verbose,
        )
        click.echo(f"  Container: {container_name}")

        health_timeout = cfg["server"].get("health_timeout", 600)
        health_interval = cfg["server"].get("health_interval", 5)
        click.echo(
            f"  Waiting for health (timeout {health_timeout}s)..."
        )
        if not wait_for_health(
            port=effective_port,
            timeout=health_timeout,
            interval=health_interval,
            verbose=verbose,
        ):
            raise click.ClickException(
                "Server failed to become healthy within timeout. "
                "Check Docker logs for details."
            )
        click.echo("  Server healthy.")
        click.echo()

    # ------------------------------------------------------------------
    # 6. VRAM capture after load
    # ------------------------------------------------------------------
    vram_after_load = capture_vram()
    gpu_stats: dict[str, Any] = {}
    if vram_after_load is not None:
        gpu_stats["after_load"] = vram_after_load.to_dict()
        click.echo(f"  VRAM after load: {vram_after_load.used_mb} MB")
        click.echo()

    # ------------------------------------------------------------------
    # 7. Run benchmarks
    # ------------------------------------------------------------------
    pp_benchmarks: dict[str, Any] = {}
    tg_benchmarks: dict[str, Any] = {}

    try:
        # PP benchmarks
        if not tg_only:
            if not os.path.isfile(corpus_path):
                raise click.ClickException(
                    f"Corpus file not found: {corpus_path}. "
                    f"Needed for PP measurement. Use --tg-only to skip PP."
                )
            pp_benchmarks = _run_pp_benchmarks(
                server_url=server_url,
                corpus_path=corpus_path,
                pp_lengths=effective_pp_lengths,
                pp_runs=pp_runs,
                warmup=warmup,
                verbose=verbose,
            )
            click.echo()

        # TG benchmarks
        if not pp_only:
            tg_benchmarks = _run_tg_benchmarks(
                server_url=server_url,
                cfg=cfg,
                quick=quick,
                verbose=verbose,
            )
            click.echo()

    finally:
        # ------------------------------------------------------------------
        # 8. VRAM capture after bench + stop container
        # ------------------------------------------------------------------
        vram_after_bench = capture_vram()
        if vram_after_bench is not None:
            gpu_stats["after_bench"] = vram_after_bench.to_dict()

        if container_name is not None:
            click.echo("Stopping server container...")
            stop_container(container_name)

    # ------------------------------------------------------------------
    # 9. Assemble and save result
    # ------------------------------------------------------------------
    result = create_result(
        label=result_label,
        matrix=matrix,
        config_file=config_file_ref,
        server_args=flags_str,
        docker_image=docker_image,
        thread_count=thread_count,
    )
    result["gpu_stats"] = gpu_stats
    result["pp_benchmarks"] = pp_benchmarks
    result["tg_benchmarks"] = tg_benchmarks

    if output_path is not None and not os.path.isdir(output_path):
        # Explicit file path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        import json
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        saved_path = output_path
    else:
        saved_path = save_result(result, output_dir=output_dir)

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------
    pp_means = [
        str(v.get("mean_tps", 0))
        for v in pp_benchmarks.values()
    ]
    tg_means = [
        str(v.get("mean_tps", 0))
        for v in tg_benchmarks.values()
    ]
    vram_str = ""
    if "after_load" in gpu_stats:
        vram_str = f" VRAM={gpu_stats['after_load']['used_mb']} MB"

    summary_parts = []
    if pp_means:
        summary_parts.append(f"PP={'/'.join(pp_means)}")
    if tg_means:
        summary_parts.append(f"TG={'/'.join(tg_means)}")

    click.echo(f"RESULT: {' '.join(summary_parts)}{vram_str}")
    click.echo(f"Results saved to {saved_path}")
