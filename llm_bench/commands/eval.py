"""llm-bench eval — lm-eval-harness wrapper for generation benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import click

from ..core.config import load_config, apply_overrides, server_args_to_flags
from ..core.docker import (
    start_container,
    stop_container,
    wait_for_health,
    check_port_available,
)


@click.command("eval")
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Config YAML path")
@click.option("--tasks", required=True, help="Comma-separated lm-eval tasks")
@click.option("--limit", type=int, default=200, help="Samples per task")
@click.option("--model", default=None, help="Model path override")
@click.option("--hf-model", default=None, help="HF model name for tokenizer")
@click.option("--skip-server", is_flag=True, help="Assume server already running")
@click.option("--server-url", default=None, help="Server URL override")
@click.option("--no-thinking", is_flag=True, help="Disable thinking mode")
@click.option("--max-gen-toks", type=int, default=256, help="Max generation tokens")
@click.option("--output-dir", default=None, help="Output directory for results")
@click.option("--dry-run", is_flag=True, help="Print lm-eval command without running")
@click.option("--verbose", is_flag=True, help="Show detailed output")
def eval_cmd(config_path, tasks, limit, model, hf_model, skip_server,
             server_url, no_thinking, max_gen_toks, output_dir, dry_run, verbose):
    """Run lm-eval-harness generation benchmarks against the server."""

    # 1. Load config
    cfg = load_config(config_path)
    if model:
        cfg = apply_overrides(cfg, model=model)

    server_cfg = cfg.get("server", {})
    server_args = cfg.get("server_args", {})
    port = server_cfg.get("port", 8080)

    # Resolve server URL
    if server_url is None:
        server_url = f"http://localhost:{port}"

    # Resolve HF model name (needed for tokenizer in lm-eval)
    if hf_model is None:
        # Try to infer from model_base in quality config
        model_base = cfg.get("quality", {}).get("model_base", "")
        if model_base:
            hf_model = f"Qwen/{model_base}"
        else:
            hf_model = "Qwen/Qwen3.5-35B-A3B"
        if verbose:
            click.echo(f"  HF model inferred: {hf_model}")

    # Resolve output directory
    if output_dir is None:
        output_dir = os.path.join(
            cfg.get("output", {}).get("dir", "./benchmarks/matrix"),
            "evals",
        )

    container_name = None

    try:
        # 2. Start server if needed
        if not skip_server:
            if not check_port_available(port):
                raise click.ClickException(
                    f"Port {port} is in use. Stop existing server or use --skip-server."
                )

            click.echo("Starting llama-server container...")
            flags = server_args_to_flags(server_args)
            container_name = start_container(
                image=server_cfg.get("image", "llm-server/llama-cpp:latest-fit"),
                server_flags=flags,
                port=port,
                model_dir=server_cfg.get("model_dir", "./models"),
                container_prefix=server_cfg.get("container_prefix", "llm-bench"),
                verbose=verbose,
            )

            click.echo("Waiting for server health...")
            healthy = wait_for_health(
                port=port,
                timeout=server_cfg.get("health_timeout", 600),
                interval=server_cfg.get("health_interval", 5),
                verbose=verbose,
            )
            if not healthy:
                raise click.ClickException(
                    "Server failed to become healthy within timeout."
                )
            click.echo("Server ready.")
        else:
            click.echo(f"Using existing server at {server_url}")

        click.echo()

        # 3. Build lm-eval command
        model_args = (
            f"model={hf_model},"
            f"base_url={server_url}/v1/completions,"
            f"tokenized_requests=False,"
            f"num_concurrent=1,"
            f"max_gen_toks={max_gen_toks}"
        )

        cmd = [
            "lm_eval",
            "--model", "local-completions",
            "--model_args", model_args,
            "--tasks", tasks,
            "--limit", str(limit),
            "--output_path", output_dir,
            "--log_samples",
        ]

        # Note: --chat_template_kwargs only works with chat completions models,
        # not local-completions. Include it if requested but log a warning.
        if no_thinking:
            click.echo(
                "WARNING: --no-thinking adds chat_template_kwargs but this only "
                "works with hf-chat/local-chat-completions, not local-completions. "
                "The flag may have no effect."
            )
            cmd.extend([
                "--chat_template_kwargs",
                json.dumps({"enable_thinking": False}),
            ])

        # 4. Dry run — print and exit
        if dry_run:
            click.echo("Dry run — command that would be executed:")
            click.echo()
            # Format as a readable multi-line command
            parts = []
            i = 0
            while i < len(cmd):
                if cmd[i].startswith("--"):
                    if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                        parts.append(f"  {cmd[i]} {cmd[i + 1]}")
                        i += 2
                        continue
                    else:
                        parts.append(f"  {cmd[i]}")
                else:
                    parts.append(f"  {cmd[i]}")
                i += 1
            click.echo(" \\\n".join(parts))
            click.echo()
            click.echo(f"Output directory: {output_dir}")
            return

        # 5. Run lm-eval (live output, not captured)
        click.echo(f"Running lm-eval: tasks={tasks}, limit={limit}, max_gen_toks={max_gen_toks}")
        click.echo(f"Output: {output_dir}")
        click.echo()

        if verbose:
            click.echo(f"  Command: {' '.join(cmd)}")
            click.echo()

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            raise click.ClickException(
                f"lm-eval exited with code {result.returncode}"
            )

        click.echo()
        click.echo(f"Results saved to: {output_dir}")

    finally:
        # 6. Stop container if we started it
        if container_name and not skip_server:
            click.echo("Stopping server container...")
            stop_container(container_name)
            click.echo("Done.")
