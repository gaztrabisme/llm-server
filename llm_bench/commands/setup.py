"""llm-bench setup — auto-detect hardware and generate config."""

from __future__ import annotations

import glob
import os
import subprocess

import click
import yaml

from ..core.gpu import detect_gpu
from ..core.hardware import detect_cpu, detect_ram


@click.command()
@click.option("--output", default="llm-bench.yaml", help="Output config path")
@click.option("--force", is_flag=True, help="Overwrite existing config")
def setup(output, force):
    """Auto-detect hardware and generate benchmark config."""
    if os.path.exists(output) and not force:
        raise click.ClickException(
            f"{output} already exists. Use --force to overwrite."
        )

    click.echo("Hardware Detection:")

    # 1. Detect GPU
    gpu = detect_gpu()
    if gpu:
        click.echo(f"  GPU: {gpu.name} ({gpu.vram_total_mb} MB)")
    else:
        click.echo("  GPU: not detected")

    # 2. Detect CPU
    cpu = detect_cpu()
    click.echo(f"  CPU: {cpu.cores} cores ({cpu.model_name})")

    # 3. Detect RAM
    ram = detect_ram()
    ram_gb = ram.total_mb // 1024
    click.echo(f"  RAM: {ram_gb} GB")

    click.echo()

    # 4. Scan for GGUF models
    model_files = sorted(glob.glob("./models/*.gguf"))
    if model_files:
        click.echo("Models found:")
        for mf in model_files:
            size_gb = os.path.getsize(mf) / (1024 ** 3)
            click.echo(f"  - {os.path.basename(mf)} ({size_gb:.1f} GB)")
    else:
        click.echo("Models found: none (place .gguf files in ./models/)")

    click.echo()

    # 5. Check Docker
    docker_ok = False
    docker_version = ""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            docker_ok = True
            docker_version = result.stdout.strip()
            # Extract just the version number
            if "version" in docker_version.lower():
                parts = docker_version.split()
                for i, p in enumerate(parts):
                    if p.lower() == "version":
                        docker_version = parts[i + 1].rstrip(",") if i + 1 < len(parts) else docker_version
                        break
            click.echo(f"Docker: OK (v{docker_version})")
        else:
            click.echo("Docker: NOT FOUND")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        click.echo("Docker: NOT FOUND")

    # 6. Check NVIDIA Container Toolkit
    toolkit_ok = False
    if docker_ok:
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "--gpus", "all",
                    "nvidia/cuda:12.8.1-base-ubuntu24.04", "nvidia-smi",
                ],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                toolkit_ok = True
                click.echo("NVIDIA Toolkit: OK")
            else:
                click.echo("NVIDIA Toolkit: FAILED (docker --gpus all not working)")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            click.echo("NVIDIA Toolkit: FAILED (timeout or docker not found)")
    else:
        click.echo("NVIDIA Toolkit: SKIPPED (Docker not available)")

    click.echo()

    # 7. Generate config with hardware-appropriate defaults
    vram_mb = gpu.vram_total_mb if gpu else 0
    if vram_mb >= 16384:
        ctx = 65536
    elif vram_mb >= 8192:
        ctx = 32768
    else:
        ctx = 8192

    threads = min(cpu.cores, 20)

    model_path = ""
    if model_files:
        # Use the first found GGUF, store as relative Docker-mapped path
        model_path = f"/models/{os.path.basename(model_files[0])}"
    else:
        model_path = "/models/YOUR_MODEL.gguf"

    config = {
        "server": {
            "image": "llm-server/llama-cpp:latest-fit",
            "port": 8080,
            "model_dir": "./models",
            "health_timeout": 600,
            "health_interval": 5,
            "container_prefix": "llm-bench",
        },
        "server_args": {
            "model": model_path,
            "ctx_size": ctx,
            "threads": threads,
            "flash_attn": True,
            "no_mmap": True,
            "jinja": True,
            "fit": bool(gpu),
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "host": "0.0.0.0",
            "port": 8080,
        },
        "output": {
            "dir": "./benchmarks/matrix",
        },
        "speed": {
            "pp_lengths": [512, 1024, 4096, 16384],
            "pp_runs": 3,
            "tg_workloads": ["short", "medium", "long", "multi_turn"],
            "tg_runs": 5,
            "tg_max_tokens": 256,
            "tg_temperature": 0.7,
            "pp_temperature": 0.0,
            "warmup_runs": 1,
            "multi_turn_turns": 5,
            "multi_turn_truncate": 200,
        },
        "quality": {
            "corpus": "benchmarks/perplexity/wikitext-2-raw/wiki.test.raw",
            "reference_quant": "Q8_0",
            "model_base": "Qwen3.5-35B-A3B",
            "ctx": 512,
        },
    }

    # 8. Check WikiText-2 corpus
    corpus_path = config["quality"]["corpus"]
    if os.path.exists(corpus_path):
        click.echo(f"WikiText-2 corpus: OK ({corpus_path})")
    else:
        click.echo(f"WikiText-2 corpus: NOT FOUND ({corpus_path})")
        click.echo("  Download: https://huggingface.co/datasets/Salesforce/wikitext")

    click.echo()

    # 9. Write YAML config
    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Config written to {output}")

    # 10. Print next steps
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  1. Review and adjust {output}")
    click.echo("  2. llm-bench speed --quick    # Quick benchmark")
    click.echo("  3. llm-bench quality --ppl-only  # Perplexity check")

    if not model_files:
        click.echo()
        click.echo("WARNING: No .gguf models found in ./models/")
        click.echo("  Place your model files there before running benchmarks.")

    if not docker_ok:
        click.echo()
        click.echo("WARNING: Docker not found. Install Docker to run benchmarks.")

    if docker_ok and not toolkit_ok:
        click.echo()
        click.echo("WARNING: NVIDIA Container Toolkit not working.")
        click.echo("  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/")
