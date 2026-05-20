"""YAML config loading, defaults, and CLI override merging."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULTS = {
    "server": {
        "image": "llm-server/llama-cpp:latest-fit",
        "port": 8080,
        "model_dir": "./models",
        "health_timeout": 600,
        "health_interval": 5,
        "container_prefix": "llm-bench",
    },
    "server_args": {
        "ctx_size": 65536,
        "threads": 20,
        "flash_attn": True,
        "no_mmap": True,
        "jinja": True,
        "fit": True,
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

# Mapping from server_args keys to llama-server CLI flags
FLAG_MAP = {
    "model": "-m",
    "ctx_size": "-c",
    "threads": "-t",
    "flash_attn": "-fa",
    "no_mmap": "--no-mmap",
    "jinja": "--jinja",
    "fit": "--fit",
    "host": "--host",
    "port": "--port",
    "cache_type_k": "-ctk",
    "cache_type_v": "-ctv",
    "n_gpu_layers": "-ngl",
    "offload_tensors": "-ot",
    "batch_size": "-b",
    "ubatch_size": "-ub",
    "mmproj": "--mmproj",
    "fit_target": "--fit-target",
    "fit_ctx": "--fit-ctx",
    "fuse_gate_up_exps": "--fuse-gate-up-exps",
    "n_cpu_moe": "--n-cpu-moe",
    "mlock": "--mlock",
    "no_warmup": "--no-warmup",
    "parallel": "-np",
    "spec_type": "--spec-type",
    "draft_block_size": "--draft-block-size",
    "cache_type_k_draft": "-ctkd",
    "cache_type_v_draft": "-ctvd",
    "ctx_checkpoints": "-ctxcp",
    "chat_template_kwargs": "--chat-template-kwargs",
    "max_tokens": "-n",
}

# Boolean flags that take "on"/"off" values (not just presence/absence)
BOOL_ON_OFF_FLAGS = {"flash_attn", "fit"}

# Boolean flags that are just presence flags (no value)
BOOL_PRESENCE_FLAGS = {"no_mmap", "jinja", "fuse_gate_up_exps", "mlock", "no_warmup"}


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load config from YAML file, merged with defaults."""
    cfg = _deep_copy_defaults()
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                user_cfg = yaml.safe_load(f) or {}
            _deep_merge(cfg, user_cfg)
    return cfg


def apply_overrides(cfg: dict[str, Any], **overrides) -> dict[str, Any]:
    """Apply CLI flag overrides to config.

    Supported overrides: model, image, threads, ctx, port, label, etc.
    None values are skipped.
    """
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "model":
            # Bare filename → prepend /models/ (Docker mount point)
            if "/" not in str(value):
                value = f"/models/{value}"
            cfg["server_args"]["model"] = value
        elif key == "image":
            cfg["server"]["image"] = value
        elif key == "threads":
            cfg["server_args"]["threads"] = value
        elif key == "ctx":
            cfg["server_args"]["ctx_size"] = value
        elif key == "port":
            cfg["server"]["port"] = value
            cfg["server_args"]["port"] = value
        elif key == "label":
            cfg["label"] = value
        elif key == "output":
            cfg["output"]["dir"] = value
        elif key == "kv_k":
            cfg["server_args"]["cache_type_k"] = value
        elif key == "kv_v":
            cfg["server_args"]["cache_type_v"] = value
        elif key == "batch":
            cfg["server_args"]["batch_size"] = value
        elif key == "ubatch":
            cfg["server_args"]["ubatch_size"] = value
        elif key == "fit_target":
            cfg["server_args"]["fit_target"] = value
        elif key == "fit_ctx":
            cfg["server_args"]["fit_ctx"] = value
        elif key == "n_cpu_moe":
            cfg["server_args"]["n_cpu_moe"] = value
        elif key == "fuse_gate_up_exps":
            cfg["server_args"]["fuse_gate_up_exps"] = value
    return cfg


def server_args_to_flags(server_args: dict[str, Any]) -> list[str]:
    """Convert server_args dict to llama-server CLI flag list."""
    flags: list[str] = []
    for key, value in server_args.items():
        flag = FLAG_MAP.get(key)
        if flag is None:
            continue
        if key in BOOL_ON_OFF_FLAGS:
            if value:
                flags.extend([flag, "on"])
        elif key in BOOL_PRESENCE_FLAGS:
            if value:
                flags.append(flag)
        else:
            flags.extend([flag, str(value)])
    return flags


def server_args_to_string(server_args: dict[str, Any]) -> str:
    """Convert server_args dict to a single CLI string."""
    return " ".join(server_args_to_flags(server_args))


def extract_matrix(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract matrix axes from config (quant, kv, batch, ctx)."""
    sa = cfg.get("server_args", {})
    model = sa.get("model", "")

    quant = _detect_quant(model)
    kv = _detect_kv(sa.get("cache_type_k", ""), sa.get("cache_type_v", ""))
    batch = _detect_batch(sa.get("batch_size"), sa.get("ubatch_size"))
    ctx = sa.get("ctx_size", 65536)

    return {"quant": quant, "kv": kv, "batch": batch, "ctx": ctx}


def resolve_model_path(cfg: dict[str, Any]) -> str | None:
    """Resolve model path from config, checking model_dir."""
    model = cfg.get("server_args", {}).get("model")
    if model is None:
        return None
    # If it's already absolute or starts with /models (Docker path), return as-is
    if os.path.isabs(model):
        return model
    # Try relative to model_dir
    model_dir = cfg.get("server", {}).get("model_dir", "./models")
    candidate = os.path.join(model_dir, model)
    if os.path.exists(candidate):
        return candidate
    return model


def _detect_quant(model_path: str) -> str:
    """Detect quantization from model filename."""
    name = os.path.basename(model_path).upper()

    # Check for MTP first — strip it, detect base quant, then append suffix
    mtp_suffix = ""
    if "MTP" in name:
        mtp_suffix = "-MTP"
        # Remove MTP (and surrounding separators) for base quant detection
        name = name.replace("-MTP-", "-").replace("-MTP.", ".").replace("-MTP", "").replace("MTP-", "")

    # UD quants (more specific patterns first)
    if "UD-IQ3_XXS" in name or "UD_IQ3_XXS" in name:
        return "UD-IQ3_XXS" + mtp_suffix
    if "UD-IQ2_XXS" in name or "UD_IQ2_XXS" in name:
        return "UD-IQ2_XXS" + mtp_suffix
    if "UD-Q2_K_XL" in name or "UD_Q2_K_XL" in name:
        return "UD-Q2_K_XL" + mtp_suffix
    if "UD-Q3_K_XL" in name or "UD_Q3_K_XL" in name:
        return "UD-Q3_K_XL" + mtp_suffix
    if "UD-Q4_K_XL" in name or "UD_Q4_K_XL" in name:
        return "UD-Q4_K_XL" + mtp_suffix
    # Non-UD IQ quants
    if "IQ3_XXS" in name:
        return "IQ3_XXS" + mtp_suffix
    if "IQ2_XXS" in name:
        return "IQ2_XXS" + mtp_suffix
    if "MXFP4" in name:
        return "MXFP4_MOE" + mtp_suffix
    if "Q4_K_L" in name:
        return "Q4_K_L" + mtp_suffix
    if "Q4_K_M" in name:
        return "Q4_K_M" + mtp_suffix
    if "Q8_0" in name:
        return "Q8_0" + mtp_suffix
    return "unknown" + mtp_suffix if mtp_suffix else "unknown"


def _detect_kv(k: str, v: str) -> str:
    """Detect KV cache config label."""
    if not k and not v:
        return "f16"
    if k == v:
        return k or "f16"
    return "asym"


def _detect_batch(batch: Any, ubatch: Any) -> str:
    """Detect batch config label."""
    if batch is None or batch == "":
        return "none"
    try:
        b = int(batch)
        ub = int(ubatch) if ubatch else b
    except (ValueError, TypeError):
        return "custom"
    if b >= 4096 and b == ub:
        return "large"
    if b >= 4096 and ub < b:
        return "asym"
    return "custom"


def _deep_copy_defaults() -> dict[str, Any]:
    """Deep copy the defaults dict."""
    import copy
    return copy.deepcopy(DEFAULTS)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
