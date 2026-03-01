"""Parse legacy .env config files (LLAMA_ARG_* format) into config dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import DEFAULTS, _deep_copy_defaults


# Mapping from LLAMA_ARG_* env vars to server_args keys
ENV_TO_KEY = {
    "LLAMA_ARG_MODEL": "model",
    "LLAMA_ARG_CTX_SIZE": "ctx_size",
    "LLAMA_ARG_THREADS": "threads",
    "LLAMA_ARG_FLASH_ATTN": "flash_attn",
    "LLAMA_ARG_NO_MMAP": "no_mmap",
    "LLAMA_ARG_JINJA": "jinja",
    "LLAMA_ARG_FIT": "fit",
    "LLAMA_ARG_HOST": "host",
    "LLAMA_ARG_PORT": "port",
    "LLAMA_ARG_CACHE_TYPE_K": "cache_type_k",
    "LLAMA_ARG_CACHE_TYPE_V": "cache_type_v",
    "LLAMA_ARG_N_GPU_LAYERS": "n_gpu_layers",
    "LLAMA_ARG_BATCH_SIZE": "batch_size",
    "LLAMA_ARG_UBATCH_SIZE": "ubatch_size",
    "LLAMA_ARG_OFFLOAD_TENSORS": "offload_tensors",
    "LLAMA_ARG_FIT_TARGET": "fit_target",
    "LLAMA_ARG_FIT_CTX": "fit_ctx",
    "LLAMA_ARG_FUSE_GATE_UP_EXPS": "fuse_gate_up_exps",
    "LLAMA_ARG_N_CPU_MOE": "n_cpu_moe",
    "LLAMA_ARG_MMPROJ": "mmproj",
}

# Keys that should be parsed as booleans
BOOL_KEYS = {"flash_attn", "no_mmap", "jinja", "fit", "fuse_gate_up_exps"}

# Keys that should be parsed as integers
INT_KEYS = {"ctx_size", "threads", "port", "n_gpu_layers", "batch_size", "ubatch_size", "fit_target", "fit_ctx", "n_cpu_moe"}


def parse_env_file(path: str | Path) -> dict[str, Any]:
    """Parse a .env file into a full config dict.

    Returns a config dict compatible with load_config() output.
    """
    cfg = _deep_copy_defaults()
    env_vars: dict[str, str] = {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            env_vars[key] = value

    # Map env vars to server_args
    for env_key, cfg_key in ENV_TO_KEY.items():
        if env_key in env_vars:
            value = env_vars[env_key]
            if cfg_key in BOOL_KEYS:
                cfg["server_args"][cfg_key] = value.lower() in ("true", "1", "on", "yes")
            elif cfg_key in INT_KEYS:
                try:
                    cfg["server_args"][cfg_key] = int(value)
                except ValueError:
                    cfg["server_args"][cfg_key] = value
            else:
                cfg["server_args"][cfg_key] = value

    # Handle Docker image override
    if "DOCKER_IMAGE" in env_vars:
        cfg["server"]["image"] = env_vars["DOCKER_IMAGE"]

    # Handle extra args
    if "IK_EXTRA_ARGS" in env_vars:
        cfg["extra_args"] = env_vars["IK_EXTRA_ARGS"]

    # Store the env file path for reference
    cfg["_env_file"] = str(path)

    return cfg


def env_label(path: str | Path) -> str:
    """Extract a label from an env file path.

    Strips 'llama-cpp-' prefix and '.env' suffix.
    """
    name = Path(path).stem
    for prefix in ("llama-cpp-", "ik-llama-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name
