"""Custom workload YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_workloads(path: str | Path) -> dict[str, Any]:
    """Load custom workload definitions from a YAML file.

    Expected format:
    ```yaml
    workloads:
      my_task:
        messages:
          - role: user
            content: "Your prompt here"
        max_tokens: 512
    ```
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "workloads" not in data:
        raise ValueError(f"Invalid workload file: {path}. Must contain 'workloads' key.")

    return data["workloads"]
