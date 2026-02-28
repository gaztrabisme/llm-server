"""CPU and RAM detection from /proc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CpuInfo:
    cores: int
    model_name: str


@dataclass
class RamInfo:
    total_mb: int


def detect_cpu() -> CpuInfo:
    """Detect CPU info from /proc/cpuinfo."""
    cores = 0
    model_name = "unknown"
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.split("\n"):
            if line.startswith("processor"):
                cores += 1
            elif line.startswith("model name") and model_name == "unknown":
                model_name = line.split(":", 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        import os
        cores = os.cpu_count() or 1
    return CpuInfo(cores=cores, model_name=model_name)


def detect_ram() -> RamInfo:
    """Detect total RAM from /proc/meminfo."""
    try:
        text = Path("/proc/meminfo").read_text()
        for line in text.split("\n"):
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return RamInfo(total_mb=kb // 1024)
    except (FileNotFoundError, PermissionError):
        pass
    return RamInfo(total_mb=0)
