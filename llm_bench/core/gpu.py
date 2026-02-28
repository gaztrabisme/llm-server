"""GPU detection and VRAM capture via nvidia-smi."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    driver_version: str
    cuda_version: str


@dataclass
class VramSnapshot:
    used_mb: int
    total_mb: int
    utilization_pct: int

    def to_dict(self) -> dict:
        return {
            "used_mb": self.used_mb,
            "total_mb": self.total_mb,
            "utilization_pct": self.utilization_pct,
        }


def detect_gpu() -> GpuInfo | None:
    """Detect GPU via nvidia-smi. Returns None if not available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None

        # Get CUDA version separately
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        # Parse CUDA version from nvidia-smi header
        cuda_ver = ""
        header_result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        for hline in header_result.stdout.split("\n"):
            if "CUDA Version" in hline:
                idx = hline.index("CUDA Version:")
                cuda_ver = hline[idx + 14:].strip().rstrip("|").strip()
                break

        return GpuInfo(
            name=parts[0],
            vram_total_mb=int(float(parts[1])),
            driver_version=parts[2],
            cuda_version=cuda_ver,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def capture_vram() -> VramSnapshot | None:
    """Capture current VRAM usage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None

        return VramSnapshot(
            used_mb=int(float(parts[0])),
            total_mb=int(float(parts[1])),
            utilization_pct=int(float(parts[2])),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
