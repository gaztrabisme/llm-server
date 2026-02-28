"""Statistical computation for benchmark runs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class RunStats:
    mean: float
    stddev: float
    n: int

    def to_dict(self) -> dict:
        return {"mean_tps": round(self.mean, 1), "stddev_tps": round(self.stddev, 1)}


def compute_stats(
    values: list[float],
    warmup: int = 1,
) -> RunStats:
    """Compute mean and stddev, skipping warmup runs.

    Args:
        values: Raw TPS measurements from all runs.
        warmup: Number of initial runs to skip (default: 1).
    """
    measured = [v for v in values[warmup:] if v > 0]
    if not measured:
        return RunStats(mean=0.0, stddev=0.0, n=0)
    mean = statistics.mean(measured)
    stddev = statistics.stdev(measured) if len(measured) >= 2 else 0.0
    return RunStats(mean=mean, stddev=stddev, n=len(measured))
