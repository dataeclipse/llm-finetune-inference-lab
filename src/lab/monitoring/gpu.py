import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from lab.logging import get_logger

logger = get_logger(__name__)

_QUERY_FIELDS = (
    "timestamp",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


@dataclass
class GpuSnapshot:
    timestamp: str
    gpu_utilization_percent: float
    memory_utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_celsius: float
    power_watts: float


def parse_smi_line(line: str) -> GpuSnapshot:
    parts = [part.strip() for part in line.split(",")]
    return GpuSnapshot(
        timestamp=parts[0],
        gpu_utilization_percent=float(parts[1]),
        memory_utilization_percent=float(parts[2]),
        memory_used_mib=float(parts[3]),
        memory_total_mib=float(parts[4]),
        temperature_celsius=float(parts[5]),
        power_watts=float(parts[6]),
    )


def query_gpu() -> GpuSnapshot | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    return parse_smi_line(first_line) if first_line else None


def sample_to_csv(path: Path, duration_seconds: float, interval_seconds: float = 5.0) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = 0
    deadline = time.monotonic() + duration_seconds
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(GpuSnapshot.__dataclass_fields__.keys())
        while time.monotonic() < deadline:
            snapshot = query_gpu()
            if snapshot is None:
                logger.warning("gpu_unavailable")
                break
            writer.writerow(
                [
                    snapshot.timestamp,
                    snapshot.gpu_utilization_percent,
                    snapshot.memory_utilization_percent,
                    snapshot.memory_used_mib,
                    snapshot.memory_total_mib,
                    snapshot.temperature_celsius,
                    snapshot.power_watts,
                ]
            )
            samples += 1
            time.sleep(interval_seconds)
    logger.info("gpu_sampling_finished", samples=samples, path=str(path))
    return samples
