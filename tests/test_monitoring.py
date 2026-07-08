from unittest.mock import MagicMock, patch

from lab.monitoring.gpu import parse_smi_line, query_gpu

SMI_LINE = "2026/07/08 12:00:00.000, 87, 62, 32510, 40960, 71, 305.5"


def test_parse_smi_line() -> None:
    snapshot = parse_smi_line(SMI_LINE)
    assert snapshot.gpu_utilization_percent == 87.0
    assert snapshot.memory_used_mib == 32510.0
    assert snapshot.memory_total_mib == 40960.0
    assert snapshot.power_watts == 305.5


def test_query_gpu_parses_subprocess_output() -> None:
    completed = MagicMock()
    completed.stdout = SMI_LINE + "\n"
    with patch("lab.monitoring.gpu.subprocess.run", return_value=completed):
        snapshot = query_gpu()
    assert snapshot is not None
    assert snapshot.temperature_celsius == 71.0


def test_query_gpu_handles_missing_binary() -> None:
    with patch("lab.monitoring.gpu.subprocess.run", side_effect=FileNotFoundError):
        assert query_gpu() is None
