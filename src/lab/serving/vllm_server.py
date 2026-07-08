import shutil
import subprocess
import time

import httpx

from lab.config import LabConfig
from lab.exceptions import ServingError
from lab.logging import get_logger

logger = get_logger(__name__)


def build_vllm_command(config: LabConfig) -> list[str]:
    return [
        "vllm",
        "serve",
        config.serve.model_path,
        "--host",
        config.serve.host,
        "--port",
        str(config.serve.port),
        "--served-model-name",
        config.serve.served_model_name,
        "--max-model-len",
        str(config.serve.max_model_len),
        "--gpu-memory-utilization",
        str(config.serve.gpu_memory_utilization),
    ]


def wait_for_health(base_url: str, timeout_seconds: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    return False


def launch_vllm(config: LabConfig) -> "subprocess.Popen[bytes]":
    if shutil.which("vllm") is None:
        raise ServingError("vllm is not installed, run: uv sync --extra serve (linux + cuda only)")
    command = build_vllm_command(config)
    logger.info("vllm_starting", command=" ".join(command))
    process = subprocess.Popen(command)
    base_url = f"http://localhost:{config.serve.port}"
    if not wait_for_health(base_url):
        process.terminate()
        raise ServingError("vllm server did not become healthy in time")
    logger.info("vllm_ready", base_url=f"{base_url}/v1", model=config.serve.served_model_name)
    return process
