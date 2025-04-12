import gc
import time
import torch
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
)

def terminate_server(server_process, gpu_index=0, threshold_mb=1024, wait_time=10):
    """
    Terminates a process and checks if GPU memory usage is below a threshold.

    Args:
        vllm_process (subprocess.Popen): The vLLM process to terminate.
        gpu_index (int): GPU index to monitor.
        threshold_mb (int): Memory usage threshold in MB.
        wait_time (int): Time to wait after termination, in seconds.

    Raises:
        RuntimeError: If GPU memory usage is still above the threshold after termination.

    Returns:
        dict: {
            'vllm_running': bool,
            'gpu_memory_used_mb': float,
            'below_threshold': bool
        }
    """
    # Terminate the process
    server_process.terminate()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(wait_time)

    # Check if process is still running
    is_running = server_process.poll() is None

    # Check GPU memory usage
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()

    used_mb = mem_info.used / 1e6
    below_threshold = used_mb < threshold_mb

    print(f"vLLM running: {is_running}, GPU used: {used_mb:.2f} MB, Below threshold: {below_threshold}")

    if not below_threshold:
        raise RuntimeError(
            f"GPU memory usage is still high: {used_mb:.2f} MB. "
            f"Expected less than {threshold_mb} MB."
        )

    return {
        'vllm_running': is_running,
        'gpu_memory_used_mb': used_mb,
        'below_threshold': below_threshold
    }
