import subprocess
import requests
import time
import os
import signal
import sys

def vllm_model_server_start(model_path):
    host = "127.0.0.1"
    port = "8000"

    # Build command to start vLLM server
    cmd = [
        "vllm", "serve",
        model_path,
        "--max-model-len=1024",
        "--dtype", "auto",
        "--api-key", "my-api-key",
        "--host", host,
        "--port", port
    ]

    # Start the server subprocess
    vllm_process = subprocess.Popen(cmd)
    print(f"vLLM server started with PID: {vllm_process.pid}")

    # Health check loop
    base_url = f"http://{host}:{port}/health"
    timeout = 60
    interval = 10
    elapsed = 0

    print("Waiting for vLLM model to load...")
    time.sleep(60)

    while elapsed < timeout:
        try:
            res = requests.get(base_url)
            if res.status_code == 200:
                print("vLLM server is ready.")
                break
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(interval)
        elapsed += interval
        print(f"Checked after {elapsed}s...")

    else:
        print("Timeout: vLLM server did not become ready in time.")

    return vllm_process  # Return process to allow later control (stop/kill)


# Example usage
if __name__ == "__main__":
    vllm_process = vllm_model_server_start("../lora/lora_16bit_merged_3b_r128_s1000_i1000_v1")

