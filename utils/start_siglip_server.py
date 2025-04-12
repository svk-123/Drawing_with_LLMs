import uvicorn
import subprocess
import time
import sys
sys.path.append("./home/vino/ML_Projects/Drawing_with_LLMs/utils/")

app_dir = "/home/vino/ML_Projects/Drawing_with_LLMs/utils"

def start_siglip_server():
    # Start the server as a subprocess
    server_process = subprocess.Popen(
        ["uvicorn", "siglip_eval_server:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=app_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print('server starting...')
    # Give it a few seconds to boot up
    time.sleep(10)
    # Check if it's running
    print("Server running:", server_process.poll() is None)

    return server_process

if __name__ == "__main__":
    server_process=start_siglip_server()