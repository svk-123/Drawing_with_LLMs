import psutil

def list_fastapi_servers():
    fastapi_procs = []

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            cmdline = ' '.join(proc.info['cmdline']).lower()
            if any(keyword in cmdline for keyword in ['uvicorn', 'fastapi', 'hypercorn', 'gunicorn']):
                fastapi_procs.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'status': proc.info['status'],
                    'cmdline': cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return fastapi_procs

# Usage
servers = list_fastapi_servers()
if servers:
    for proc in servers:
        print(f"[PID {proc['pid']}] {proc['name']} ({proc['status']})\n    {proc['cmdline']}\n")
else:
    print("No FastAPI-related servers found.")