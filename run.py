import os
import sys
import subprocess
import webbrowser
import time

def main():
    print("=" * 60)
    print("       IRCTC TICKET CANCELLATION POLICY RAG RUNNER       ")
    print("=" * 60)
    
    # 1. Determine base path and venv path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base_dir, ".venv")
    
    if not os.path.exists(venv_dir):
        print(f"Error: Virtual environment not found at {venv_dir}.")
        print("Please run '.venv\\Scripts\\pip install -r requirements.txt' or recreate the venv.")
        sys.exit(1)
        
    # Choose python executable from venv
    if sys.platform == "win32":
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        uvicorn_exe = os.path.join(venv_dir, "Scripts", "uvicorn.exe")
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")
        uvicorn_exe = os.path.join(venv_dir, "bin", "uvicorn")
        
    if not os.path.exists(python_exe):
        print(f"Error: Python executable not found at {python_exe}.")
        sys.exit(1)

    print("[1/3] Verifying requirements and configurations...")
    print(f"Project directory: {base_dir}")
    print(f"Using Virtual Env: {python_exe}\n")
    
    # 2. Check if uvicorn is installed, if not fallback to run via python module
    use_module_run = False
    if not os.path.exists(uvicorn_exe):
        print("Uvicorn executable not found. Running as python module...")
        use_module_run = True

    # 3. Inform the user and start the server
    print("[2/3] Starting FastAPI Server...")
    print("The RAG Engine will load the SentenceTransformer model on startup.")
    print("This might take 10-15 seconds on the very first launch as the embedding model downloads.")
    print("-" * 60)
    
    # Start server in a subprocess
    if use_module_run:
        cmd = [python_exe, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    else:
        cmd = [uvicorn_exe, "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
        
    try:
        # Start server process
        server_process = subprocess.Popen(cmd, cwd=base_dir)
        
        # 4. Open browser automatically after a short delay
        print("\n[3/3] Opening dashboard in your web browser...")
        print("Dashboard URL: http://127.0.0.1:8000")
        print("Press Ctrl+C in this terminal window to stop the server.")
        print("-" * 60)
        
        time.sleep(3) # Wait for FastAPI to initialize
        webbrowser.open("http://127.0.0.1:8000")
        
        # Keep process running
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\n[!] Stopping FastAPI Server...")
        server_process.terminate()
        server_process.wait()
        print("[x] Server stopped successfully. Goodbye!")
    except Exception as e:
        print(f"\n[!] Error starting server: {str(e)}")
        if 'server_process' in locals() and server_process:
            server_process.terminate()

if __name__ == "__main__":
    main()
