#!/usr/bin/env python3
"""
Launcher script for the Parallelism Optimization Framework TUI.
This script starts the API server and launches the TUI client.
"""

import os
import sys
import subprocess
import time
import argparse
import signal
import logging
from threading import Thread
import atexit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("tui-launcher")

# Global variables
server_process = None
tui_process = None
api_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "api_server.py")
tui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tui", "target", "release", "parallel-opt-tui")
tui_debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tui", "target", "debug", "parallel-opt-tui")

def cleanup():
    """Clean up processes on exit."""
    logger.info("Cleaning up processes...")
    if server_process:
        logger.info("Terminating API server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    
    if tui_process and tui_process.poll() is None:
        logger.info("Terminating TUI client...")
        tui_process.terminate()

def start_api_server(host='127.0.0.1', port=8000, debug=False):
    """Start the API server in a subprocess."""
    global server_process
    
    if not os.path.exists(api_server_path):
        logger.error(f"API server not found at {api_server_path}")
        sys.exit(1)
    
    logger.info(f"Starting API server on {host}:{port}")
    cmd = [sys.executable, api_server_path, '--host', host, '--port', str(port)]
    if debug:
        cmd.append('--debug')
    
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not debug else None,
        stderr=subprocess.PIPE if not debug else None,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    logger.info("API server started with PID {}".format(server_process.pid))
    
    # Give the server a moment to start
    time.sleep(1.0)
    
    return server_process

def build_tui(release=True):
    """Build the Rust TUI application."""
    tui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tui")
    
    if not os.path.exists(tui_dir):
        logger.error(f"TUI directory not found at {tui_dir}")
        sys.exit(1)
    
    logger.info("Building TUI application...")
    
    build_cmd = ['cargo', 'build']
    if release:
        build_cmd.append('--release')
    
    build_process = subprocess.run(
        build_cmd,
        cwd=tui_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if build_process.returncode != 0:
        logger.error("Failed to build TUI application")
        logger.error(build_process.stderr.decode())
        sys.exit(1)
    
    logger.info("TUI application built successfully")

def start_tui(host='127.0.0.1', port=8000, release=True):
    """Start the TUI client."""
    global tui_process
    
    tui_path_to_use = tui_path if release else tui_debug_path
    
    if not os.path.exists(tui_path_to_use):
        logger.error(f"TUI client not found at {tui_path_to_use}")
        logger.info("Building TUI client...")
        build_tui(release)
    
    logger.info("Starting TUI client")
    cmd = [tui_path_to_use, '--api-url', f'http://{host}:{port}']
    
    tui_process = subprocess.Popen(cmd)
    logger.info("TUI client started with PID {}".format(tui_process.pid))
    
    return tui_process

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch the Parallelism Optimization Framework TUI")
    parser.add_argument('--host', default='127.0.0.1', help='Host for the API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-build', action='store_true', help='Skip building the TUI')
    parser.add_argument('--release', action='store_true', help='Use release build')
    args = parser.parse_args()
    
    # Register cleanup function
    atexit.register(cleanup)
    
    # Start the API server
    server_thread = Thread(target=start_api_server, args=(args.host, args.port, args.debug))
    server_thread.daemon = True
    server_thread.start()
    
    # Build the TUI if needed
    if not args.no_build:
        build_tui(args.release)
    
    # Start the TUI
    tui_process = start_tui(args.host, args.port, args.release)
    
    try:
        # Wait for the TUI to exit
        return_code = tui_process.wait()
        logger.info(f"TUI client exited with code {return_code}")
        
        # Clean up server
        cleanup()
        
        return return_code
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup()
        return 0

if __name__ == "__main__":
    sys.exit(main())
