#!/usr/bin/env python3
"""
Script to optimize system settings for better benchmarking results.
"""
import os
import sys
import argparse
import subprocess
import platform

def set_cpu_governor(governor="performance"):
    """Set CPU governor to improve performance consistency."""
    if platform.system() != "Linux":
        print("CPU governor can only be set on Linux systems.")
        return False
    
    try:
        # Check if we're running as root
        if os.geteuid() != 0:
            print("Setting CPU governor requires root privileges.")
            print(f"Try running: sudo cpupower frequency-set -g {governor}")
            return False
        
        # Get available governors
        result = subprocess.run(
            ["cpupower", "frequency-info", "-g"], 
            capture_output=True, 
            text=True
        )
        
        if governor not in result.stdout:
            print(f"Governor '{governor}' not available. Available governors:")
            print(result.stdout)
            return False
        
        # Set governor
        subprocess.run(["cpupower", "frequency-set", "-g", governor])
        print(f"CPU governor set to '{governor}'")
        return True
    
    except Exception as e:
        print(f"Error setting CPU governor: {str(e)}")
        return False

def disable_hyperthreading():
    """Disable hyperthreading for more consistent benchmarks."""
    if platform.system() != "Linux":
        print("Hyperthreading can only be disabled on Linux systems.")
        return False
    
    try:
        # Check if we're running as root
        if os.geteuid() != 0:
            print("Disabling hyperthreading requires root privileges.")
            print("Try running: sudo bash -c 'echo 0 > /sys/devices/system/cpu/smt/control'")
            return False
        
        # Check if SMT control exists
        if not os.path.exists("/sys/devices/system/cpu/smt/control"):
            print("SMT control not available on this system.")
            return False
        
        # Disable hyperthreading
        with open("/sys/devices/system/cpu/smt/control", "w") as f:
            f.write("0")
        
        print("Hyperthreading disabled.")
        return True
    
    except Exception as e:
        print(f"Error disabling hyperthreading: {str(e)}")
        return False

def optimize_memory_settings():
    """Optimize memory settings for better performance."""
    if platform.system() != "Linux":
        print("Memory settings can only be optimized on Linux systems.")
        return False
    
    try:
        # Check if we're running as root
        if os.geteuid() != 0:
            print("Optimizing memory settings requires root privileges.")
            return False
        
        # Drop caches
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        
        # Set swappiness to reduce swapping
        with open("/proc/sys/vm/swappiness", "w") as f:
            f.write("10")
        
        print("Memory settings optimized.")
        return True
    
    except Exception as e:
        print(f"Error optimizing memory settings: {str(e)}")
        return False

def main():
    """Run the optimization script with command-line arguments."""
    parser = argparse.ArgumentParser(description="Optimize system for benchmarking")
    parser.add_argument("--governor", type=str, choices=["performance", "powersave", "ondemand"],
                       default="performance", help="CPU governor to use")
    parser.add_argument("--no-ht", action="store_true",
                       help="Disable hyperthreading")
    parser.add_argument("--memory", action="store_true",
                       help="Optimize memory settings")
    
    args = parser.parse_args()
    
    # Check if we're running as root
    if os.geteuid() != 0:
        print("Warning: This script requires root privileges for most operations.")
        print("Try running with sudo.")
    
    # Apply requested optimizations
    if args.governor:
        set_cpu_governor(args.governor)
    
    if args.no_ht:
        disable_hyperthreading()
    
    if args.memory:
        optimize_memory_settings()
    
    print("\nSystem optimization complete.")
    print("Note: These settings may revert after system reboot.")

if __name__ == "__main__":
    main()
