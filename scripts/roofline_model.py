#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def get_cpu_peak_performance():
    """
    Estimate CPU peak theoretical performance based on system information.
    This is a rough estimate and could be made more accurate with detailed CPU specs.
    
    Returns estimated peak GFLOPS for float32.
    """
    try:
        # Get CPU model
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
        
        # Count physical cores
        physical_cores = os.cpu_count() // 2  # Assuming hyperthreading
        if physical_cores < 1:
            physical_cores = os.cpu_count()
        
        # Estimate frequency (GHz)
        freq = 0
        for line in cpu_info.split('\n'):
            if 'cpu MHz' in line:
                freq = float(line.split(':')[1].strip()) / 1000
                break
        
        if freq == 0:
            freq = 3.0  # Default assumption if can't detect
        
        # Check if AVX-512 is supported
        has_avx512 = 'avx512' in cpu_info.lower()
        
        # Estimate FLOPs per cycle per core
        if has_avx512:
            # AVX-512: 16 float32 operations per cycle
            flops_per_cycle = 16
        elif 'avx2' in cpu_info.lower():
            # AVX2: 8 float32 operations per cycle
            flops_per_cycle = 8
        elif 'avx' in cpu_info.lower():
            # AVX: 8 float32 operations per cycle (but less efficient than AVX2)
            flops_per_cycle = 8
        else:
            # SSE: 4 float32 operations per cycle
            flops_per_cycle = 4
        
        # Calculate peak GFLOPS
        peak_gflops = physical_cores * freq * flops_per_cycle
        
        print(f"Estimated CPU peak performance:")
        print(f"  Frequency: {freq:.2f} GHz")
        print(f"  Physical cores: {physical_cores}")
        print(f"  Vector instructions: {'AVX-512' if has_avx512 else 'AVX2/AVX/SSE'}")
        print(f"  Theoretical peak: {peak_gflops:.1f} GFLOPS (float32)")
        
        return peak_gflops
    
    except Exception as e:
        print(f"Error estimating CPU peak performance: {e}")
        print("Using default estimate of 500 GFLOPS")
        return 500  # Default estimate

def get_gpu_peak_performance():
    """
    Estimate GPU peak theoretical performance using CUDA.
    
    Returns estimated peak GFLOPS for float32.
    """
    try:
        # Try to get GPU information using nvidia-smi
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,cuda_version', '--format=csv,noheader'], 
                               stdout=subprocess.PIPE, text=True)
        gpu_info = result.stdout.strip()
        
        if not gpu_info:
            return None  # No GPU found
        
        print(f"GPU detected: {gpu_info}")
        
        # This is a very rough estimate - in reality, we would need to look up the specific GPU model
        # and get its specifications
        
        # For modern GPUs, a reasonable estimate might be:
        peak_gflops = 10000  # 10 TFLOPS is common for mid-range modern GPUs
        
        # Check for specific GPU models to improve estimate
        gpu_name = gpu_info.lower()
        
        # NVIDIA RTX 3000 series estimates
        if 'rtx 3090' in gpu_name:
            peak_gflops = 35600
        elif 'rtx 3080' in gpu_name:
            peak_gflops = 29800
        elif 'rtx 3070' in gpu_name:
            peak_gflops = 20400
        elif 'rtx 3060' in gpu_name:
            peak_gflops = 12700
            
        # NVIDIA RTX 2000 series estimates
        elif 'rtx 2080' in gpu_name:
            peak_gflops = 14200
        elif 'rtx 2070' in gpu_name:
            peak_gflops = 9000
        elif 'rtx 2060' in gpu_name:
            peak_gflops = 6500
            
        # NVIDIA 1000 series estimates
        elif 'gtx 1080' in gpu_name:
            peak_gflops = 9000
        elif 'gtx 1070' in gpu_name:
            peak_gflops = 6500
        elif 'gtx 1060' in gpu_name:
            peak_gflops = 4000
            
        # NVIDIA Quadro/Tesla estimates
        elif 'tesla v100' in gpu_name:
            peak_gflops = 14000
        elif 'tesla a100' in gpu_name:
            peak_gflops = 19500
        
        print(f"Estimated GPU peak performance: {peak_gflops:.1f} GFLOPS (float32)")
        
        return peak_gflops
    
    except Exception as e:
        print(f"Error estimating GPU peak performance: {e}")
        print("No GPU detected or nvidia-smi not available")
        return None

def get_memory_bandwidth():
    """
    Estimate memory bandwidth based on system information.
    This is a rough estimate and could be made more accurate with detailed specs.
    
    Returns estimated memory bandwidth in GB/s.
    """
    try:
        # Try to determine RAM type and speed from system info
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.read()
        
        # Total memory gives us a hint about the system class
        total_memory_kb = 0
        for line in mem_info.split('\n'):
            if 'MemTotal' in line:
                total_memory_kb = int(line.split(':')[1].strip().split(' ')[0])
                break
        
        total_memory_gb = total_memory_kb / (1024 * 1024)
        
        # Make a rough estimate based on system memory size
        # These are very approximate values
        if total_memory_gb >= 64:
            # High-end desktop or server with DDR4-3200 or better
            bandwidth = 50  # GB/s
        elif total_memory_gb >= 16:
            # Mid-range desktop with DDR4-2400 or similar
            bandwidth = 35  # GB/s
        else:
            # Entry-level system with DDR4-2133 or similar
            bandwidth = 25  # GB/s
        
        print(f"Estimated memory bandwidth: {bandwidth:.1f} GB/s")
        print(f"(Based on system with {total_memory_gb:.1f} GB RAM)")
        
        return bandwidth
    
    except Exception as e:
        print(f"Error estimating memory bandwidth: {e}")
        print("Using default estimate of 30 GB/s")
        return 30  # Default estimate

def calculate_operational_intensity(matrix_size):
    """
    Calculate operational intensity for matrix multiplication.
    
    Operational intensity = FLOPs / Memory accesses
    For matrix multiplication: 2*N^3 FLOPs / (3*N^2 * 4 bytes)
    
    Returns operational intensity in FLOPs/byte.
    """
    # Matrix multiplication: 2*N^3 FLOPs
    flops = 2 * matrix_size**3
    
    # Memory accesses: 3 matrices of size N^2, each element is 4 bytes (float32)
    bytes_accessed = 3 * matrix_size**2 * 4
    
    # Operational intensity
    intensity = flops / bytes_accessed
    
    return intensity

def plot_roofline_model(results_file, output_file=None):
    """
    Create a roofline model plot using benchmark results.
    
    Args:
        results_file: CSV file with benchmark results
        output_file: Output file for the plot (default: 'roofline_model.png')
    """
    # Set default output file
    if output_file is None:
        output_file = 'roofline_model.png'
    
    # Get system parameters
    peak_gflops_cpu = get_cpu_peak_performance()
    peak_gflops_gpu = get_gpu_peak_performance()
    memory_bandwidth = get_memory_bandwidth()  # GB/s
    
    # Read benchmark results
    df = pd.read_csv(results_file)
    
    # Calculate operational intensity for each matrix size
    sizes = df['Size'].unique()
    intensities = {size: calculate_operational_intensity(size) for size in sizes}
    
    # Create roofline plot
    plt.figure(figsize=(12, 8))
    
    # Define operational intensity range for the plot
    x_min = 0.1
    x_max = 100
    x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    
    # Memory-bound region
    y_memory = memory_bandwidth * x
    
    # Compute-bound region (CPU)
    y_cpu = np.ones_like(x) * peak_gflops_cpu
    
    # Plot the roofline
    plt.loglog(x, np.minimum(y_memory, y_cpu), 'b-', linewidth=2, label='CPU Roofline')
    
    # Add GPU roofline if available
    if peak_gflops_gpu:
        y_gpu = np.ones_like(x) * peak_gflops_gpu
        plt.loglog(x, np.minimum(y_memory, y_gpu), 'r-', linewidth=2, label='GPU Roofline')
    
    # Add points for each implementation and matrix size
    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', 'x', '+']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    marker_idx = 0
    
    # Group by implementation
    for impl, group in df.groupby('Implementation'):
        # Skip implementations with very low performance (outliers)
        if group['PeakGFLOPS'].max() < 1:
            continue
        
        # Plot each matrix size
        for size in sizes:
            # Get the result for this implementation and size
            row = group[group['Size'] == size]
            if len(row) == 0:
                continue
                
            # Get peak GFLOPS
            gflops = row['PeakGFLOPS'].values[0]
            
            # Get operational intensity
            intensity = intensities[size]
            
            # Plot the point
            plt.loglog(intensity, gflops, marker=markers[marker_idx % len(markers)], 
                      color=colors[marker_idx % len(colors)], 
                      markersize=8, label=f'{impl} ({size}x{size})' if size == sizes[-1] else '')
        
        marker_idx += 1
    
    # Add labels and legend
    plt.xlabel('Operational Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Roofline Model for Matrix Multiplication Implementations')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add regions labels
    plt.text(0.2, memory_bandwidth * 0.2 * 0.5, 'Memory-Bound Region', 
             rotation=45, ha='left', va='bottom', alpha=0.7)
    plt.text(30, peak_gflops_cpu * 0.7, 'Compute-Bound Region (CPU)', 
             ha='center', va='center', alpha=0.7)
    
    # Set axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(0.1, peak_gflops_gpu * 1.1 if peak_gflops_gpu else peak_gflops_cpu * 1.1)
    
    # Create the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize='small')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Roofline model saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a roofline model plot from benchmark results')
    parser.add_argument('results_file', help='CSV file with benchmark results')
    parser.add_argument('--output', '-o', help='Output file for the plot (default: roofline_model.png)')
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found")
        sys.exit(1)
    
    # Plot the roofline model
    plot_roofline_model(args.results_file, args.output)

if __name__ == "__main__":
    main()
