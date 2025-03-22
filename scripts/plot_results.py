#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_performance_comparison(csv_file):
    """
    Plot performance comparison between different implementations.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a directory for the plots
    os.makedirs('plots', exist_ok=True)
    
    # Get unique matrix sizes
    sizes = df['Size'].unique()
    
    # Plot GFLOPS for each matrix size
    for size in sizes:
        # Filter data for the current size
        df_size = df[df['Size'] == size]
        
        # Sort by peak GFLOPS
        df_size = df_size.sort_values(by='PeakGFLOPS', ascending=False)
        
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        
        # Create bar plot for both average and peak GFLOPS
        implementations = df_size['Implementation']
        avg_gflops = df_size['AvgGFLOPS']
        peak_gflops = df_size['PeakGFLOPS']
        
        x = np.arange(len(implementations))
        width = 0.35
        
        plt.bar(x - width/2, avg_gflops, width, label='Average GFLOPS')
        plt.bar(x + width/2, peak_gflops, width, label='Peak GFLOPS')
        
        plt.xlabel('Implementation')
        plt.ylabel('GFLOPS (higher is better)')
        plt.title(f'Matrix Multiplication Performance ({size}x{size})')
        plt.xticks(x, implementations, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/performance_comparison_{size}.png')
        plt.close()
    
    # Plot scaling across matrix sizes for each implementation
    implementations = df['Implementation'].unique()
    
    plt.figure(figsize=(12, 6))
    
    for impl in implementations:
        # Filter data for the current implementation
        df_impl = df[df['Implementation'] == impl]
        
        # Sort by matrix size
        df_impl = df_impl.sort_values(by='Size')
        
        # Plot peak GFLOPS vs. matrix size
        plt.plot(df_impl['Size'], df_impl['PeakGFLOPS'], marker='o', label=impl)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Peak GFLOPS (higher is better)')
    plt.title('Performance Scaling with Matrix Size')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/scaling_comparison.png')
    plt.close()
    
    print(f"Plots have been saved to the 'plots' directory")

def plot_memory_transfer_analysis(csv_file):
    """
    Plot memory transfer analysis for GPU implementations.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a directory for the plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot memory transfer overhead
    plt.figure(figsize=(10, 6))
    
    # Sort by matrix size
    df = df.sort_values(by='Size')
    
    # Plot transfer overhead vs. matrix size
    plt.plot(df['Size'], df['TransferOverhead'], marker='o', label='Transfer Overhead (%)')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Transfer Overhead (%)')
    plt.title('GPU Memory Transfer Overhead')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/memory_transfer_overhead.png')
    plt.close()
    
    # Plot time breakdown
    plt.figure(figsize=(12, 6))
    
    # Create stacked bar plot for time breakdown
    ind = np.arange(len(df['Size']))
    width = 0.7
    
    p1 = plt.bar(ind, df['HostToDeviceTime'], width, label='Host to Device Transfer')
    p2 = plt.bar(ind, df['ComputeTime'], width, bottom=df['HostToDeviceTime'], label='Computation')
    p3 = plt.bar(ind, df['DeviceToHostTime'], width, 
                 bottom=df['HostToDeviceTime'] + df['ComputeTime'], label='Device to Host Transfer')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.title('GPU Execution Time Breakdown')
    plt.xticks(ind, df['Size'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/gpu_time_breakdown.png')
    plt.close()
    
    print(f"Memory transfer analysis plots have been saved to the 'plots' directory")

def plot_library_comparison(csv_file):
    """
    Plot comparison between different library implementations.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a directory for the plots
    os.makedirs('plots', exist_ok=True)
    
    # Get unique matrix sizes
    sizes = df['Size'].unique()
    
    # Plot GFLOPS for each matrix size
    for size in sizes:
        # Filter data for the current size
        df_size = df[df['Size'] == size]
        
        # Sort by peak GFLOPS
        df_size = df_size.sort_values(by='PeakGFLOPS', ascending=False)
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        
        # Create bar plot for peak GFLOPS
        libraries = df_size['Library']
        peak_gflops = df_size['PeakGFLOPS']
        
        bars = plt.bar(libraries, peak_gflops)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Library')
        plt.ylabel('Peak GFLOPS (higher is better)')
        plt.title(f'Matrix Multiplication Library Comparison ({size}x{size})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'plots/library_comparison_{size}.png')
        plt.close()
    
    # Plot scaling across matrix sizes for each library
    libraries = df['Library'].unique()
    
    plt.figure(figsize=(10, 6))
    
    for lib in libraries:
        # Filter data for the current library
        df_lib = df[df['Library'] == lib]
        
        # Sort by matrix size
        df_lib = df_lib.sort_values(by='Size')
        
        # Plot peak GFLOPS vs. matrix size
        plt.plot(df_lib['Size'], df_lib['PeakGFLOPS'], marker='o', label=lib)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Peak GFLOPS (higher is better)')
    plt.title('Library Performance Scaling with Matrix Size')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/library_scaling_comparison.png')
    plt.close()
    
    print(f"Library comparison plots have been saved to the 'plots' directory")

def main():
    """
    Main function to parse command line arguments and call appropriate plotting function.
    """
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <csv_file> [plot_type]")
        print("  plot_type: 'performance', 'memory', or 'library' (default: performance)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    plot_type = sys.argv[2] if len(sys.argv) > 2 else 'performance'
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        sys.exit(1)
    
    if plot_type == 'performance':
        plot_performance_comparison(csv_file)
    elif plot_type == 'memory':
        plot_memory_transfer_analysis(csv_file)
    elif plot_type == 'library':
        plot_library_comparison(csv_file)
    else:
        print(f"Error: Unknown plot type '{plot_type}'")
        print("Supported types: 'performance', 'memory', or 'library'")
        sys.exit(1)

if __name__ == "__main__":
    main()
