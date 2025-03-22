#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def load_data(csv_file):
    """Load benchmark data from a CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    
    return pd.read_csv(csv_file)

def plot_gflops_comparison(df, output_dir):
    """Plot GFLOPS comparison for all implementations."""
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = df['Matrix Size'].values
    
    # Plot GFLOPS for each implementation
    if 'Our Best (GFLOPS)' in df.columns:
        plt.plot(matrix_sizes, df['Our Best (GFLOPS)'], 'o-', color='red', linewidth=2, 
                 label='Our Implementation')
    
    if 'OpenBLAS (GFLOPS)' in df.columns:
        plt.plot(matrix_sizes, df['OpenBLAS (GFLOPS)'], 's-', color='blue', linewidth=2, 
                 label='OpenBLAS')
    
    if 'Intel MKL (GFLOPS)' in df.columns:
        plt.plot(matrix_sizes, df['Intel MKL (GFLOPS)'], '^-', color='green', linewidth=2, 
                 label='Intel MKL')
    
    if 'cuBLAS (GFLOPS)' in df.columns:
        plt.plot(matrix_sizes, df['cuBLAS (GFLOPS)'], 'D-', color='purple', linewidth=2, 
                 label='cuBLAS')
    
    plt.title('Matrix Multiplication Performance Comparison', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Performance (GFLOPS)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Set x-axis to show actual matrix sizes
    plt.xticks(matrix_sizes, [str(size) for size in matrix_sizes])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gflops_comparison.png'), dpi=300)
    plt.close()

def plot_execution_time(df, output_dir):
    """Plot execution time comparison."""
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = df['Matrix Size'].values
    
    # Plot time for each implementation (in milliseconds)
    if 'Our Best (ms)' in df.columns:
        plt.plot(matrix_sizes, df['Our Best (ms)'], 'o-', color='red', linewidth=2, 
                 label='Our Implementation')
    
    if 'OpenBLAS (ms)' in df.columns:
        plt.plot(matrix_sizes, df['OpenBLAS (ms)'], 's-', color='blue', linewidth=2, 
                 label='OpenBLAS')
    
    if 'Intel MKL (ms)' in df.columns:
        plt.plot(matrix_sizes, df['Intel MKL (ms)'], '^-', color='green', linewidth=2, 
                 label='Intel MKL')
    
    if 'cuBLAS (ms)' in df.columns:
        plt.plot(matrix_sizes, df['cuBLAS (ms)'], 'D-', color='purple', linewidth=2, 
                 label='cuBLAS')
    
    plt.title('Matrix Multiplication Execution Time Comparison', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Set x-axis to show actual matrix sizes
    plt.xticks(matrix_sizes, [str(size) for size in matrix_sizes])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time.png'), dpi=300)
    plt.close()

def plot_percentage_comparison(df, output_dir):
    """Plot percentage of our implementation relative to libraries."""
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = df['Matrix Size'].values
    
    # Plot percentage for each library comparison
    if 'Percentage of OpenBLAS' in df.columns:
        plt.plot(matrix_sizes, df['Percentage of OpenBLAS'], 's-', color='blue', linewidth=2, 
                 label='vs OpenBLAS')
    
    if 'Percentage of Intel MKL' in df.columns:
        plt.plot(matrix_sizes, df['Percentage of Intel MKL'], '^-', color='green', linewidth=2, 
                 label='vs Intel MKL')
    
    if 'Percentage of cuBLAS' in df.columns:
        plt.plot(matrix_sizes, df['Percentage of cuBLAS'], 'D-', color='purple', linewidth=2, 
                 label='vs cuBLAS')
    
    # Add a reference line at 100%
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='100% (Equal Performance)')
    
    plt.title('Our Implementation Performance Relative to Libraries', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.xscale('log', base=2)
    
    # Set x-axis to show actual matrix sizes
    plt.xticks(matrix_sizes, [str(size) for size in matrix_sizes])
    
    # Set y-axis limits to show percentages clearly
    plt.ylim(0, max(100, df[['Percentage of OpenBLAS', 'Percentage of Intel MKL', 
                            'Percentage of cuBLAS']].max().max() * 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percentage_comparison.png'), dpi=300)
    plt.close()

def create_bar_chart(df, output_dir):
    """Create bar chart for selected matrix sizes."""
    # For clarity, use a subset of matrix sizes if there are many
    if len(df) > 4:
        indices = [0, len(df)//3, 2*len(df)//3, -1]  # First, 1/3, 2/3, and last
        selected_df = df.iloc[indices].copy()
    else:
        selected_df = df.copy()
    
    plt.figure(figsize=(14, 8))
    
    # Set up the positions for bars
    matrix_sizes = selected_df['Matrix Size'].values
    x = np.arange(len(matrix_sizes))
    width = 0.2  # Width of bars
    
    # Create bars for each implementation
    if 'Our Best (GFLOPS)' in selected_df.columns:
        plt.bar(x - 1.5*width, selected_df['Our Best (GFLOPS)'], width, 
                label='Our Implementation', color='red')
    
    if 'OpenBLAS (GFLOPS)' in selected_df.columns:
        plt.bar(x - 0.5*width, selected_df['OpenBLAS (GFLOPS)'], width, 
                label='OpenBLAS', color='blue')
    
    if 'Intel MKL (GFLOPS)' in selected_df.columns:
        plt.bar(x + 0.5*width, selected_df['Intel MKL (GFLOPS)'], width, 
                label='Intel MKL', color='green')
    
    if 'cuBLAS (GFLOPS)' in selected_df.columns:
        plt.bar(x + 1.5*width, selected_df['cuBLAS (GFLOPS)'], width, 
                label='cuBLAS', color='purple')
    
    plt.title('Matrix Multiplication Performance by Implementation', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Performance (GFLOPS)', fontsize=14)
    plt.xticks(x, [str(size) for size in matrix_sizes])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_bar_chart.png'), dpi=300)
    plt.close()

def plot_cpu_scaling(cpu_df, output_dir):
    """Plot performance scaling for CPU implementations."""
    if cpu_df is None or cpu_df.empty:
        return
    
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = cpu_df['Matrix Size'].values
    
    # Plot GFLOPS for each CPU implementation
    if 'Naive (GFLOPS)' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Naive (GFLOPS)'], 'o-', color='gray', linewidth=2, 
                 label='Naive')
    
    if 'Blocked (GFLOPS)' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Blocked (GFLOPS)'], 's-', color='blue', linewidth=2, 
                 label='Blocked')
    
    if 'SIMD (GFLOPS)' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['SIMD (GFLOPS)'], '^-', color='green', linewidth=2, 
                 label='SIMD')
    
    if 'Threaded (GFLOPS)' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Threaded (GFLOPS)'], 'D-', color='orange', linewidth=2, 
                 label='Threaded')
    
    if 'Combined (GFLOPS)' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Combined (GFLOPS)'], '*-', color='red', linewidth=2, 
                 label='Combined (SIMD + Threaded)')
    
    plt.title('CPU Implementation Performance Scaling', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Performance (GFLOPS)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Set x-axis to show actual matrix sizes
    plt.xticks(matrix_sizes, [str(size) for size in matrix_sizes])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_scaling.png'), dpi=300)
    plt.close()

def plot_speedup_vs_naive(cpu_df, output_dir):
    """Plot speedup relative to naive implementation."""
    if cpu_df is None or cpu_df.empty:
        return
    
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = cpu_df['Matrix Size'].values
    
    # Plot speedup for each CPU implementation
    if 'Speedup Blocked' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Speedup Blocked'], 's-', color='blue', linewidth=2, 
                 label='Blocked vs Naive')
    
    if 'Speedup SIMD' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Speedup SIMD'], '^-', color='green', linewidth=2, 
                 label='SIMD vs Naive')
    
    if 'Speedup Threaded' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Speedup Threaded'], 'D-', color='orange', linewidth=2, 
                 label='Threaded vs Naive')
    
    if 'Speedup Combined' in cpu_df.columns:
        plt.plot(matrix_sizes, cpu_df['Speedup Combined'], '*-', color='red', linewidth=2, 
                 label='Combined vs Naive')
    
    plt.title('Speedup Relative to Naive Implementation', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Speedup (X times faster)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.xscale('log', base=2)
    
    # Set x-axis to show actual matrix sizes
    plt.xticks(matrix_sizes, [str(size) for size in matrix_sizes])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_vs_naive.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--lib-results', type=str, default='library_comparison_results.csv',
                      help='Path to library comparison results CSV file')
    parser.add_argument('--cpu-results', type=str, default='cpu_benchmark_results.csv',
                      help='Path to CPU benchmark results CSV file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Process library comparison results
    try:
        lib_df = load_data(args.lib_results)
        plot_gflops_comparison(lib_df, args.output_dir)
        plot_execution_time(lib_df, args.output_dir)
        plot_percentage_comparison(lib_df, args.output_dir)
        create_bar_chart(lib_df, args.output_dir)
        print(f"Library comparison visualizations saved to {args.output_dir}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    # Process CPU benchmark results if available
    try:
        cpu_df = load_data(args.cpu_results)
        plot_cpu_scaling(cpu_df, args.output_dir)
        plot_speedup_vs_naive(cpu_df, args.output_dir)
        print(f"CPU scaling visualizations saved to {args.output_dir}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()