#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_benchmark_data(file_path):
    """Read benchmark data from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def plot_execution_time(df, output_dir, log_scale=True):
    """Plot execution time comparison."""
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['Matrix Size'], df['Our Best (ms)'], 'o-', color='red', linewidth=2, markersize=8, label='Our Algorithm')
    plt.plot(df['Matrix Size'], df['OpenBLAS (ms)'], 's-', color='purple', linewidth=2, markersize=8, label='OpenBLAS')
    plt.plot(df['Matrix Size'], df['Intel MKL (ms)'], '^-', color='blue', linewidth=2, markersize=8, label='Intel MKL')
    plt.plot(df['Matrix Size'], df['cuBLAS (ms)'], 'D-', color='green', linewidth=2, markersize=8, label='cuBLAS')
    
    plt.title('Matrix Multiplication Execution Time Comparison', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    
    if log_scale:
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.7)
        output_file = os.path.join(output_dir, 'execution_time_log_scale.png')
    else:
        plt.grid(True, ls="--", alpha=0.7)
        output_file = os.path.join(output_dir, 'execution_time.png')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved execution time plot to {output_file}")
    plt.close()

def plot_gflops(df, output_dir):
    """Plot GFLOPS performance comparison."""
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['Matrix Size'], df['Our Best (GFLOPS)'], 'o-', color='red', linewidth=2, markersize=8, label='Our Algorithm')
    plt.plot(df['Matrix Size'], df['OpenBLAS (GFLOPS)'], 's-', color='purple', linewidth=2, markersize=8, label='OpenBLAS')
    plt.plot(df['Matrix Size'], df['Intel MKL (GFLOPS)'], '^-', color='blue', linewidth=2, markersize=8, label='Intel MKL')
    plt.plot(df['Matrix Size'], df['cuBLAS (GFLOPS)'], 'D-', color='green', linewidth=2, markersize=8, label='cuBLAS')
    
    plt.title('Matrix Multiplication Performance Comparison', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Performance (GFLOPS)', fontsize=14)
    plt.grid(True, ls="--", alpha=0.7)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'gflops_performance.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved GFLOPS performance plot to {output_file}")
    plt.close()

def plot_percentage_of_libraries(df, output_dir):
    """Plot percentage performance relative to libraries."""
    plt.figure(figsize=(12, 8))
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='100% (Equal Performance)')
    
    plt.plot(df['Matrix Size'], df['Percentage of OpenBLAS'], 's-', color='purple', linewidth=2, markersize=8, label='% of OpenBLAS')
    plt.plot(df['Matrix Size'], df['Percentage of Intel MKL'], '^-', color='blue', linewidth=2, markersize=8, label='% of Intel MKL')
    plt.plot(df['Matrix Size'], df['Percentage of cuBLAS'], 'D-', color='green', linewidth=2, markersize=8, label='% of cuBLAS')
    
    plt.title('Our Algorithm Performance Relative to Libraries', fontsize=16)
    plt.xlabel('Matrix Size (N×N)', fontsize=14)
    plt.ylabel('Percentage of Library Performance (%)', fontsize=14)
    plt.grid(True, ls="--", alpha=0.7)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'percentage_of_libraries.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved percentage performance plot to {output_file}")
    plt.close()

def create_bar_chart(df, output_dir):
    """Create bar chart comparing performance for specific matrix sizes."""
    # Select a subset of matrix sizes for clarity
    if len(df) > 4:
        # Choose a representative subset (small, medium, large)
        indices = [0, len(df)//3, 2*len(df)//3, len(df)-1]
        selected_df = df.iloc[indices]
    else:
        selected_df = df
    
    matrix_sizes = selected_df['Matrix Size'].astype(str).tolist()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Set width of bars
    bar_width = 0.2
    index = np.arange(len(matrix_sizes))
    
    # Time comparison (lower is better)
    ax1.bar(index - 1.5*bar_width, selected_df['Our Best (ms)'], bar_width, color='red', label='Our Algorithm')
    ax1.bar(index - 0.5*bar_width, selected_df['OpenBLAS (ms)'], bar_width, color='purple', label='OpenBLAS')
    ax1.bar(index + 0.5*bar_width, selected_df['Intel MKL (ms)'], bar_width, color='blue', label='Intel MKL')
    ax1.bar(index + 1.5*bar_width, selected_df['cuBLAS (ms)'], bar_width, color='green', label='cuBLAS')
    
    ax1.set_xlabel('Matrix Size (N×N)', fontsize=14)
    ax1.set_ylabel('Execution Time (ms)', fontsize=14)
    ax1.set_title('Execution Time Comparison (lower is better)', fontsize=16)
    ax1.set_xticks(index)
    ax1.set_xticklabels(matrix_sizes)
    ax1.legend()
    
    # GFLOPS comparison (higher is better)
    ax2.bar(index - 1.5*bar_width, selected_df['Our Best (GFLOPS)'], bar_width, color='red', label='Our Algorithm')
    ax2.bar(index - 0.5*bar_width, selected_df['OpenBLAS (GFLOPS)'], bar_width, color='purple', label='OpenBLAS')
    ax2.bar(index + 0.5*bar_width, selected_df['Intel MKL (GFLOPS)'], bar_width, color='blue', label='Intel MKL')
    ax2.bar(index + 1.5*bar_width, selected_df['cuBLAS (GFLOPS)'], bar_width, color='green', label='cuBLAS')
    
    ax2.set_xlabel('Matrix Size (N×N)', fontsize=14)
    ax2.set_ylabel('Performance (GFLOPS)', fontsize=14)
    ax2.set_title('Performance Comparison (higher is better)', fontsize=16)
    ax2.set_xticks(index)
    ax2.set_xticklabels(matrix_sizes)
    ax2.legend()
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'performance_bar_chart.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved bar chart comparison to {output_file}")
    plt.close()

def create_radar_chart(df, output_dir):
    """Create radar chart comparing normalized performance metrics."""
    # Choose the largest matrix size for the radar chart
    largest_size_row = df.iloc[-1]
    
    # Categories for radar chart
    categories = ['Execution\nSpeed', 'GFLOPS', 'OpenBLAS\nComparison', 'MKL\nComparison', 'cuBLAS\nComparison']
    
    # Values need to be normalized between 0 and 1, where 1 is best performance
    # For execution time, lower is better so we invert the normalization
    best_time = min(largest_size_row['Our Best (ms)'], largest_size_row['OpenBLAS (ms)'], 
                    largest_size_row['Intel MKL (ms)'], largest_size_row['cuBLAS (ms)'])
    
    # For GFLOPS, higher is better
    max_gflops = max(largest_size_row['Our Best (GFLOPS)'], largest_size_row['OpenBLAS (GFLOPS)'], 
                     largest_size_row['Intel MKL (GFLOPS)'], largest_size_row['cuBLAS (GFLOPS)'])
    
    # Normalized values
    our_values = [
        best_time / largest_size_row['Our Best (ms)'],
        largest_size_row['Our Best (GFLOPS)'] / max_gflops,
        largest_size_row['Percentage of OpenBLAS'] / 100,
        largest_size_row['Percentage of Intel MKL'] / 100,
        largest_size_row['Percentage of cuBLAS'] / 100
    ]
    
    openblas_values = [
        best_time / largest_size_row['OpenBLAS (ms)'],
        largest_size_row['OpenBLAS (GFLOPS)'] / max_gflops,
        1.0,  # 100% of itself
        largest_size_row['OpenBLAS (GFLOPS)'] / largest_size_row['Intel MKL (GFLOPS)'] if largest_size_row['Intel MKL (GFLOPS)'] > 0 else 0,
        largest_size_row['OpenBLAS (GFLOPS)'] / largest_size_row['cuBLAS (GFLOPS)'] if largest_size_row['cuBLAS (GFLOPS)'] > 0 else 0
    ]
    
    mkl_values = [
        best_time / largest_size_row['Intel MKL (ms)'],
        largest_size_row['Intel MKL (GFLOPS)'] / max_gflops,
        largest_size_row['Intel MKL (GFLOPS)'] / largest_size_row['OpenBLAS (GFLOPS)'] if largest_size_row['OpenBLAS (GFLOPS)'] > 0 else 0,
        1.0,  # 100% of itself
        largest_size_row['Intel MKL (GFLOPS)'] / largest_size_row['cuBLAS (GFLOPS)'] if largest_size_row['cuBLAS (GFLOPS)'] > 0 else 0
    ]
    
    cublas_values = [
        best_time / largest_size_row['cuBLAS (ms)'],
        largest_size_row['cuBLAS (GFLOPS)'] / max_gflops,
        largest_size_row['cuBLAS (GFLOPS)'] / largest_size_row['OpenBLAS (GFLOPS)'] if largest_size_row['OpenBLAS (GFLOPS)'] > 0 else 0,
        largest_size_row['cuBLAS (GFLOPS)'] / largest_size_row['Intel MKL (GFLOPS)'] if largest_size_row['Intel MKL (GFLOPS)'] > 0 else 0,
        1.0  # 100% of itself
    ]
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Extend all values to close the loop
    our_values += our_values[:1]
    openblas_values += openblas_values[:1]
    mkl_values += mkl_values[:1]
    cublas_values += cublas_values[:1]
    
    # Initialize the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Draw the lines
    ax.plot(angles, our_values, 'o-', linewidth=2, color='red', label='Our Algorithm')
    ax.plot(angles, openblas_values, 's-', linewidth=2, color='purple', label='OpenBLAS')
    ax.plot(angles, mkl_values, '^-', linewidth=2, color='blue', label='Intel MKL')
    ax.plot(angles, cublas_values, 'D-', linewidth=2, color='green', label='cuBLAS')
    
    # Fill areas
    ax.fill(angles, our_values, color='red', alpha=0.1)
    ax.fill(angles, openblas_values, color='purple', alpha=0.1)
    ax.fill(angles, mkl_values, color='blue', alpha=0.1)
    ax.fill(angles, cublas_values, color='green', alpha=0.1)
    
    # Add title and legend
    plt.title(f'Performance Comparison for {largest_size_row["Matrix Size"]}×{largest_size_row["Matrix Size"]} Matrix', fontsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Adjust the layout and save
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'radar_chart_comparison.png')
    plt.savefig(output_file, dpi=300)
    print(f"Saved radar chart to {output_file}")
    plt.close()

def generate_all_visualizations(input_file, output_dir):
    """Generate all visualizations for the benchmark data."""
    try:
        df = read_benchmark_data(input_file)
        
        # Create output directory if it doesn't exist
        create_directory_if_not_exists(output_dir)
        
        # Generate all plots
        plot_execution_time(df, output_dir, log_scale=True)
        plot_execution_time(df, output_dir, log_scale=False)
        plot_gflops(df, output_dir)
        plot_percentage_of_libraries(df, output_dir)
        create_bar_chart(df, output_dir)
        
        if len(df) > 0:  # Only create radar chart if we have data
            create_radar_chart(df, output_dir)
        
        print(f"All visualizations have been saved to {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False
    
    return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate visualizations from benchmark results.')
    parser.add_argument('--input', default='library_comparison_results.csv', 
                        help='Input CSV file with benchmark results')
    parser.add_argument('--output-dir', default='../showcase/visualizations',
                        help='Directory where visualizations will be saved')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.input):
        input_file = os.path.join(script_dir, '..', 'data', 'results', args.input)
    else:
        input_file = args.input
    
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    # Generate visualizations
    result = generate_all_visualizations(input_file, output_dir)
    
    return 0 if result else 1

if __name__ == "__main__":
    main()
