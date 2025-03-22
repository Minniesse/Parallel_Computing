import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_operation_comparison(df: pd.DataFrame, output_path: str = None):
    """
    Plot a comparison of different backends for operations.
    
    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Operation', y='Mean Time (s)', hue='Backend', data=df)
    
    # Add labels and title
    plt.xlabel('Operation')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.title('Performance Comparison of Different Backends')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    # Show the plot
    plt.show()

def plot_speedup(df: pd.DataFrame, baseline: str = 'sequential', output_path: str = None):
    """
    Plot speedup relative to a baseline backend.
    
    Args:
        df: DataFrame with benchmark results
        baseline: Baseline backend for speedup calculation
        output_path: Path to save the plot (optional)
    """
    # Group by Operation and get the baseline times
    baseline_times = df[df['Backend'] == baseline].set_index('Operation')['Mean Time (s)']
    
    # Calculate speedup
    df_speedup = df.copy()
    df_speedup['Speedup'] = df.apply(
        lambda row: baseline_times[row['Operation']] / row['Mean Time (s)'],
        axis=1
    )
    
    # Remove baseline entries
    df_speedup = df_speedup[df_speedup['Backend'] != baseline]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Operation', y='Speedup', hue='Backend', data=df_speedup)
    
    # Add labels and title
    plt.xlabel('Operation')
    plt.ylabel(f'Speedup (relative to {baseline})')
    plt.title(f'Speedup Comparison (Higher is Better)')
    
    # Add a horizontal line at y=1 (baseline performance)
    plt.axhline(y=1, color='r', linestyle='--')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fx')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    # Show the plot
    plt.show()

def plot_image_size_scaling(df: pd.DataFrame, output_path: str = None):
    """
    Plot execution time vs image size.
    
    Args:
        df: DataFrame with benchmark results
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with line
    plt.scatter(df['Pixels'], df['Mean Time (s)'], s=100)
    plt.plot(df['Pixels'], df['Mean Time (s)'], 'b-')
    
    # Add data labels
    for i, row in df.iterrows():
        plt.annotate(f"{row['Image Size']}\n{row['Mean Time (s)']:.3f}s", 
                    (row['Pixels'], row['Mean Time (s)']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Add labels and title
    plt.xlabel('Number of Pixels')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.title('Execution Time vs Image Size')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(output_path)
    
    # Show the plot
    plt.show()

def main():
    """Visualize benchmark results with command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize image processing benchmark results")
    parser.add_argument("input", type=str, help="CSV file with benchmark results")
    parser.add_argument("--plot-type", type=str, choices=["comparison", "speedup", "scaling"],
                        default="comparison", help="Type of plot to generate")
    parser.add_argument("--baseline", type=str, default="sequential", 
                        help="Baseline backend for speedup calculation")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output file for the plot (PNG, PDF, etc.)")
    
    args = parser.parse_args()
    
    # Load benchmark results
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    df = pd.read_csv(args.input)
    
    # Generate requested plot
    if args.plot_type == "comparison":
        if 'Operation' in df.columns and 'Backend' in df.columns:
            plot_operation_comparison(df, args.output)
        else:
            print("Error: Input data is not compatible with operation comparison plot.")
    
    elif args.plot_type == "speedup":
        if 'Operation' in df.columns and 'Backend' in df.columns:
            plot_speedup(df, args.baseline, args.output)
        else:
            print("Error: Input data is not compatible with speedup plot.")
    
    elif args.plot_type == "scaling":
        if 'Image Size' in df.columns and 'Pixels' in df.columns:
            plot_image_size_scaling(df, args.output)
        else:
            print("Error: Input data is not compatible with image size scaling plot.")

if __name__ == "__main__":
    main()
