"""
Example demonstrating the adaptive optimization framework that automatically
determines the best execution strategy based on model complexity.
"""

import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parallel_opt.model_optimizer import AdaptiveOptimizer
from example_model import ExampleModel, LargeModel, TransformerModel

def benchmark_model(model_class, batch_sizes=[1, 16, 32], iterations=10):
    """Benchmark model performance with and without adaptive optimization"""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n=== Testing with batch size {batch_size} ===")
        
        # Create model instance
        model = model_class()
        
        # Create input data
        input_data = torch.randn(batch_size, 3, 32, 32)
        
        # Baseline performance (direct execution)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_data = input_data.to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data)
        
        # Measure baseline performance
        baseline_times = []
        for _ in range(iterations):
            start = time.time()
            with torch.no_grad():
                _ = model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            baseline_times.append((time.time() - start) * 1000)  # ms
        
        # Create adaptive optimizer
        adaptive_model = AdaptiveOptimizer(
            model,
            energy_aware=True,
            cache_dir="./cache",
            small_model_overhead=0.3  # Allow up to 30% overhead
        )
        
        # Warm-up with adaptive optimizer
        _ = adaptive_model(input_data)
        for _ in range(4):
            _ = adaptive_model(input_data)
        
        # Measure adaptive performance
        adaptive_times = []
        for _ in range(iterations):
            start = time.time()
            _ = adaptive_model(input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            adaptive_times.append((time.time() - start) * 1000)  # ms
        
        # Store results
        results[batch_size] = {
            'baseline_mean': np.mean(baseline_times),
            'baseline_std': np.std(baseline_times),
            'adaptive_mean': np.mean(adaptive_times),
            'adaptive_std': np.std(adaptive_times),
            'speedup': np.mean(baseline_times) / np.mean(adaptive_times)
        }
        
        # Print results
        print(f"  Baseline: {results[batch_size]['baseline_mean']:.2f} ms ± {results[batch_size]['baseline_std']:.2f} ms")
        print(f"  Adaptive: {results[batch_size]['adaptive_mean']:.2f} ms ± {results[batch_size]['adaptive_std']:.2f} ms")
        print(f"  Speedup: {results[batch_size]['speedup']:.2f}x")
    
    return results

def main():
    """Run benchmarks on different models"""
    os.makedirs("./adaptive_results", exist_ok=True)
    
    models = {
        'ExampleModel': ExampleModel,
        'LargeModel': LargeModel
    }
    
    all_results = {}
    
    for model_name, model_class in models.items():
        print(f"\n\n========== Benchmarking {model_name} ==========")
        all_results[model_name] = benchmark_model(model_class)
    
    # Create summary visualization
    fig, axes = plt.subplots(1, len(models), figsize=(12, 6), sharey=True)
    
    batch_sizes = list(all_results[list(models.keys())[0]].keys())
    
    for i, (model_name, results) in enumerate(all_results.items()):
        ax = axes[i] if len(models) > 1 else axes
        
        # Extract data
        baselines = [results[bs]['baseline_mean'] for bs in batch_sizes]
        adaptives = [results[bs]['adaptive_mean'] for bs in batch_sizes]
        
        # Set up x-ticks
        x = np.arange(len(batch_sizes))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, baselines, width, label='Baseline', color='lightcoral')
        ax.bar(x + width/2, adaptives, width, label='Adaptive', color='seagreen')
        
        # Add labels and title
        ax.set_xlabel('Batch Size')
        if i == 0:
            ax.set_ylabel('Execution Time (ms)')
        ax.set_title(model_name)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.legend()
        
        # Add speedup annotations
        for j, bs in enumerate(batch_sizes):
            speedup = results[bs]['speedup']
            ax.annotate(f"{speedup:.2f}x", 
                       xy=(j, max(results[bs]['baseline_mean'], results[bs]['adaptive_mean']) + 0.1),
                       ha='center',
                       fontweight='bold',
                       color='green' if speedup > 1.0 else 'red')
    
    plt.tight_layout()
    plt.savefig("./adaptive_results/model_comparison.png")
    
    print("\nBenchmark complete. See ./adaptive_results/ for visualization.")

if __name__ == "__main__":
    main()
