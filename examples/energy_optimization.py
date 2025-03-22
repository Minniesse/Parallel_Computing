"""
Example demonstrating energy-aware optimization for deep learning models.
Compares standard PyTorch execution with our energy-optimized distribution.
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

from src.utils.energy_monitor import EnergyMonitor
from src.visualization.performance_visualizer import PerformanceVisualizer

# Import from basic_usage example
from basic_usage import OptimizedParallel

# Define a more complex model for better demonstrating energy optimization
class ComplexModel(nn.Module):
    """A larger model with more compute-intensive operations"""
    
    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def run_baseline(model, input_data, num_iterations=10):
    """Run standard PyTorch model and measure energy"""
    # Create energy monitor
    energy_monitor = EnergyMonitor(sampling_interval=0.1, log_file="./energy_baseline.json")
    
    # Move model to first available GPU (or CPU if no GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_data = input_data.to(device)
    
    # Warm-up run
    with torch.no_grad():
        _ = model(input_data)
    
    # Start energy monitoring
    energy_monitor.start_monitoring("baseline")
    
    # Run iterations
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)
    
    # Stop monitoring
    total_time = time.time() - start_time
    energy_monitor.stop_monitoring()
    
    # Get energy summary
    energy_summary = energy_monitor.get_last_session_summary()
    
    return {
        'execution_time': total_time / num_iterations,
        'total_time': total_time,
        'energy_summary': energy_summary
    }

def run_optimized(model, input_data, num_iterations=10):
    """Run energy-optimized model distribution"""
    # Create energy monitor
    energy_monitor = EnergyMonitor(sampling_interval=0.1, log_file="./energy_optimized.json")
    
    # Create optimized model
    optimized_model = OptimizedParallel(
        model,
        energy_aware=True,  # Enable energy-aware optimization
        cache_dir="./cache"
    )
    
    # Warm-up run (this will trigger analysis and optimization)
    _ = optimized_model(input_data)
    
    # Start energy monitoring
    energy_monitor.start_monitoring("optimized")
    
    # Run iterations
    start_time = time.time()
    for _ in range(num_iterations):
        _ = optimized_model(input_data)
    
    # Stop monitoring
    total_time = time.time() - start_time
    energy_monitor.stop_monitoring()
    
    # Get energy summary
    energy_summary = energy_monitor.get_last_session_summary()
    
    # Also get performance metrics from our framework
    performance_metrics = optimized_model.get_performance_metrics()
    
    return {
        'execution_time': total_time / num_iterations,
        'total_time': total_time,
        'energy_summary': energy_summary,
        'performance_metrics': performance_metrics
    }

def visualize_comparison(baseline_results, optimized_results):
    """Visualize comparison between baseline and optimized results"""
    # Create output directory
    os.makedirs("./energy_comparison", exist_ok=True)
    
    # Extract data
    baseline_time = baseline_results['execution_time'] * 1000  # Convert to ms
    optimized_time = optimized_results['execution_time'] * 1000  # Convert to ms
    
    baseline_energy = baseline_results['energy_summary']['total_energy_joules']
    optimized_energy = optimized_results['energy_summary']['total_energy_joules']
    
    # Calculate improvements
    time_improvement = (baseline_time - optimized_time) / baseline_time * 100
    energy_improvement = (baseline_energy - optimized_energy) / baseline_energy * 100
    
    # Create comparison plots
    plt.figure(figsize=(12, 5))
    
    # Execution time comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(['Baseline', 'Optimized'], [baseline_time, optimized_time])
    bars[0].set_color('lightcoral')
    bars[1].set_color('seagreen')
    plt.title('Execution Time Comparison')
    plt.ylabel('Time per iteration (ms)')
    plt.grid(axis='y', alpha=0.3)
    # Add time improvement as text
    plt.text(1, optimized_time + 5, f"{time_improvement:.1f}% faster", 
             ha='center', va='bottom', fontweight='bold')
    
    # Energy comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(['Baseline', 'Optimized'], [baseline_energy, optimized_energy])
    bars[0].set_color('lightcoral')
    bars[1].set_color('seagreen')
    plt.title('Energy Consumption Comparison')
    plt.ylabel('Energy (joules)')
    plt.grid(axis='y', alpha=0.3)
    # Add energy improvement as text
    plt.text(1, optimized_energy + 5, f"{energy_improvement:.1f}% less energy", 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("./energy_comparison/performance_comparison.png")
    
    # Create more detailed energy visualization using our visualizer
    visualizer = PerformanceVisualizer(output_dir="./energy_comparison")
    
    # Get device-specific energy data
    baseline_device_energy = []
    optimized_device_energy = []
    device_labels = []
    
    for device, stats in baseline_results['energy_summary']['energy'].items():
        device_labels.append(device)
        baseline_device_energy.append(stats['energy_joules'])
        
        # Find matching device in optimized results
        if device in optimized_results['energy_summary']['energy']:
            optimized_device_energy.append(optimized_results['energy_summary']['energy'][device]['energy_joules'])
        else:
            optimized_device_energy.append(0)
    
    # Plot device-specific energy comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(device_labels))
    width = 0.35
    
    plt.bar(x - width/2, baseline_device_energy, width, label='Baseline', color='lightcoral')
    plt.bar(x + width/2, optimized_device_energy, width, label='Optimized', color='seagreen')
    
    plt.xlabel('Device')
    plt.ylabel('Energy (joules)')
    plt.title('Energy Consumption by Device')
    plt.xticks(x, device_labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./energy_comparison/device_energy_comparison.png")
    
    print("Visualizations saved to ./energy_comparison/")

def main():
    # Create model
    model = ComplexModel()
    
    # Create input data
    batch_size = 16
    input_data = torch.randn(batch_size, 3, 32, 32)
    
    # Number of iterations for each run
    num_iterations = 20
    
    print("Running baseline model...")
    baseline_results = run_baseline(model, input_data, num_iterations)
    
    print("\nRunning energy-optimized model...")
    optimized_results = run_optimized(model, input_data, num_iterations)
    
    # Print results
    print("\n=== Results ===")
    print(f"Baseline avg. time: {baseline_results['execution_time']*1000:.2f} ms")
    print(f"Optimized avg. time: {optimized_results['execution_time']*1000:.2f} ms")
    print(f"Baseline energy: {baseline_results['energy_summary']['total_energy_joules']:.2f} joules")
    print(f"Optimized energy: {optimized_results['energy_summary']['total_energy_joules']:.2f} joules")
    
    # Calculate improvements
    time_improvement = (baseline_results['execution_time'] - optimized_results['execution_time']) / baseline_results['execution_time'] * 100
    energy_improvement = (baseline_results['energy_summary']['total_energy_joules'] - optimized_results['energy_summary']['total_energy_joules']) / baseline_results['energy_summary']['total_energy_joules'] * 100
    
    print(f"\nTime improvement: {time_improvement:.1f}%")
    print(f"Energy improvement: {energy_improvement:.1f}%")
    
    # Create visualizations
    print("\nCreating comparison visualizations...")
    visualize_comparison(baseline_results, optimized_results)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
