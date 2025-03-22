"""
Example demonstrating optimization across heterogeneous hardware (CPU+GPU).
Compares different strategies for distributing workloads across heterogeneous hardware.
"""

import torch
import torch.nn as nn
import time
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_analysis.computational_graph import ComputationalGraph
from src.hardware_profiling.device_catalog import DeviceCatalog
from src.hardware_profiling.hardware_profiler import HardwareProfiler
from src.distribution_strategies.strategy_generator import StrategyGenerator
from src.runtime.execution_engine import ExecutionEngine
from src.runtime.performance_monitor import PerformanceMonitor
from src.visualization.performance_visualizer import PerformanceVisualizer

# Define a model with components well-suited for different hardware types
class HeterogeneousModel(nn.Module):
    """
    A model with components that have different optimal hardware:
    - Convolutional layers: typically best on GPU
    - Element-wise operations: can be efficient on CPU for smaller sizes
    - Memory-bound operations: may be better on CPU in some cases
    """
    
    def __init__(self):
        super().__init__()
        # CNN feature extraction (typically GPU-optimal)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Feature processing (mix of compute patterns)
        self.feature_processing = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Memory-bound, potentially CPU-friendly
        )
        
        # Classification head (small compute)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.feature_processing(x)
        x = self.classifier(x)
        return x

class Strategy:
    """Helper class to handle different optimization strategies"""
    
    def __init__(self, name, model, input_data, communication_aware=True, energy_aware=False):
        self.name = name
        self.model = model
        self.input_data = input_data
        self.communication_aware = communication_aware
        self.energy_aware = energy_aware
        
        # Will be initialized during execution
        self.graph = None
        self.strategy = None
        self.execution_engine = None
        self.performance_monitor = None
        self.metrics = {}
        
        # Set up
        self._initialize()
    
    def _initialize(self):
        """Initialize components for this strategy"""
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            log_file=f"./heterogeneous_results/{self.name}_metrics.json"
        )
        
        # Discover hardware
        device_catalog = DeviceCatalog(
            cache_file="./cache/device_info.json"
        )
        
        # Create energy monitor if needed
        energy_monitor = None
        if self.energy_aware:
            from src.utils.energy_monitor import EnergyMonitor
            energy_monitor = EnergyMonitor(
                log_file=f"./heterogeneous_results/{self.name}_energy.json"
            )
        
        # Hardware profiler
        hardware_profiler = HardwareProfiler(
            device_catalog,
            cache_file="./cache/hardware_profiles.json",
            energy_monitor=energy_monitor
        )
        
        # Extract computational graph
        print(f"[{self.name}] Analyzing computational graph...")
        self.graph = ComputationalGraph(self.model, (self.input_data,))
        
        # Generate distribution strategy
        print(f"[{self.name}] Generating distribution strategy...")
        strategy_generator = StrategyGenerator(
            self.graph,
            device_catalog,
            hardware_profiler,
            energy_aware=self.energy_aware,
            communication_aware=self.communication_aware
        )
        self.strategy = strategy_generator.generate_strategy()
        
        # Create execution engine
        print(f"[{self.name}] Setting up execution engine...")
        self.execution_engine = ExecutionEngine(
            self.model,
            self.strategy,
            performance_monitor=self.performance_monitor,
            enable_dynamic_adjustment=True
        )
    
    def run(self, num_iterations=10):
        """Execute the model with this strategy"""
        # Warm-up run
        _ = self.execution_engine.forward(self.input_data)
        
        # Actual timed runs
        start_time = time.time()
        for _ in range(num_iterations):
            outputs = self.execution_engine.forward(self.input_data)
            # Ensure all GPU operations are completed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Calculate timing
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        # Get performance metrics
        self.metrics = self.performance_monitor.get_current_metrics()
        historical_metrics = self.performance_monitor.get_historical_metrics()
        
        print(f"[{self.name}] Average execution time: {avg_time*1000:.2f} ms")
        
        return {
            'avg_time': avg_time,
            'total_time': total_time,
            'current_metrics': self.metrics,
            'historical_metrics': historical_metrics
        }
    
    def visualize(self, output_dir):
        """Create visualizations for this strategy"""
        if not self.metrics:
            print(f"[{self.name}] No metrics available for visualization")
            return
            
        # Create output directory
        strategy_dir = os.path.join(output_dir, self.name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Create visualizer
        visualizer = PerformanceVisualizer(output_dir=strategy_dir)
        
        # Get metrics
        current_metrics = self.metrics
        historical_metrics = self.performance_monitor.get_historical_metrics()
        
        # Create visualizations
        visualizer.plot_device_utilization(current_metrics, 
                                          title=f"{self.name} - Device Utilization")
        
        visualizer.plot_operation_distribution(current_metrics,
                                              title=f"{self.name} - Operation Distribution")
        
        visualizer.plot_resource_utilization_timeline(historical_metrics,
                                                     title=f"{self.name} - Resource Utilization")
        
        print(f"[{self.name}] Visualizations saved to {strategy_dir}")

def compare_strategies(strategies, results, output_dir):
    """Compare different strategies and visualize results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract execution times
    names = [s.name for s in strategies]
    times = [results[s.name]['avg_time'] * 1000 for s in strategies]  # Convert to ms
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, times)
    
    # Add values on top of bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_val:.2f} ms', ha='center', va='bottom')
    
    plt.title('Execution Time Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Time per iteration (ms)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "strategy_time_comparison.png"))
    
    # Create visualizer for detailed comparison
    visualizer = PerformanceVisualizer(output_dir=output_dir)
    
    # Prepare data for strategy comparison
    strategy_metrics = [results[s.name]['current_metrics'] for s in strategies]
    
    # Create comparison visualization
    visualizer.plot_strategy_comparison(
        strategies=[s.name for s in strategies],
        metrics=strategy_metrics,
        title="Strategy Performance Comparison"
    )
    
    print(f"Strategy comparisons saved to {output_dir}")

def main():
    # Create output directory
    os.makedirs("./heterogeneous_results", exist_ok=True)
    
    # Create model
    model = HeterogeneousModel()
    
    # Create input data
    batch_size = 32
    input_data = torch.randn(batch_size, 3, 32, 32)
    
    # Define different optimization strategies
    strategies = [
        Strategy("GPU_Only", model, input_data, communication_aware=False, energy_aware=False),
        Strategy("CPU_GPU_Split", model, input_data, communication_aware=True, energy_aware=False),
        Strategy("Energy_Optimized", model, input_data, communication_aware=True, energy_aware=True)
    ]
    
    # Execute each strategy
    results = {}
    for strategy in strategies:
        print(f"\nExecuting strategy: {strategy.name}")
        results[strategy.name] = strategy.run(num_iterations=20)
        
        # Create visualizations
        strategy.visualize("./heterogeneous_results")
    
    # Compare strategies
    print("\nComparing strategies...")
    compare_strategies(strategies, results, "./heterogeneous_results")
    
    print("\nAll strategies executed and compared. Results saved to ./heterogeneous_results")

if __name__ == "__main__":
    main()
