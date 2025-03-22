"""
Parallelism Optimization Framework for Deep Learning.

This framework automatically analyzes and distributes deep learning workloads 
across heterogeneous computing resources, optimizing both performance and energy efficiency.
"""

import torch
import torch.nn as nn
import time
import os
from typing import Any, Dict, List, Optional, Tuple

from ..graph_analysis.computational_graph import ComputationalGraph
from ..hardware_profiling.device_catalog import DeviceCatalog
from ..hardware_profiling.hardware_profiler import HardwareProfiler
from ..distribution_strategies.strategy_generator import StrategyGenerator
from ..runtime.execution_engine import ExecutionEngine
from ..runtime.performance_monitor import PerformanceMonitor
from ..utils.energy_monitor import EnergyMonitor
from ..visualization.performance_visualizer import PerformanceVisualizer

__version__ = "1.0.0"

class OptimizedParallel:
    """
    Main wrapper for the optimization framework.
    Makes it easy to integrate with existing PyTorch models.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        energy_aware: bool = False,
        communication_aware: bool = True,
        enable_monitoring: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the optimization framework for a PyTorch model.
        
        Args:
            model: PyTorch model to optimize
            energy_aware: Whether to optimize for energy efficiency
            communication_aware: Whether to optimize communication patterns
            enable_monitoring: Whether to collect performance metrics
            cache_dir: Directory to cache profiling data (None for no caching)
        """
        self.model = model
        self.energy_aware = energy_aware
        self.communication_aware = communication_aware
        self.enable_monitoring = enable_monitoring
        self.cache_dir = cache_dir
        
        # Create cache directories if needed
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.device_cache = os.path.join(cache_dir, "device_info.json")
            self.profile_cache = os.path.join(cache_dir, "hardware_profiles.json")
            self.metrics_file = os.path.join(cache_dir, "performance_metrics.json")
            self.energy_log = os.path.join(cache_dir, "energy_measurements.json")
        else:
            self.device_cache = None
            self.profile_cache = None
            self.metrics_file = None
            self.energy_log = None
        
        # Initialize components
        self.device_catalog = DeviceCatalog(cache_file=self.device_cache)
        self.energy_monitor = EnergyMonitor(log_file=self.energy_log) if energy_aware else None
        self.hardware_profiler = HardwareProfiler(
            self.device_catalog, cache_file=self.profile_cache, energy_monitor=self.energy_monitor
        )
        self.performance_monitor = PerformanceMonitor(log_file=self.metrics_file) if enable_monitoring else None
        
        # Will be initialized when first called
        self.graph = None
        self.strategy = None
        self.execution_engine = None
        
        print(f"Device catalog: Found {self.device_catalog.get_device_count('cpu')} CPUs and "
              f"{self.device_catalog.get_device_count('gpu')} GPUs")
    
    def __call__(self, *args, **kwargs):
        """
        Execute the model with automatic distribution.
        
        Args:
            *args, **kwargs: Arguments to pass to the model
            
        Returns:
            Model output
        """
        # Initialize on first call
        if self.execution_engine is None:
            self._initialize_with_inputs(*args, **kwargs)
        
        # Execute using the engine
        return self.execution_engine.forward(*args, **kwargs)
    
    def _initialize_with_inputs(self, *args, **kwargs):
        """Initialize the framework using example inputs"""
        print("First call, initializing optimization framework...")
        start_time = time.time()
        
        # Extract computational graph
        print("Analyzing computational graph...")
        self.graph = ComputationalGraph(self.model, args)
        
        # Generate optimal distribution strategy
        print("Generating optimal distribution strategy...")
        self.strategy_generator = StrategyGenerator(
            self.graph,
            self.device_catalog,
            self.hardware_profiler,
            energy_aware=self.energy_aware,
            communication_aware=self.communication_aware
        )
        self.strategy = self.strategy_generator.generate_strategy()
        
        # Create execution engine
        print("Setting up execution engine...")
        self.execution_engine = ExecutionEngine(
            self.model,
            self.strategy,
            performance_monitor=self.performance_monitor,
            enable_dynamic_adjustment=True
        )
        
        print(f"Initialization complete in {time.time() - start_time:.2f} seconds")
        self._print_strategy_summary()
    
    def _print_strategy_summary(self):
        """Print a summary of the distribution strategy"""
        if not self.strategy:
            return
            
        print("\n----- Distribution Strategy Summary -----")
        print(f"Total devices: {len(self.strategy.device_assignments)}")
        
        for i, assignment in enumerate(self.strategy.device_assignments):
            device_id = assignment.device_id
            print(f"Device {device_id}: {len(assignment.operation_ids)} operations")
            print(f"  Estimated compute time: {assignment.estimated_compute_time:.6f} seconds")
            print(f"  Estimated memory usage: {self.format_bytes(assignment.estimated_memory_usage)}")
            if self.energy_aware:
                print(f"  Estimated energy consumption: {assignment.energy_consumption:.3f} joules")
        
        if len(self.strategy.device_assignments) > 1:
            print(f"Communication cost: {self.strategy.communication_cost:.6f} seconds")
        print(f"Total estimated time: {self.strategy.estimated_total_time:.6f} seconds")
        if self.energy_aware:
            print(f"Total estimated energy: {self.strategy.estimated_energy:.3f} joules")
    
    def format_bytes(self, size_bytes):
        """Format bytes as human-readable string"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.2f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.2f} GB"
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        if not self.performance_monitor:
            return None
        return self.performance_monitor.get_current_metrics()
    
    def get_historical_metrics(self):
        """Get historical performance metrics"""
        if not self.performance_monitor:
            return None
        return self.performance_monitor.get_historical_metrics()
    
    def visualize_performance(self, output_dir=None, interactive=True):
        """
        Visualize performance metrics.
        
        Args:
            output_dir: Directory to save visualization files
            interactive: Whether to use interactive Plotly visualizations
        """
        if not self.performance_monitor:
            print("Performance monitoring not enabled. No metrics to visualize.")
            return
            
        visualizer = PerformanceVisualizer(output_dir=output_dir)
        
        # Get metrics
        current_metrics = self.get_performance_metrics()
        historical_metrics = self.get_historical_metrics()
        
        # Create visualizations
        visualizer.plot_device_utilization(current_metrics, use_plotly=interactive)
        visualizer.plot_operation_distribution(current_metrics, use_plotly=interactive)
        visualizer.plot_resource_utilization_timeline(historical_metrics, use_plotly=interactive)
        visualizer.plot_execution_time_trends(historical_metrics, use_plotly=interactive)
        
        print(f"Visualizations {'saved to ' + output_dir if output_dir else 'displayed'}.")
