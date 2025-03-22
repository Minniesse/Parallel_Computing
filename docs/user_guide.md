# User Guide: Parallelism Optimization Framework

This guide provides comprehensive instructions for using the Parallelism Optimization Framework to accelerate your deep learning workloads.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Configuration Options](#configuration-options)
4. [Advanced Features](#advanced-features)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10.0 or higher
- CUDA toolkit (for GPU support)

### Install from PyPI

The simplest way to install the framework is via pip:

```bash
pip install parallel-opt
```

### Install from Source

Alternatively, you can install from source for the latest features:

```bash
git clone https://github.com/yourusername/parallelism-optimization-framework.git
cd parallelism-optimization-framework
pip install -e .
```

### Verify Installation

To verify the installation:

```python
import parallel_opt
print(parallel_opt.__version__)
```

## Basic Usage

Using the framework requires minimal changes to your existing PyTorch code:

```python
import torch
from parallel_opt import OptimizedParallel

# Your existing PyTorch model
model = YourModel()

# Wrap with our framework for automatic optimization
optimized_model = OptimizedParallel(model)

# Use as you would use the original model
outputs = optimized_model(inputs)
```

### What Happens Behind the Scenes

1. On the first call, the framework analyzes your model's computational graph
2. It profiles available hardware characteristics
3. It generates an optimal distribution strategy
4. It executes your model using the optimized strategy
5. Subsequent calls use the same strategy with minimal overhead

## Configuration Options

The `OptimizedParallel` wrapper accepts several arguments to customize its behavior:

```python
optimized_model = OptimizedParallel(
    model,                  # Your PyTorch model
    energy_aware=True,      # Whether to optimize for energy efficiency
    communication_aware=True, # Whether to optimize communication patterns
    enable_monitoring=True, # Whether to collect performance metrics
    cache_dir="./cache"     # Directory to cache profiling data
)
```

### Energy-Aware Optimization

When `energy_aware` is True, the framework will:
- Monitor power consumption during execution
- Consider energy efficiency when generating distribution strategies
- Balance performance against power consumption

### Communication Optimization

When `communication_aware` is True, the framework will:
- Analyze data dependencies between operations
- Minimize data movement between devices
- Group operations to reduce communication overhead

### Performance Monitoring

When `enable_monitoring` is True, the framework will:
- Track execution time for operations
- Monitor device utilization
- Record memory usage patterns
- These metrics can be visualized or accessed programmatically

### Caching

Setting `cache_dir` allows the framework to:
- Cache hardware profiling data for faster initialization
- Store performance metrics between runs
- Retain optimized strategies for similar workloads

## Advanced Features

### Performance Visualization

The framework includes tools to visualize performance metrics:

```python
# After running your model
optimized_model.visualize_performance(
    output_dir="./visualizations",  # Where to save visualizations
    interactive=True                # Whether to use interactive Plotly charts
)
```

This generates several useful visualizations:
- Device utilization distribution
- Operation distribution across devices
- Resource utilization timeline
- Execution time trends

### Accessing Performance Metrics

You can access performance metrics programmatically:

```python
# Get current metrics
metrics = optimized_model.get_performance_metrics()

# Get historical metrics
historical = optimized_model.get_historical_metrics()

# Access specific values
avg_time = metrics['avg_execution_time']
device_utils = metrics['device_utilization']
```

### Custom Distribution Strategies

For advanced users, it's possible to implement custom distribution strategies:

```python
from parallel_opt.distribution_strategies import StrategyGenerator, DistributionStrategy

class MyCustomStrategy(StrategyGenerator):
    # Implement custom strategy logic
    # ...

# Use your custom strategy
strategy = MyCustomStrategy(graph, device_catalog, hardware_profiler).generate_strategy()
```

## Performance Tuning

### Optimizing for Different Hardware

The framework automatically adapts to your hardware, but you can influence this:

- For CPU-heavy workloads with limited GPU memory, enable `communication_aware`
- For multi-GPU systems, ensure CUDA and NCCL are properly configured
- For heterogeneous systems, the framework will automatically balance workloads

### Memory Optimization

To optimize memory usage:

```python
# Enable aggressive memory optimization
from parallel_opt.runtime import MemoryOptimizer

memory_optimizer = MemoryOptimizer(aggressive=True)
optimized_model = OptimizedParallel(model, memory_optimizer=memory_optimizer)
```

### Batch Size Considerations

The framework's optimization strategy depends on the batch size:

- Smaller batch sizes may benefit more from intra-operation parallelism
- Larger batch sizes often work better with data parallelism
- The framework automatically adapts to your batch size

## Troubleshooting

### Common Issues

**Issue: Error during model analysis**

This typically occurs with models containing dynamic control flow. Try:

```python
# Use dynamic-aware analysis
optimized_model = OptimizedParallel(model, dynamic_aware=True)
```

**Issue: Performance degradation**

If you observe slower performance:

1. Check hardware utilization with `optimized_model.visualize_performance()`
2. Ensure your model is complex enough to benefit from parallelization
3. Try disabling dynamic adjustments: `enable_dynamic_adjustment=False`

**Issue: Out of memory errors**

If you encounter OOM errors:

1. Reduce batch size
2. Enable memory optimization
3. Use the `memory_fraction` parameter to limit GPU memory usage:

```python
optimized_model = OptimizedParallel(model, memory_fraction=0.8)  # Use 80% of available memory
```

### Getting Help

For more assistance:

- Check the [examples](../examples/) directory for sample applications
- Review the [API documentation](api_reference.md) for detailed function descriptions
- Submit issues on our GitHub repository for specific problems
