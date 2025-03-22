import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import argparse
from PIL import Image

# Import pipeline and operations
from src.core.pipeline import Pipeline
from src.core.utils import load_image, save_image
from src.operations.filters import GaussianBlur, EdgeDetection

def run_operation_benchmark(image_path: str, operation: Callable, 
                           backends: List[str], iterations: int = 5) -> pd.DataFrame:
    """
    Benchmark a single operation with different backends.
    
    Args:
        image_path: Path to the input image
        operation: Operation factory function
        backends: List of backends to benchmark
        iterations: Number of iterations to run for each benchmark
        
    Returns:
        DataFrame with benchmark results
    """
    # Load image
    image = load_image(image_path)
    
    # Prepare results storage
    results = []
    
    # Run benchmarks for each backend
    for backend in backends:
        # Create operation with the current backend
        op = operation(backend=backend)
        
        # Warm-up run
        op(image.copy())
        
        # Timed runs
        times = []
        for _ in range(iterations):
            start_time = time.time()
            op(image.copy())
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Record results
        results.append({
            'Operation': op.name,
            'Backend': backend,
            'Mean Time (s)': np.mean(times),
            'Median Time (s)': np.median(times),
            'Min Time (s)': np.min(times),
            'Max Time (s)': np.max(times),
            'Std Time (s)': np.std(times)
        })
    
    return pd.DataFrame(results)

def run_pipeline_benchmark(image_path: str, operations: List[Callable], 
                          backends: List[str], iterations: int = 5) -> pd.DataFrame:
    """
    Benchmark a pipeline of operations with different backends.
    
    Args:
        image_path: Path to the input image
        operations: List of operation factory functions
        backends: List of backends to benchmark
        iterations: Number of iterations to run for each benchmark
        
    Returns:
        DataFrame with benchmark results
    """
    # Load image
    image = load_image(image_path)
    
    # Prepare results storage
    results = []
    
    # Run benchmarks for each backend
    for backend in backends:
        # Create pipeline with operations using the current backend
        pipeline = Pipeline(name=f"Benchmark-Pipeline-{backend}")
        for op_factory in operations:
            pipeline.add_operation(op_factory(backend=backend))
        
        # Warm-up run
        pipeline(image.copy())
        
        # Timed runs
        times = []
        for _ in range(iterations):
            start_time = time.time()
            pipeline(image.copy())
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Record results
        results.append({
            'Pipeline': pipeline.name,
            'Backend': backend,
            'Mean Time (s)': np.mean(times),
            'Median Time (s)': np.median(times),
            'Min Time (s)': np.min(times),
            'Max Time (s)': np.max(times),
            'Std Time (s)': np.std(times)
        })
    
    return pd.DataFrame(results)

def benchmark_image_sizes(operation: Callable, backend: str,
                         sizes: List[Tuple[int, int]], iterations: int = 5) -> pd.DataFrame:
    """
    Benchmark an operation with different image sizes.
    
    Args:
        operation: Operation factory function
        backend: Backend to use
        sizes: List of image sizes to benchmark
        iterations: Number of iterations to run for each benchmark
        
    Returns:
        DataFrame with benchmark results
    """
    # Create a test image
    max_size = max(sizes, key=lambda x: x[0] * x[1])
    test_image = np.random.randint(0, 256, (max_size[0], max_size[1], 3), dtype=np.uint8)
    
    # Prepare results storage
    results = []
    
    # Create operation
    op = operation(backend=backend)
    
    # Run benchmarks for each size
    for width, height in sizes:
        # Resize image
        resized = test_image[:height, :width].copy()
        
        # Warm-up run
        op(resized.copy())
        
        # Timed runs
        times = []
        for _ in range(iterations):
            start_time = time.time()
            op(resized.copy())
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Record results
        results.append({
            'Operation': op.name,
            'Image Size': f"{width}x{height}",
            'Width': width,
            'Height': height,
            'Pixels': width * height,
            'Mean Time (s)': np.mean(times),
            'Median Time (s)': np.median(times),
            'Min Time (s)': np.min(times),
            'Max Time (s)': np.max(times),
            'Std Time (s)': np.std(times)
        })
    
    return pd.DataFrame(results)

def main():
    """Run benchmarks with command-line arguments."""
    parser = argparse.ArgumentParser(description="Run image processing benchmarks")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--benchmark", type=str, choices=["operations", "pipeline", "sizes"],
                       default="operations", help="Type of benchmark to run")
    parser.add_argument("--backends", type=str, nargs="+", 
                       default=["sequential", "vectorized", "numba", "multicore", "gpu"],
                       help="Backends to benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", 
                       help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    # Create sample image if none provided
    if args.image is None or not os.path.exists(args.image):
        print("No image provided or image not found. Using random test image.")
        test_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        test_image_path = "test_image.png"
        Image.fromarray(test_image).save(test_image_path)
        args.image = test_image_path
    
    # Run the selected benchmark
    if args.benchmark == "operations":
        # Benchmark individual operations
        blur_results = run_operation_benchmark(
            args.image, 
            lambda backend: GaussianBlur(kernel_size=5, sigma=1.0, backend=backend),
            args.backends,
            args.iterations
        )
        
        edge_results = run_operation_benchmark(
            args.image,
            lambda backend: EdgeDetection(backend=backend),
            args.backends,
            args.iterations
        )
        
        # Combine results
        results = pd.concat([blur_results, edge_results])
        
    elif args.benchmark == "pipeline":
        # Benchmark a pipeline of operations
        results = run_pipeline_benchmark(
            args.image,
            [
                lambda backend: GaussianBlur(kernel_size=5, sigma=1.0, backend=backend),
                lambda backend: EdgeDetection(backend=backend)
            ],
            args.backends,
            args.iterations
        )
        
    elif args.benchmark == "sizes":
        # Benchmark with different image sizes
        sizes = [
            (640, 480),    # VGA
            (1280, 720),   # 720p
            (1920, 1080),  # 1080p
            (2560, 1440),  # 1440p
            (3840, 2160)   # 4K
        ]
        
        results = benchmark_image_sizes(
            lambda backend: GaussianBlur(kernel_size=5, sigma=1.0, backend=backend),
            args.backends[0],  # Use the first backend
            sizes,
            args.iterations
        )
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"Benchmark results saved to {args.output}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print(results[['Operation' if 'Operation' in results.columns else 'Pipeline', 
                  'Backend' if 'Backend' in results.columns else 'Image Size', 
                  'Mean Time (s)', 'Median Time (s)']])

if __name__ == "__main__":
    main()
