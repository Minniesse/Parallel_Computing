import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
import argparse
from PIL import Image
import traceback
from tqdm import tqdm  # For progress bars
import multiprocessing

# Add parent directory to path so 'src' module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        print(f"Benchmarking with backend: {backend}")
        try:
            # Create operation with the current backend
            op = operation(backend=backend)
            
            # Warm-up run
            print(f"  Running warm-up...")
            op(image.copy())
            
            # Timed runs
            times = []
            for i in tqdm(range(iterations), desc=f"  {op.name} ({backend})"):
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
        except Exception as e:
            print(f"Error with backend {backend}: {str(e)}")
            traceback.print_exc()
            results.append({
                'Operation': operation(backend='sequential').name,
                'Backend': backend,
                'Mean Time (s)': float('nan'),
                'Median Time (s)': float('nan'),
                'Min Time (s)': float('nan'),
                'Max Time (s)': float('nan'),
                'Std Time (s)': float('nan'),
                'Error': str(e)
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
        print(f"Benchmarking pipeline with backend: {backend}")
        try:
            # Create pipeline with operations using the current backend
            pipeline = Pipeline(name=f"Benchmark-Pipeline-{backend}")
            for op_factory in operations:
                pipeline.add_operation(op_factory(backend=backend))
            
            # Warm-up run
            print(f"  Running warm-up...")
            pipeline(image.copy())
            
            # Timed runs
            times = []
            for i in tqdm(range(iterations), desc=f"  Pipeline ({backend})"):
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
        except Exception as e:
            print(f"Error with pipeline using backend {backend}: {str(e)}")
            traceback.print_exc()
            results.append({
                'Pipeline': f"Benchmark-Pipeline-{backend}",
                'Backend': backend,
                'Mean Time (s)': float('nan'),
                'Median Time (s)': float('nan'),
                'Min Time (s)': float('nan'),
                'Max Time (s)': float('nan'),
                'Std Time (s)': float('nan'),
                'Error': str(e)
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
    parser.add_argument("--test-size", type=str, default="small", 
                       choices=["small", "medium", "large", "full"],
                       help="Size of test image to generate if no image is provided")
    parser.add_argument("--cpu-percent", type=int, default=75,
                       help="Percentage of CPU cores to use for multicore backend (1-100)")
    parser.add_argument("--single-core", action="store_true",
                       help="Run sequential benchmark on a single core")
    
    args = parser.parse_args()
    
    # Create sample image if none provided
    if args.image is None or not os.path.exists(args.image):
        print("No image provided or image not found. Using random test image.")
        
        # Size mapping
        size_map = {
            "small": (320, 240),     # Small image
            "medium": (640, 480),    # Medium image (VGA)
            "large": (1280, 720),    # Large image (720p)
            "full": (1920, 1080)     # Full HD image (1080p)
        }
        
        width, height = size_map.get(args.test_size, size_map["small"])
        print(f"Generating {width}x{height} test image...")
        test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        test_image_path = "test_image.png"
        Image.fromarray(test_image).save(test_image_path)
        args.image = test_image_path
    
    print(f"Using image: {args.image}")
    print(f"Backends to benchmark: {args.backends}")
    print(f"Number of iterations: {args.iterations}")
    
    try:
        # Run the selected benchmark
        if args.benchmark == "operations":
            # Benchmark individual operations
            print("\nBenchmarking Gaussian Blur...")
            
            # Create a custom operation factory with CPU usage control
            def create_gaussian_blur(backend):
                if backend == "multicore":
                    # Limit CPU usage for multicore
                    max_workers = max(1, int(multiprocessing.cpu_count() * args.cpu_percent / 100))
                    return GaussianBlur(
                        kernel_size=5, 
                        sigma=1.0, 
                        backend=backend,
                        tile_size=(512, 512)  # Larger tiles for less overhead
                    )
                else:
                    return GaussianBlur(kernel_size=5, sigma=1.0, backend=backend)
            
            blur_results = run_operation_benchmark(
                args.image, 
                create_gaussian_blur,
                args.backends,
                args.iterations
            )
            
            print("\nBenchmarking Edge Detection...")
            
            # Create a custom operation factory with CPU usage control
            def create_edge_detection(backend):
                if backend == "multicore":
                    # Limit CPU usage for multicore
                    max_workers = max(1, int(multiprocessing.cpu_count() * args.cpu_percent / 100))
                    return EdgeDetection(
                        backend=backend,
                        tile_size=(512, 512)  # Larger tiles for less overhead
                    )
                else:
                    return EdgeDetection(backend=backend)
            
            edge_results = run_operation_benchmark(
                args.image,
                create_edge_detection,
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
                (320, 240),    # Small
                (640, 480),    # VGA
                (1280, 720),   # 720p
                (1920, 1080),  # 1080p
            ]
            
            if len(args.backends) > 0:
                backend = args.backends[0]
                print(f"\nBenchmarking different image sizes with {backend} backend...")
                results = benchmark_image_sizes(
                    lambda backend: GaussianBlur(kernel_size=5, sigma=1.0, backend=backend),
                    backend,
                    sizes,
                    args.iterations
                )
            else:
                print("No backends specified.")
                return
    
        # Save results
        results.to_csv(args.output, index=False)
        print(f"\nBenchmark results saved to {args.output}")
        
        # Print summary
        print("\nBenchmark Summary:")
        summary_cols = ['Operation' if 'Operation' in results.columns else 'Pipeline', 
                      'Backend' if 'Backend' in results.columns else 'Image Size', 
                      'Mean Time (s)', 'Median Time (s)']
        summary_cols = [col for col in summary_cols if col in results.columns]
        if 'Error' in results.columns:
            summary_cols.append('Error')
        print(results[summary_cols])
    
    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
