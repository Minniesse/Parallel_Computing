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

# Add external libraries for comparison
import cv2
try:
    # Check if CUDA is available in OpenCV
    cv2_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
except:
    cv2_cuda_available = False
    
# Import scikit-image components
from skimage.filters import gaussian as skimage_gaussian
from skimage.feature import canny as skimage_canny

# Add parent directory to path so 'src' module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pipeline and operations
from src.core.pipeline import Pipeline
from src.core.utils import load_image, save_image
from src.operations.filters import GaussianBlur, EdgeDetection

# External library wrappers
class ExternalLibraryOperation:
    """Base class for external library operations to match interface with custom operations."""
    def __init__(self, name):
        self.name = name
        
    def __call__(self, image):
        raise NotImplementedError("Subclasses must implement __call__")

class OpenCVGaussianBlur(ExternalLibraryOperation):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__("OpenCV-CPU-GaussianBlur")
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, image):
        # OpenCV expects kernel size as tuple and BGR format
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 3:
            return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        return image

class OpenCVCudaGaussianBlur(ExternalLibraryOperation):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__("OpenCV-CUDA-GaussianBlur")
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, image):
        if not cv2_cuda_available:
            raise RuntimeError("CUDA is not available in OpenCV")
            
        # Convert to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        
        # Apply filter
        result = cv2.cuda.createGaussianFilter(
            gpu_image.type(), gpu_image.type(),
            (self.kernel_size, self.kernel_size), 
            self.sigma
        ).apply(gpu_image)
        
        # Download result
        return result.download()

class SkimageGaussianBlur(ExternalLibraryOperation):
    def __init__(self, sigma=1.0):
        super().__init__("Skimage-GaussianBlur")
        self.sigma = sigma
        
    def __call__(self, image):
        # skimage accepts sigma not kernel size
        return skimage_gaussian(image, sigma=self.sigma, preserve_range=True).astype(image.dtype)

class OpenCVEdgeDetection(ExternalLibraryOperation):
    def __init__(self):
        super().__init__("OpenCV-CPU-EdgeDetection")
        
    def __call__(self, image):
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # Apply Canny edge detection
        return cv2.Canny(gray, 100, 200)

class OpenCVCudaEdgeDetection(ExternalLibraryOperation):
    def __init__(self):
        super().__init__("OpenCV-CUDA-EdgeDetection")
        
    def __call__(self, image):
        if not cv2_cuda_available:
            raise RuntimeError("CUDA is not available in OpenCV")
            
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Upload to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(gray)
        
        # Apply Canny
        detector = cv2.cuda.createCannyEdgeDetector(100, 200)
        result = detector.detect(gpu_image)
        
        # Download result
        return result.download()

class SkimageEdgeDetection(ExternalLibraryOperation):
    def __init__(self):
        super().__init__("Skimage-EdgeDetection")
        
    def __call__(self, image):
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # Apply Canny edge detection
        return skimage_canny(gray, sigma=1.0)

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

def run_external_comparison(image_path: str, operation_type: str, 
                           iterations: int = 5) -> pd.DataFrame:
    """
    Benchmark custom implementation against external libraries.
    
    Args:
        image_path: Path to the input image
        operation_type: Type of operation ('gaussian' or 'edge')
        iterations: Number of iterations to run for each benchmark
        
    Returns:
        DataFrame with benchmark results
    """
    # Load image
    image = load_image(image_path)
    
    # Prepare results storage
    results = []
    
    # Define operations based on type
    operations = []
    if operation_type == 'gaussian':
        # Custom implementations
        operations.extend([
            ('Custom-Sequential', GaussianBlur(kernel_size=5, sigma=1.0, backend='sequential')),
            ('Custom-Vectorized', GaussianBlur(kernel_size=5, sigma=1.0, backend='vectorized')),
            ('Custom-Numba', GaussianBlur(kernel_size=5, sigma=1.0, backend='numba')),
            ('Custom-Multicore', GaussianBlur(kernel_size=5, sigma=1.0, backend='multicore', tile_size=(512, 512))),
        ])
        # Add GPU if supported
        try:
            operations.append(('Custom-GPU', GaussianBlur(kernel_size=5, sigma=1.0, backend='gpu')))
        except Exception as e:
            print(f"GPU backend not available: {str(e)}")
            
        # External libraries
        operations.extend([
            ('OpenCV-CPU', OpenCVGaussianBlur(kernel_size=5, sigma=1.0)),
            ('Skimage', SkimageGaussianBlur(sigma=1.0)),
        ])
        # Add OpenCV CUDA if available
        if cv2_cuda_available:
            operations.append(('OpenCV-CUDA', OpenCVCudaGaussianBlur(kernel_size=5, sigma=1.0)))
    
    elif operation_type == 'edge':
        # Custom implementations
        operations.extend([
            ('Custom-Sequential', EdgeDetection(backend='sequential')),
            ('Custom-Vectorized', EdgeDetection(backend='vectorized')),
            ('Custom-Numba', EdgeDetection(backend='numba')),
            ('Custom-Multicore', EdgeDetection(backend='multicore', tile_size=(512, 512))),
        ])
        # Add GPU if supported
        try:
            operations.append(('Custom-GPU', EdgeDetection(backend='gpu')))
        except Exception as e:
            print(f"GPU backend not available: {str(e)}")
            
        # External libraries
        operations.extend([
            ('OpenCV-CPU', OpenCVEdgeDetection()),
            ('Skimage', SkimageEdgeDetection()),
        ])
        # Add OpenCV CUDA if available
        if cv2_cuda_available:
            operations.append(('OpenCV-CUDA', OpenCVCudaEdgeDetection()))
    
    # Run benchmarks for each operation
    for name, op in operations:
        print(f"Benchmarking: {name}")
        try:
            # Warm-up run
            print(f"  Running warm-up...")
            op(image.copy())
            
            # Timed runs
            times = []
            for i in tqdm(range(iterations), desc=f"  {name}"):
                start_time = time.time()
                op(image.copy())
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Record results
            results.append({
                'Operation': op.name,
                'Implementation': name,
                'Mean Time (s)': np.mean(times),
                'Median Time (s)': np.median(times),
                'Min Time (s)': np.min(times),
                'Max Time (s)': np.max(times),
                'Std Time (s)': np.std(times)
            })
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            traceback.print_exc()
            results.append({
                'Operation': op.name,
                'Implementation': name,
                'Mean Time (s)': float('nan'),
                'Median Time (s)': float('nan'),
                'Min Time (s)': float('nan'),
                'Max Time (s)': float('nan'),
                'Std Time (s)': float('nan'),
                'Error': str(e)
            })
    
    return pd.DataFrame(results)

def plot_benchmark_comparison(results, output_file=None):
    """
    Generate a bar chart comparing the performance of different implementations.
    
    Args:
        results: DataFrame with benchmark results
        output_file: Path to save the plot image
    """
    if 'Implementation' not in results.columns:
        if 'Backend' in results.columns:
            results['Implementation'] = results['Backend']
        else:
            return  # Can't plot without implementation/backend info
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for mean execution times
    implementations = results['Implementation'].unique()
    operations = results['Operation'].unique()
    
    bar_width = 0.8 / len(operations)
    index = np.arange(len(implementations))
    
    # Create bars for each operation
    for i, operation in enumerate(operations):
        op_data = results[results['Operation'] == operation]
        times = []
        for impl in implementations:
            impl_data = op_data[op_data['Implementation'] == impl]
            if not impl_data.empty:
                times.append(impl_data['Mean Time (s)'].values[0])
            else:
                times.append(0)
                
        plt.bar(index + i * bar_width, times, bar_width, 
                label=operation, alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('Implementation')
    plt.ylabel('Mean Execution Time (s)')
    plt.title('Performance Comparison')
    plt.xticks(index + bar_width * (len(operations) - 1) / 2, implementations, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file)
        print(f"Performance comparison plot saved to {output_file}")
    
    plt.close()

def main():
    """Run benchmarks with command-line arguments."""
    parser = argparse.ArgumentParser(description="Run image processing benchmarks")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--benchmark", type=str, 
                        choices=["operations", "pipeline", "sizes", "external-comparison"],
                        default="operations", help="Type of benchmark to run")
    parser.add_argument("--backends", type=str, nargs="+", 
                       default=["sequential", "vectorized", "numba", "multicore", "gpu"],
                       help="Backends to benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", 
                       help="Output file for benchmark results")
    parser.add_argument("--plot", type=str, default=True,
                       help="Generate performance comparison plot and save to this file")
    parser.add_argument("--test-size", type=str, default="small", 
                       choices=["small", "medium", "large", "full"],
                       help="Size of test image to generate if no image is provided")
    parser.add_argument("--cpu-percent", type=int, default=75,
                       help="Percentage of CPU cores to use for multicore backend (1-100)")
    parser.add_argument("--single-core", action="store_true",
                       help="Run sequential benchmark on a single core")
    parser.add_argument("--operation", type=str, default="gaussian",
                       choices=["gaussian", "edge"],
                       help="Operation to benchmark when using external comparison")
    
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
            
        elif args.benchmark == "external-comparison":
            # Benchmark against external libraries
            print("\nBenchmarking against external libraries...")
            results = run_external_comparison(
                args.image,
                args.operation,
                args.iterations
            )
    
        # Save results
        results.to_csv(args.output, index=False)
        print(f"\nBenchmark results saved to {args.output}")
        
        # Generate plot if requested
        if args.plot:
            plot_benchmark_comparison(results, args.plot)
        
        # Print summary
        print("\nBenchmark Summary:")
        if args.benchmark == "external-comparison":
            summary_cols = ['Operation', 'Implementation', 'Mean Time (s)', 'Median Time (s)']
        else:
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
