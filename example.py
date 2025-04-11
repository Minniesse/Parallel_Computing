"""
Example demonstrating the parallel image processing pipeline.
"""
import os
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.core.pipeline import Pipeline, ParallelPipeline
from src.core.scheduler import TileScheduler, WorkStealingScheduler
from src.core.utils import load_image, save_image
from src.operations.filters import GaussianBlur, EdgeDetection
from src.operations.color_ops import ColorTransformation
from src.operations.transformations import Resize, Rotate, Crop
from src.operations.features import HarrisCornerDetection, CannyEdgeDetection

def create_test_image(output_path: str, size: tuple = (1024, 1024)):
    """Create a test image if none is provided."""
    # Create a test pattern with a mix of shapes for testing edge detection
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Add background color
    image[:, :] = [240, 240, 240]
    
    # Add shapes
    # Circle
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 4
    y, x = np.ogrid[:size[0], :size[1]]
    circle_mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    image[circle_mask] = [0, 128, 255]
    
    # Rectangle
    rect_start = (size[0] // 4, size[1] // 4)
    rect_end = (3 * size[0] // 4, 3 * size[1] // 4)
    rect_width = 10
    # Top
    image[rect_start[0]:rect_start[0]+rect_width, rect_start[1]:rect_end[1]] = [255, 0, 0]
    # Bottom
    image[rect_end[0]-rect_width:rect_end[0], rect_start[1]:rect_end[1]] = [255, 0, 0]
    # Left
    image[rect_start[0]:rect_end[0], rect_start[1]:rect_start[1]+rect_width] = [255, 0, 0]
    # Right
    image[rect_start[0]:rect_end[0], rect_end[1]-rect_width:rect_end[1]] = [255, 0, 0]
    
    # Add grid pattern for testing blurring
    grid_spacing = 32
    for i in range(0, size[0], grid_spacing):
        image[i:i+1, :] = [0, 0, 0]
    for j in range(0, size[1], grid_spacing):
        image[:, j:j+1] = [0, 0, 0]
    
    # Save the image
    Image.fromarray(image).save(output_path)
    return output_path

def compare_backends(image_path: str, output_dir: str, operation: str, 
                    backends: list = ['sequential', 'vectorized', 'multicore', 'gpu', 'torch']):
    """
    Compare different backends for an operation.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output images
        operation: Operation to test (blur, edge, color)
        backends: List of backends to compare
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print("Backend: ", backends)
    
    # Load image
    image = load_image(image_path)
    
    # Print image info
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Save original
    save_image(image, os.path.join(output_dir, 'original.png'))
    
    # Create operation based on type
    operations = {}
    timings = {}
    
    for backend in backends:
        if operation == 'blur':
            op = GaussianBlur(kernel_size=9, sigma=2.0, backend=backend)
        elif operation == 'edge':
            op = EdgeDetection(backend=backend)
        elif operation == 'color':
            op = ColorTransformation(transformation="sepia", backend=backend)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Apply operation and time it
        print(f"Applying {op.name}...")
        start_time = time.time()
        processed = op(image)
        elapsed = time.time() - start_time
        
        # Save result and timing
        operations[backend] = processed
        timings[backend] = elapsed
        save_image(processed, os.path.join(output_dir, f'{operation}_{backend}.png'))
        print(f"  {backend} took {elapsed:.3f} seconds")
    
    # Print speedup compared to sequential
    if 'sequential' in backends:
        sequential_time = timings['sequential']
        print("\nSpeedup compared to sequential:")
        for backend, timing in timings.items():
            if backend != 'sequential':
                speedup = sequential_time / timing
                print(f"  {backend}: {speedup:.2f}x faster")
    
    return operations, timings

def demo_pipeline(image_path: str, output_dir: str, use_parallel: bool = False):
    """
    Demonstrate a pipeline of operations.
    
    Args:
        image_path: Path to input image
        output_dir: Directory for output images
        use_parallel: Whether to use parallel pipeline
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = load_image(image_path)
    
    # Create pipeline
    if use_parallel:
        scheduler = TileScheduler(tile_size=(256, 256))
        pipeline = ParallelPipeline(name="ParallelDemoPipeline", scheduler=scheduler)
    else:
        pipeline = Pipeline(name="DemoPipeline")
    
    # Add operations to the pipeline
    pipeline.add_operation(GaussianBlur(kernel_size=5, sigma=1.0, backend="auto"))
    pipeline.add_operation(EdgeDetection(backend="auto"))
    pipeline.add_operation(ColorTransformation(transformation="invert", backend="auto"))
    
    # Apply pipeline and time it
    print(f"Applying pipeline {pipeline.name}...")
    start_time = time.time()
    processed = pipeline(image)
    elapsed = time.time() - start_time
    
    # Save result
    save_image(processed, os.path.join(output_dir, f'pipeline_{"parallel" if use_parallel else "sequential"}.png'))
    print(f"Pipeline took {elapsed:.3f} seconds")
    
    # Print pipeline performance stats
    stats = pipeline.get_performance_stats()
    print("\nPipeline performance stats:")
    print(f"  Total time: {stats['total_time']:.3f} seconds")
    print("  Operation times:")
    for op_name, op_time in stats['operations'].items():
        print(f"    {op_name}: {op_time:.3f} seconds")
    
    return processed, stats

def main():
    """Run the example with fixed parameters for demonstration."""
    # Define parameters
    output_dir = "output_demo"
    backends = [
        "sequential",  # Standard single-threaded Python execution
        "vectorized",  # Uses NumPy for optimized array operations
        "multicore",   # Uses multiprocessing/threading for parallel execution on CPU cores
        "gpu",         # Uses GPU acceleration (e.g., via CuPy or Numba if available)
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test image
    test_image_path = os.path.join(output_dir, "test_image.png")
    image_path = create_test_image(test_image_path)
    print(f"Created test image: {image_path}\n")
    
    # --- Compare Backends ---
    print("="*20 + " Comparing Backends " + "="*20)
    
    print("\n--- Comparing Blur Backends ---")
    compare_backends(image_path, os.path.join(output_dir, "compare_blur"), "blur", backends)
    
    print("\n--- Comparing Edge Detection Backends ---")
    compare_backends(image_path, os.path.join(output_dir, "compare_edge"), "edge", backends)
    
    print("\n--- Comparing Color Transformation Backends ---")
    compare_backends(image_path, os.path.join(output_dir, "compare_color"), "color", backends)
    
    print("\n" + "="*20 + " Demonstrating Pipelines " + "="*20)
    
    # --- Demo Sequential Pipeline ---
    print("\n--- Running Sequential Pipeline ---")
    demo_pipeline(image_path, os.path.join(output_dir, "pipeline_sequential"), use_parallel=False)
    
    # --- Demo Parallel Pipeline ---
    print("\n--- Running Parallel Pipeline ---")
    demo_pipeline(image_path, os.path.join(output_dir, "pipeline_parallel"), use_parallel=True)
    
    print("\nDemo finished. Check the '{}' directory for results.".format(output_dir))


if __name__ == "__main__":
    main()
