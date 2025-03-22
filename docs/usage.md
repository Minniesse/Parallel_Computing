# Parallel Image Processing - Usage Guide

This guide provides examples of how to use the parallel image processing library.

## Installation

Install the package from the project directory:

```bash
pip install -e .
```

## Basic Usage

### Loading and Saving Images

```python
from src.core.utils import load_image, save_image

# Load an image
image = load_image("path/to/image.jpg")

# Save an image
save_image(image, "output/processed.jpg")
```

### Single Operations

Each operation can be used on its own with different backends:

```python
import numpy as np
from src.operations.filters import GaussianBlur
from src.operations.color_ops import ColorTransformation

# Load or create an image
image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

# Apply Gaussian blur with different backends
gaussian_blur = GaussianBlur(kernel_size=5, sigma=1.0, backend="gpu")
blurred_image = gaussian_blur(image)

# Apply color transformation
sepia = ColorTransformation(transformation="sepia", backend="vectorized")
sepia_image = sepia(image)
```

### Creating Pipelines

Chain multiple operations together in a pipeline:

```python
from src.core.pipeline import Pipeline
from src.operations.filters import GaussianBlur, EdgeDetection
from src.operations.color_ops import ColorTransformation

# Create a pipeline
pipeline = Pipeline(name="MyImagePipeline")

# Add operations to the pipeline
pipeline.add_operation(GaussianBlur(kernel_size=5, sigma=1.0, backend="auto"))
pipeline.add_operation(EdgeDetection(backend="auto"))
pipeline.add_operation(ColorTransformation(transformation="invert", backend="auto"))

# Process an image through the pipeline
processed_image = pipeline(image)

# Get performance statistics
stats = pipeline.get_performance_stats()
print(f"Total pipeline time: {stats['total_time']:.3f} seconds")
```

### Parallel Processing

Use the parallel pipeline for improved performance:

```python
from src.core.pipeline import ParallelPipeline
from src.core.scheduler import TileScheduler, WorkStealingScheduler

# Create a tile scheduler
scheduler = TileScheduler(tile_size=(256, 256))

# Create a parallel pipeline with the scheduler
parallel_pipeline = ParallelPipeline(name="ParallelPipeline", scheduler=scheduler)

# Add operations
parallel_pipeline.add_operation(GaussianBlur(kernel_size=5, sigma=1.0, backend="auto"))
parallel_pipeline.add_operation(EdgeDetection(backend="auto"))

# Process an image
processed_image = parallel_pipeline(image)
```

## Available Operations

### Filters

- **GaussianBlur**: Apply Gaussian blur to an image
- **EdgeDetection**: Detect edges in an image

### Color Operations

- **ColorTransformation**: Apply color transformations (grayscale, invert, sepia)

### Geometric Transformations

- **Resize**: Resize an image to a target size
- **Rotate**: Rotate an image by a specified angle
- **Crop**: Crop a region from an image

### Feature Detection

- **HarrisCornerDetection**: Detect corners in an image using Harris algorithm
- **CannyEdgeDetection**: Detect edges using the Canny algorithm

## Available Backends

Most operations support multiple backends:

- **sequential**: Pure Python implementation (slowest but most compatible)
- **vectorized**: NumPy vectorized implementation
- **numba**: Numba JIT-compiled implementation (for compute-intensive operations)
- **multicore**: Multi-core CPU implementation using process pools
- **gpu**: GPU-accelerated implementation using CuPy
- **torch**: GPU-accelerated implementation using PyTorch

Use "auto" to automatically select the best available backend.

## Benchmarking

Use the benchmarking tools to compare performance:

```python
from benchmark.run_benchmarks import run_operation_benchmark

# Compare backends for Gaussian blur
results = run_operation_benchmark(
    "image.jpg",
    lambda backend: GaussianBlur(kernel_size=5, sigma=1.0, backend=backend),
    ["sequential", "vectorized", "multicore", "gpu"],
    iterations=5
)

# Visualize results
from benchmark.visualize_results import plot_operation_comparison
plot_operation_comparison(results)
```

## Example Script

Run the included example script to see the library in action:

```bash
python example.py --image my_image.jpg --operation pipeline --parallel
```

Options:
- `--image`: Path to input image (will create a test image if not provided)
- `--output`: Output directory (default: "output")
- `--operation`: Operation to perform (blur, edge, color, pipeline)
- `--backends`: Backends to compare (for individual operations)
- `--parallel`: Use parallel pipeline (for pipeline operation)
