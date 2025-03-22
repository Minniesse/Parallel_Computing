#!/bin/bash

# Directory containing the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/data/results"
VISUALIZATIONS_DIR="$PROJECT_ROOT/showcase/visualizations"

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$VISUALIZATIONS_DIR"

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Please build the project first."
    exit 1
fi

# Change to build directory
cd "$BUILD_DIR"

# Run the CPU benchmark
echo "Running CPU benchmark..."
if [ -f "./benchmark_cpu" ]; then
    ./benchmark_cpu --output "$RESULTS_DIR/cpu_benchmark_results.csv"
elif [ -f "./bin/benchmark_cpu" ]; then
    ./bin/benchmark_cpu --output "$RESULTS_DIR/cpu_benchmark_results.csv"
else
    echo "CPU benchmark executable not found. Skipping."
fi

# Run the GPU benchmark
echo "Running GPU benchmark..."
if [ -f "./benchmark_gpu" ]; then
    ./benchmark_gpu --output "$RESULTS_DIR/gpu_benchmark_results.csv"
elif [ -f "./bin/benchmark_gpu" ]; then
    ./bin/benchmark_gpu --output "$RESULTS_DIR/gpu_benchmark_results.csv"
else
    echo "GPU benchmark executable not found. Skipping."
fi

# Run the library comparison benchmark
echo "Running library comparison benchmark..."
if [ -f "./compare_libraries" ]; then
    ./compare_libraries --output "$RESULTS_DIR/library_comparison_results.csv"
elif [ -f "./bin/compare_libraries" ]; then
    ./bin/compare_libraries --output "$RESULTS_DIR/library_comparison_results.csv"
else
    echo "Library comparison executable not found. Please build the project first."
    exit 1
fi

# Run the visualization script
cd "$SCRIPT_DIR"
echo "Generating visualizations..."
if [ -f "./visualize_benchmarks.py" ]; then
    # Only use the parameters currently supported by the script
    python3 ./visualize_benchmarks.py --input "$RESULTS_DIR/library_comparison_results.csv" --output-dir "$VISUALIZATIONS_DIR"
    
    # Visualize CPU results if available
    if [ -f "$RESULTS_DIR/cpu_benchmark_results.csv" ]; then
        echo "Note: CPU benchmark results available but not visualized. Update visualize_benchmarks.py to support them."
    fi
    
    # Visualize GPU results if available
    if [ -f "$RESULTS_DIR/gpu_benchmark_results.csv" ]; then
        echo "Note: GPU benchmark results available but not visualized. Update visualize_benchmarks.py to support them."
    fi
else
    echo "Visualization script not found."
    exit 1
fi

echo "Benchmark and visualization completed successfully!"
echo "Results saved to: $RESULTS_DIR"
echo "Visualizations saved to: $VISUALIZATIONS_DIR"
