#!/bin/bash

# Script to run all benchmarks and generate plots

# Find the project root and build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    echo "       Make sure you've built the project with build.sh first."
    exit 1
fi

# Check if executables exist
if [[ ! -f "$BUILD_DIR/benchmark_cpu" ]] || [[ ! -f "$BUILD_DIR/benchmark_gpu" ]] || 
   [[ ! -f "$BUILD_DIR/compare_libraries" ]] || [[ ! -f "$BUILD_DIR/benchmark_all" ]]; then
    echo "Error: Benchmark executables not found in $BUILD_DIR."
    echo "       Please build the project with build.sh first."
    exit 1
fi

# Go to build directory to run benchmarks
cd $BUILD_DIR

# Set default matrix sizes and iterations
SIZES=(128 256 512 1024 2048 4096)
ITERATIONS=3

# Parse command line arguments
if [ $# -ge 1 ]; then
    SIZES=($@)
    ITERATIONS=${SIZES[-1]}
    unset 'SIZES[-1]'
fi

# Create results directory
mkdir -p results

echo "==================================================="
echo "Running all matrix multiplication benchmarks"
echo "==================================================="
echo "Matrix sizes: ${SIZES[@]}"
echo "Iterations per benchmark: $ITERATIONS"
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "==================================================="

# Run CPU benchmarks
echo -e "\nRunning CPU benchmarks..."
./benchmark_cpu ${SIZES[@]} $ITERATIONS
if [ $? -eq 0 ] && [ -f "cpu_benchmark_results.csv" ]; then
    mv cpu_benchmark_results.csv results/
    echo "CPU benchmarks completed successfully."
else
    echo "Error: CPU benchmarks failed or results file not generated."
fi

# Run GPU benchmarks if CUDA is available
if which nvcc > /dev/null 2>&1; then
    echo -e "\nRunning GPU benchmarks..."
    ./benchmark_gpu ${SIZES[@]} $ITERATIONS
    if [ $? -eq 0 ]; then
        if [ -f "gpu_benchmark_results.csv" ]; then
            mv gpu_benchmark_results.csv results/
        fi
        if [ -f "gpu_memory_transfer.csv" ]; then
            mv gpu_memory_transfer.csv results/
        fi
        echo "GPU benchmarks completed successfully."
    else
        echo "Error: GPU benchmarks failed or results files not generated."
    fi
else
    echo "CUDA not found, skipping GPU benchmarks"
fi

# Run library comparison benchmarks
echo -e "\nRunning library comparison benchmarks..."
./compare_libraries ${SIZES[@]} $ITERATIONS
if [ $? -eq 0 ] && [ -f "library_comparison.csv" ]; then
    mv library_comparison.csv results/
    echo "Library comparison benchmarks completed successfully."
else
    echo "Error: Library comparison benchmarks failed or results file not generated."
fi

# Run comprehensive benchmarks
echo -e "\nRunning comprehensive benchmarks..."
./benchmark_all ${SIZES[@]} $ITERATIONS
if [ $? -eq 0 ] && [ -f "benchmark_results.csv" ]; then
    mv benchmark_results.csv results/
    echo "Comprehensive benchmarks completed successfully."
else
    echo "Error: Comprehensive benchmarks failed or results file not generated."
fi

# Generate plots if Python and required packages are available
if which python3 > /dev/null 2>&1; then
    echo -e "\nGenerating plots..."
    
    # Check if required Python packages are installed
    if python3 -c "import pandas, matplotlib, numpy" > /dev/null 2>&1; then
        # Generate CPU benchmark plots
        if [ -f "results/cpu_benchmark_results.csv" ]; then
            python3 $PROJECT_ROOT/scripts/plot_results.py results/cpu_benchmark_results.csv performance
        fi
        
        # Generate GPU benchmark plots if available
        if [ -f "results/gpu_benchmark_results.csv" ]; then
            python3 $PROJECT_ROOT/scripts/plot_results.py results/gpu_benchmark_results.csv performance
        fi
        
        if [ -f "results/gpu_memory_transfer.csv" ]; then
            python3 $PROJECT_ROOT/scripts/plot_results.py results/gpu_memory_transfer.csv memory
        fi
        
        # Generate library comparison plots
        if [ -f "results/library_comparison.csv" ]; then
            python3 $PROJECT_ROOT/scripts/plot_results.py results/library_comparison.csv library
        fi
        
        # Generate comprehensive benchmark plots
        if [ -f "results/benchmark_results.csv" ]; then
            python3 $PROJECT_ROOT/scripts/plot_results.py results/benchmark_results.csv performance
        fi
        
        echo "Plots have been generated in the plots directory"
    else
        echo "Warning: Required Python packages (pandas, matplotlib, numpy) not found."
        echo "Please install them with: pip install pandas matplotlib numpy"
        echo "Then run: python3 $PROJECT_ROOT/scripts/plot_results.py results/benchmark_results.csv"
    fi
else
    echo "Python not found, skipping plot generation"
    echo "To generate plots manually, run: python3 $PROJECT_ROOT/scripts/plot_results.py results/benchmark_results.csv"
fi

echo -e "\nAll benchmarks completed. Results saved in the results directory."
