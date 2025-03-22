#!/bin/bash

# Script to run all benchmarks and generate plots
# Make sure to run this from the build directory

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
echo "==================================================="

# Run CPU benchmarks
echo -e "\nRunning CPU benchmarks..."
./benchmark_cpu ${SIZES[@]} $ITERATIONS
mv cpu_benchmark_results.csv results/

# Run GPU benchmarks if CUDA is available
if which nvcc > /dev/null 2>&1; then
    echo -e "\nRunning GPU benchmarks..."
    ./benchmark_gpu ${SIZES[@]} $ITERATIONS
    mv gpu_benchmark_results.csv results/
    mv gpu_memory_transfer.csv results/
else
    echo "CUDA not found, skipping GPU benchmarks"
fi

# Run library comparison benchmarks
echo -e "\nRunning library comparison benchmarks..."
./compare_libraries ${SIZES[@]} $ITERATIONS
mv library_comparison.csv results/

# Run comprehensive benchmarks
echo -e "\nRunning comprehensive benchmarks..."
./benchmark_all ${SIZES[@]} $ITERATIONS
mv benchmark_results.csv results/

# Generate plots if Python and required packages are available
if which python3 > /dev/null 2>&1; then
    echo -e "\nGenerating plots..."
    
    # Check if required Python packages are installed
    if python3 -c "import pandas, matplotlib, numpy" > /dev/null 2>&1; then
        # Generate CPU benchmark plots
        python3 ../scripts/plot_results.py results/cpu_benchmark_results.csv performance
        
        # Generate GPU benchmark plots if available
        if [ -f "results/gpu_benchmark_results.csv" ]; then
            python3 ../scripts/plot_results.py results/gpu_benchmark_results.csv performance
            python3 ../scripts/plot_results.py results/gpu_memory_transfer.csv memory
        fi
        
        # Generate library comparison plots
        python3 ../scripts/plot_results.py results/library_comparison.csv library
        
        # Generate comprehensive benchmark plots
        python3 ../scripts/plot_results.py results/benchmark_results.csv performance
        
        echo "Plots have been generated in the plots directory"
    else
        echo "Warning: Required Python packages (pandas, matplotlib, numpy) not found."
        echo "Please install them with: pip install pandas matplotlib numpy"
        echo "Then run: python3 ../scripts/plot_results.py results/benchmark_results.csv"
    fi
else
    echo "Python not found, skipping plot generation"
    echo "To generate plots manually, run: python3 ../scripts/plot_results.py results/benchmark_results.csv"
fi

echo -e "\nAll benchmarks completed. Results saved in the results directory."
