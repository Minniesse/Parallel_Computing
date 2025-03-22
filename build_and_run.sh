#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Build failed. Please fix the errors and try again."
    exit 1
fi

# Print available executables
echo "Available executables:"
find . -type f -executable -not -path "*/\.*" | sort

# # Run CPU benchmark if it exists
# if [ -f "./benchmark_cpu" ]; then
#     echo "Running CPU benchmark..."
#     ./benchmark_cpu
# elif [ -f "./bin/benchmark_cpu" ]; then
#     echo "Running CPU benchmark..."
#     ./bin/benchmark_cpu
# else
#     echo "CPU benchmark executable not found"
# fi

# # Check if GPU benchmark is available
# if [ -f "./benchmark_gpu" ]; then
#     echo "Running GPU benchmark..."
#     ./benchmark_gpu
# elif [ -f "./bin/benchmark_gpu" ]; then
#     echo "Running GPU benchmark..."
#     ./bin/benchmark_gpu
# else
#     echo "GPU benchmark executable not found"
# fi

# # Run comparison if available
# if [ -f "./compare_libraries" ]; then
#     echo "Running library comparison..."
#     ./compare_libraries
# elif [ -f "./bin/compare_libraries" ]; then
#     echo "Running library comparison..."
#     ./bin/compare_libraries
# else
#     echo "Library comparison executable not found"
# fi

# # Run comprehensive benchmark if it exists
# if [ -f "./benchmark_all" ]; then
#     echo "Running comprehensive benchmark..."
#     ./benchmark_all
# elif [ -f "./bin/benchmark_all" ]; then
#     echo "Running comprehensive benchmark..."
#     ./bin/benchmark_all
# else
#     echo "Comprehensive benchmark executable not found"
# fi

# # Return to original directory
# cd ..

# echo "All benchmarks completed"
