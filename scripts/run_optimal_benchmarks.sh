#!/bin/bash

# Script to run benchmarks with optimal settings
# Usage: ./run_optimal_benchmarks.sh [test_size] [output_file]

# Default parameters
TEST_SIZE=${1:-small}
OUTPUT_FILE=${2:-benchmark_results_optimized.csv}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Optimize system if possible
if command_exists cpupower && [ "$EUID" -eq 0 ]; then
  echo "Setting CPU governor to performance mode..."
  cpupower frequency-set -g performance
else
  echo "Warning: cpupower not available or not running as root."
  echo "For optimal results, run: sudo cpupower frequency-set -g performance"
fi

# Clear caches if possible
if [ "$EUID" -eq 0 ]; then
  echo "Clearing system caches..."
  sync
  echo 3 > /proc/sys/vm/drop_caches
else
  echo "Warning: Cannot clear caches without root privileges."
  echo "For optimal results, run: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
fi

# Run benchmarks for each backend separately to avoid interference
echo "Running sequential benchmark..."
python ../benchmark/run_benchmarks.py --backends sequential --test-size $TEST_SIZE --output ${OUTPUT_FILE%.csv}_sequential.csv --iterations 3

echo "Running vectorized benchmark..."
python ../benchmark/run_benchmarks.py --backends vectorized --test-size $TEST_SIZE --output ${OUTPUT_FILE%.csv}_vectorized.csv --iterations 5

echo "Running numba benchmark..."
python ../benchmark/run_benchmarks.py --backends numba --test-size $TEST_SIZE --output ${OUTPUT_FILE%.csv}_numba.csv --iterations 5

echo "Running multicore benchmark..."
python ../benchmark/run_benchmarks.py --backends multicore --test-size $TEST_SIZE --output ${OUTPUT_FILE%.csv}_multicore.csv --iterations 5 --cpu-percent 75

echo "Running GPU benchmark..."
python ../benchmark/run_benchmarks.py --backends gpu --test-size $TEST_SIZE --output ${OUTPUT_FILE%.csv}_gpu.csv --iterations 5

# Combine results
echo "Combining results..."
python -c "
import pandas as pd
import glob

# Get all result files
files = glob.glob('${OUTPUT_FILE%.csv}_*.csv')
dfs = [pd.read_csv(f) for f in files]

# Combine and save
combined = pd.concat(dfs)
combined.to_csv('$OUTPUT_FILE', index=False)
print(f'Results saved to {OUTPUT_FILE}')
"

echo "Benchmarking complete!"
