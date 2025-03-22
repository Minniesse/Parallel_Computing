# High-Performance Parallel Matrix Multiplication - Project Structure

This document describes the organization and structure of the project.

## Directory Structure

```
matrix-multiplication/
│
├── include/                      # Header files
│   ├── common/                   # Common utility headers
│   │   ├── matrix.h              # Matrix data structure definitions
│   │   ├── timing.h              # Timing and benchmarking utilities
│   │   └── utils.h               # General utility functions
│   │
│   ├── cpu/                      # CPU implementation headers
│   │   ├── naive.h               # Naive implementation header
│   │   ├── blocked.h             # Cache-blocked implementation header
│   │   ├── simd.h                # SIMD implementation header
│   │   └── threaded.h            # Multithreaded implementation header
│   │
│   └── gpu/                      # GPU implementation headers
│       ├── cuda_wrapper.h        # Wrapper for CUDA implementation
│       └── multi_gpu.h           # Multi-GPU implementation header
│
├── src/                          # Source code files
│   ├── common/                   # Common utilities implementation
│   │   ├── matrix.cpp            # Matrix operations implementation
│   │   ├── timing.cpp            # Timing utilities implementation
│   │   └── utils.cpp             # General utilities implementation
│   │
│   ├── cpu/                      # CPU implementations
│   │   ├── naive.cpp             # Naive triple-loop implementation
│   │   ├── blocked.cpp           # Cache-blocked implementation
│   │   ├── simd.cpp              # SIMD (AVX-512) implementation
│   │   └── threaded.cpp          # OpenMP multithreaded implementation
│   │
│   ├── gpu/                      # GPU implementations
│   │   ├── cuda_wrapper.cpp      # C++ wrapper for CUDA kernels
│   │   ├── matmul_kernel.cu      # CUDA kernels for matrix multiplication
│   │   └── multi_gpu.cpp         # Multi-GPU implementation
│   │
│   ├── main.cpp                  # Main program entry point
│   └── adaptive.cpp              # Adaptive algorithm selection implementation
│
├── test/                         # Testing directory
│   ├── test_correctness.cpp      # Correctness verification tests
│   ├── test_cpu.cpp              # CPU implementation tests
│   └── test_gpu.cpp              # GPU implementation tests
│
├── benchmark/                    # Benchmarking code
│   ├── benchmark_all.cpp         # Complete benchmarking suite
│   ├── benchmark_cpu.cpp         # CPU implementation benchmarks
│   ├── benchmark_gpu.cpp         # GPU implementation benchmarks
│   └── compare_libraries.cpp     # Comparison with MKL and cuBLAS
│
├── deep_learning/                # Deep learning integration
│   ├── pytorch_extension/        # PyTorch C++ extension
│   │   ├── matmul_extension.cpp  # Extension implementation
│   │   └── setup.py              # Extension build script
│   │
│   ├── models/                   # Neural network models
│   │   └── resnet50.py           # ResNet-50 implementation
│   │
│   ├── train.py                  # Training script
│   ├── benchmark.py              # Deep learning benchmarking script
│   └── visualize.py              # Result visualization script
│
├── scripts/                      # Utility scripts
│   ├── generate_matrices.py      # Script to generate test matrices
│   ├── plot_results.py           # Script to visualize benchmark results
│   ├── roofline_model.py         # Script to generate roofline model
│   └── run_benchmarks.sh         # Script to run all benchmarks
│
├── data/                         # Data directory
│   ├── matrices/                 # Test matrices
│   └── results/                  # Benchmark results
│
├── docs/                         # Documentation
│   ├── project_proposal.md       # Project proposal
│   ├── implementation_details.md # Implementation documentation
│   ├── performance_analysis.md   # Performance analysis
│   └── showcase_presentation.pdf # Showcase presentation slides
│
├── showcase/                     # Project showcase materials
│   ├── demo/                     # Live demonstration code
│   │   ├── interactive_demo.py   # Interactive performance demo
│   │   └── real_time_monitor.py  # Real-time performance monitor
│   │
│   ├── visualizations/           # Pre-generated visualizations
│   │   ├── roofline_plots/       # Roofline model visualizations
│   │   ├── scaling_plots/        # Performance scaling charts
│   │   └── training_plots/       # Neural network training visualizations
│   │
│   ├── analysis/                 # Analysis results
│   │   ├── cost_analysis.xlsx    # Cloud cost savings analysis
│   │   └── energy_data.csv       # Energy efficiency measurements
│   │
│   └── presentation/             # Presentation materials
│       ├── slides.pptx           # Presentation slides
│       └── demo_script.md        # Demonstration script
│
├── CMakeLists.txt                # Main CMake configuration file
├── README.md                     # Project overview and instructions
└── LICENSE                       # Project license
```

## Implementation Details

### CPU Implementations

1. **Naive Implementation**
   - Basic triple-nested loop implementation (O(n³))
   - Used as baseline for performance comparison

2. **Cache-Blocked Implementation**
   - Divides matrices into cache-sized blocks
   - Improves temporal and spatial locality
   - Block size tuned to L1/L2 cache size

3. **SIMD Implementation**
   - Uses AVX-512 vector instructions (16 FP32 elements)
   - Falls back to AVX2/AVX/SSE based on hardware support
   - Includes memory alignment and padding optimizations

4. **Multithreaded Implementation**
   - Uses OpenMP for shared-memory parallelism
   - Dynamic scheduling for load balancing
   - Thread affinity optimizations for NUMA systems

5. **Combined Implementation**
   - Integrates blocking, SIMD, and multithreading
   - Represents the best CPU performance achievable

### GPU Implementations

1. **Basic CUDA Implementation**
   - Tiled shared memory approach
   - Coalesced memory access patterns
   - Register blocking optimizations

2. **cuBLAS Integration**
   - Wrapper for NVIDIA's optimized BLAS library
   - Used for performance comparison

3. **Tensor Core Implementation**
   - Utilizes NVIDIA Tensor Cores on supported hardware
   - Mixed-precision multiplication (FP16 compute with FP32 accumulation)

4. **Multi-GPU Implementation**
   - Distributes work across multiple GPUs
   - Managed data transfers and synchronization

### Adaptive Selection System

The project includes an automatic algorithm selection system that:

1. Profiles the system hardware capabilities
2. Considers matrix dimensions and available memory
3. Selects the optimal implementation based on benchmarking data
4. Handles edge cases (small matrices, irregular shapes)

## Build System

The project uses CMake as its build system with the following features:

- Automatic detection of available instruction sets (AVX-512, AVX2, etc.)
- Optional dependencies (CUDA, OpenBLAS, MKL)
- Cross-platform support (tested on Linux, macOS, Windows)
- Performance-critical compilation flags

## Performance Analysis Tools

The project includes several performance analysis tools:

1. **Roofline Model Generator**
   - Scripts to generate roofline performance models
   - Visualizes compute-bound vs. memory-bound regions

2. **Benchmarking Suite**
   - Comprehensive performance testing across matrix sizes
   - Comparison with industry-standard libraries

3. **Deep Learning Integration**
   - Demonstrates real-world impact on training speed
   - Measures memory, power, and throughput metrics

## Getting Started

To build the project:

```bash
mkdir build
cd build
cmake ..
make
```

To run the benchmarks:

```bash
./bin/benchmark_all
```

To run the tests:

```bash
./bin/test_correctness
```

For detailed documentation, see the docs/ directory.
