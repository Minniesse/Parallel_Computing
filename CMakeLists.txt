cmake_minimum_required(VERSION 3.10)
project(ParallelMatrixMultiplication LANGUAGES CXX C)

# Set output directory - must be set before any targets are created
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Check for SIMD instruction sets
include(CheckCXXSourceCompiles)

# Check for AVX-512
check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m512 a = _mm512_set1_ps(1.0f);
        __m512 b = _mm512_set1_ps(2.0f);
        __m512 c = _mm512_add_ps(a, b);
        return 0;
    }
" HAS_AVX512)

# Check for AVX2
check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m256 a = _mm256_set1_ps(1.0f);
        __m256 b = _mm256_set1_ps(2.0f);
        __m256 c = _mm256_add_ps(a, b);
        return 0;
    }
" HAS_AVX2)

# Check for AVX
check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m256 a = _mm256_set1_ps(1.0f);
        __m256 b = _mm256_set1_ps(2.0f);
        __m256 c = _mm256_add_ps(a, b);
        return 0;
    }
" HAS_AVX)

# Check for SSE
check_cxx_source_compiles("
    #include <xmmintrin.h>
    int main() {
        __m128 a = _mm_set1_ps(1.0f);
        __m128 b = _mm_set1_ps(2.0f);
        __m128 c = _mm_add_ps(a, b);
        return 0;
    }
" HAS_SSE)

# Configure SIMD instruction set flags
if(HAS_AVX512)
    add_definitions(-DHAS_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
endif()
if(HAS_AVX2)
    add_definitions(-DHAS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()
if(HAS_AVX)
    add_definitions(-DHAS_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()
if(HAS_SSE)
    add_definitions(-DHAS_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

# Try to find CUDA
option(USE_CUDA "Build with CUDA support" ON)
if(USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        add_definitions(-DHAS_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        # Set CUDA compilation flags
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_60")
        message(STATUS "CUDA found. Building with GPU support.")
    else()
        message(STATUS "CUDA not found. Building without GPU support.")
    endif()
endif()

# Try to find OpenBLAS
find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    add_definitions(-DHAS_OPENBLAS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    message(STATUS "OpenBLAS found.")
else()
    message(STATUS "OpenBLAS not found. Will not compare against it.")
endif()

# Try to find MKL
find_package(MKL QUIET)
if(MKL_FOUND)
    add_definitions(-DHAS_MKL)
    include_directories(${MKL_INCLUDE_DIR})
    message(STATUS "Intel MKL found.")
else()
    message(STATUS "Intel MKL not found. Will not compare against it.")
endif()

# Include directories
include_directories(include)

# Source files
set(COMMON_SOURCES
    src/common/matrix.cpp
    src/common/timing.cpp
    src/common/utils.cpp
)

set(CPU_SOURCES
    src/cpu/naive.cpp
    src/cpu/blocked.cpp
    src/cpu/simd.cpp
    src/cpu/threaded.cpp
)

set(GPU_SOURCES
    src/gpu/cuda_wrapper.cpp
)

if(CUDA_FOUND)
    # CUDA kernel files
    set(CUDA_KERNELS
        src/gpu/matmul_kernel.cu
        src/gpu/multi_gpu.cpp
    )
    cuda_add_library(cuda_kernels ${CUDA_KERNELS})
endif()

# Create library targets
add_library(common ${COMMON_SOURCES})
add_library(cpu_implementations ${CPU_SOURCES})

# Link libraries
if(CUDA_FOUND)
    add_library(gpu_implementations ${GPU_SOURCES})
    target_link_libraries(gpu_implementations common cuda_kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

target_link_libraries(cpu_implementations common)

# Benchmark executables
add_executable(benchmark_cpu benchmark/benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu common cpu_implementations)

if(CUDA_FOUND)
    add_executable(benchmark_gpu benchmark/benchmark_gpu.cpp)
    target_link_libraries(benchmark_gpu common gpu_implementations)
    
    add_executable(benchmark_all benchmark/benchmark_all.cpp)
    target_link_libraries(benchmark_all common cpu_implementations gpu_implementations)
    
    add_executable(compare_libraries benchmark/compare_libraries.cpp)
    target_link_libraries(compare_libraries common cpu_implementations gpu_implementations)
    if(OpenBLAS_FOUND)
        target_link_libraries(compare_libraries ${OpenBLAS_LIBRARIES})
    endif()
    if(MKL_FOUND)
        target_link_libraries(compare_libraries ${MKL_LIBRARIES})
    endif()
else()
    add_executable(benchmark_all benchmark/benchmark_all.cpp)
    target_link_libraries(benchmark_all common cpu_implementations)
    
    add_executable(compare_libraries benchmark/compare_libraries.cpp)
    target_link_libraries(compare_libraries common cpu_implementations)
    if(OpenBLAS_FOUND)
        target_link_libraries(compare_libraries ${OpenBLAS_LIBRARIES})
    endif()
    if(MKL_FOUND)
        target_link_libraries(compare_libraries ${MKL_LIBRARIES})
    endif()
endif()

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
