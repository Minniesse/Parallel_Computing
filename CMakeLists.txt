cmake_minimum_required(VERSION 3.14)
project(ParallelMatrixMultiplication LANGUAGES C CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(MKL CONFIG)

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mfma")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

# Main executable
add_executable(matrix_mult src/main.cpp)
target_link_libraries(matrix_mult
    matrix_common
    matrix_cpu
    matrix_gpu
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# If MKL is found, link it
if(MKL_FOUND)
    target_link_libraries(matrix_mult ${MKL_LIBRARIES})
    add_definitions(-DUSE_MKL)
endif()

# Install targets
install(TARGETS matrix_mult DESTINATION bin)
