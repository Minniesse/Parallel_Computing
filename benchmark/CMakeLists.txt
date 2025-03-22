# Benchmarking executable
add_executable(benchmark_all benchmark_all.cpp)
target_link_libraries(benchmark_all
    matrix_common
    matrix_cpu
    matrix_gpu
)

# CPU benchmark executable
add_executable(benchmark_cpu benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu
    matrix_common
    matrix_cpu
)

# GPU benchmark executable
add_executable(benchmark_gpu benchmark_gpu.cpp)
target_link_libraries(benchmark_gpu
    matrix_common
    matrix_gpu
)

# Library comparison executable
add_executable(compare_libraries compare_libraries.cpp)
target_link_libraries(compare_libraries
    matrix_common
    matrix_cpu
    matrix_gpu
)

# If MKL is found, link it to the comparison executable
if(MKL_FOUND)
    target_link_libraries(compare_libraries ${MKL_LIBRARIES})
    target_compile_definitions(compare_libraries PRIVATE USE_MKL)
endif()
