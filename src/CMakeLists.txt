# Common library
add_library(matrix_common
    common/matrix.cpp
    common/timing.cpp
)

# CPU library
add_library(matrix_cpu
    cpu/naive.cpp
    cpu/blocked.cpp
    cpu/simd.cpp
    cpu/threaded.cpp
)
target_link_libraries(matrix_cpu
    matrix_common
    OpenMP::OpenMP_CXX
)

# GPU library
cuda_add_library(matrix_gpu
    gpu/cuda_wrapper.cu
    adaptive.cpp
)
target_link_libraries(matrix_gpu
    matrix_common
    matrix_cpu
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)
