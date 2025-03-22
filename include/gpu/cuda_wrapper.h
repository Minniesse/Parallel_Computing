#pragma once

#include "common/matrix.h"
#include <cuda_runtime.h>

// Forward declare the wrapper function for direct kernel access
extern "C" void runMatrixMulSharedKernel(const float* A, const float* B, float* C, 
                                        int m, int n, int k, dim3 gridDim, dim3 blockDim);

namespace MatrixMult {
namespace GPU {

// CUDA matrix multiplication
void cuda_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// CUDA matrix multiplication with shared memory
void cuda_shared_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// cuBLAS matrix multiplication (for reference)
void cublas_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Check if CUDA is available and get device properties
bool check_cuda_available();
void print_cuda_device_info();

} // namespace GPU
} // namespace MatrixMult
