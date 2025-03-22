#pragma once

#include "common/matrix.h"
#include <omp.h>

namespace MatrixMult {
namespace CPU {

// OpenMP multithreaded matrix multiplication
void threaded_multiply(const Matrix& A, const Matrix& B, Matrix& C, 
                       size_t block_size = 64);

// OpenMP + SIMD combined implementation
void threaded_simd_multiply(const Matrix& A, const Matrix& B, Matrix& C, 
                           size_t block_size = 64);

// Get optimal number of threads based on matrix size
int get_optimal_thread_count(size_t matrix_size);

} // namespace CPU
} // namespace MatrixMult
