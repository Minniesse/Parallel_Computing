#pragma once

#include "common/matrix.h"
#include <immintrin.h>  // For AVX-512 intrinsics

namespace MatrixMult {
namespace CPU {

// SIMD optimized matrix multiplication using AVX-512
void simd_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// AVX-512 kernel for block multiplication
void multiply_block_avx512(const float* A, const float* B, float* C, 
                          int block_size, int lda, int ldb, int ldc);

// Check if AVX-512 is supported on the current CPU
bool is_avx512_supported();

} // namespace CPU
} // namespace MatrixMult
