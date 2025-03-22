#pragma once

#include "common/matrix.h"

namespace MatrixMult {
namespace CPU {

// Cache-blocked matrix multiplication
void blocked_multiply(const Matrix& A, const Matrix& B, Matrix& C, 
                      size_t block_size = 64);

// Helper function to multiply blocks
void multiply_block(const Matrix& A, const Matrix& B, Matrix& C,
                   size_t i_start, size_t j_start, size_t p_start,
                   size_t block_size);

// Auto-tune to find optimal block size
size_t find_optimal_block_size(size_t matrix_size);

} // namespace CPU
} // namespace MatrixMult
