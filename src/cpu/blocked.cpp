#include "cpu/blocked.h"
#include "common/timing.h"
#include <algorithm>

namespace MatrixMult {
namespace CPU {

void multiply_block(const Matrix& A, const Matrix& B, Matrix& C,
                  size_t i_start, size_t j_start, size_t p_start,
                  size_t block_size) {
    size_t i_end = std::min(i_start + block_size, A.rows());
    size_t j_end = std::min(j_start + block_size, B.cols());
    size_t p_end = std::min(p_start + block_size, A.cols());
    
    for (size_t i = i_start; i < i_end; ++i) {
        for (size_t j = j_start; j < j_end; ++j) {
            float sum = 0.0f;
            for (size_t p = p_start; p < p_end; ++p) {
                sum += A.at(i, p) * B.at(p, j);
            }
            C.at(i, j) += sum;
        }
    }
}

void blocked_multiply(const Matrix& A, const Matrix& B, Matrix& C, size_t block_size) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    // Initialize C to zeros
    for (size_t i = 0; i < C.rows(); ++i) {
        for (size_t j = 0; j < C.cols(); ++j) {
            C.at(i, j) = 0.0f;
        }
    }
    
    // Use auto-tuned block size if not specified
    if (block_size == 0) {
        block_size = find_optimal_block_size(A.rows());
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Blocked matrix multiplication
    for (size_t i = 0; i < m; i += block_size) {
        for (size_t j = 0; j < n; j += block_size) {
            for (size_t p = 0; p < k; p += block_size) {
                multiply_block(A, B, C, i, j, p, block_size);
            }
        }
    }
}

size_t find_optimal_block_size(size_t matrix_size) {
    // Simple heuristic: use block sizes based on matrix size
    if (matrix_size <= 256) {
        return 32;
    } else if (matrix_size <= 512) {
        return 64;
    } else if (matrix_size <= 1024) {
        return 128;
    } else {
        return 256;
    }
    
    // For a more sophisticated approach, auto-tuning would run
    // benchmarks with different block sizes and select the best one
}

} // namespace CPU
} // namespace MatrixMult
