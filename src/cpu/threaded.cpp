#include "cpu/threaded.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include <algorithm>
#include <cstring>

namespace MatrixMult {
namespace CPU {

int get_optimal_thread_count(size_t matrix_size) {
    int max_threads = omp_get_max_threads();
    
    // For small matrices, limit thread count to avoid overhead
    if (matrix_size < 512) {
        return std::min(max_threads, 4);
    } else if (matrix_size < 1024) {
        return std::min(max_threads, 8);
    } else {
        return max_threads;
    }
}

void threaded_multiply(const Matrix& A, const Matrix& B, Matrix& C, size_t block_size) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Clear C matrix
    std::memset(C.data(), 0, C.size() * sizeof(float));
    
    // Use auto-tuned block size if not specified
    if (block_size == 0) {
        block_size = find_optimal_block_size(m);
    }
    
    // Set optimal thread count
    int num_threads = get_optimal_thread_count(m);
    
    // Parallel blocked matrix multiplication
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < m; i += block_size) {
        for (size_t j = 0; j < n; j += block_size) {
            for (size_t p = 0; p < k; p += block_size) {
                multiply_block(A, B, C, i, j, p, block_size);
            }
        }
    }
}

void threaded_simd_multiply(const Matrix& A, const Matrix& B, Matrix& C, size_t block_size) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    if (!is_avx512_supported()) {
        // Fall back to regular threaded implementation if AVX-512 is not supported
        threaded_multiply(A, B, C, block_size);
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Clear C matrix
    std::memset(C.data(), 0, C.size() * sizeof(float));
    
    // Use auto-tuned block size if not specified
    if (block_size == 0) {
        block_size = find_optimal_block_size(m);
        // Make block size multiple of 16 for AVX-512
        block_size = (block_size / 16) * 16;
        if (block_size == 0) block_size = 16;
    }
    
    // Set optimal thread count
    int num_threads = get_optimal_thread_count(m);
    
    // Parallel SIMD matrix multiplication
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < m; i += block_size) {
        size_t i_end = std::min(i + block_size, m);
        size_t i_size = i_end - i;
        
        for (size_t j = 0; j < n; j += block_size) {
            size_t j_end = std::min(j + block_size, n);
            size_t j_size = j_end - j;
            
            float* C_block = C.data() + i * n + j;
            
            for (size_t p = 0; p < k; p += block_size) {
                size_t p_end = std::min(p + block_size, k);
                size_t p_size = p_end - p;
                
                const float* A_block = A.data() + i * k + p;
                const float* B_block = B.data() + p * n + j;
                
                // Process the block using AVX-512
                multiply_block_avx512(A_block, B_block, C_block, 
                                    i_size, k, n, n);
            }
        }
    }
}

} // namespace CPU
} // namespace MatrixMult
