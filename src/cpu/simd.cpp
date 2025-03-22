#include "cpu/simd.h"
#include "cpu/blocked.h" // Add this include for blocked_multiply
#include <cstring>  // For memset

namespace MatrixMult {
namespace CPU {

bool is_avx512_supported() {
#ifdef __AVX512F__
    return true;
#else
    return false;
#endif
}

void multiply_block_avx512(const float* A, const float* B, float* C, 
                         int block_size, int lda, int ldb, int ldc) {
#ifdef __AVX512F__
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j += 16) {
            // Load 16 zeros into the vector register
            __m512 c_vec = _mm512_setzero_ps();
            
            for (int k = 0; k < block_size; k++) {
                // Broadcast single A element to all elements of a vector
                __m512 a_vec = _mm512_set1_ps(A[i*lda + k]);
                
                // Load 16 elements from B
                __m512 b_vec = _mm512_loadu_ps(&B[k*ldb + j]);
                
                // Multiply and accumulate
                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
            }
            
            // Store the result back to C
            _mm512_storeu_ps(&C[i*ldc + j], c_vec);
        }
    }
#endif
}

void simd_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    if (!is_avx512_supported()) {
        // Fall back to blocked implementation if AVX-512 is not supported
        blocked_multiply(A, B, C);
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Clear C matrix
    std::memset(C.data(), 0, C.size() * sizeof(float));
    
    // Choose block size that's a multiple of 16 for AVX-512
    size_t block_size = 64;
    
    // Block sizes for each dimension
    for (size_t i = 0; i < m; i += block_size) {
        size_t i_end = std::min(i + block_size, m);
        size_t i_size = i_end - i;
        
        for (size_t j = 0; j < n; j += block_size) {
            size_t j_end = std::min(j + block_size, n);
            size_t j_size = j_end - j;
            
            // If the block is not a multiple of 16, pad it
            size_t j_padded = ((j_size + 15) / 16) * 16;
            
            // Create padded temporary storage if needed
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
