#include "cpu/simd.h"
#include "common/utils.h"
#include "cpu/naive.h" // Add this include for cpu::naive

#ifdef HAS_AVX512
#include <immintrin.h>
#endif

#ifdef HAS_AVX2
#include <immintrin.h>
#endif

#ifdef HAS_AVX
#include <immintrin.h>
#endif

#ifdef HAS_SSE
#include <xmmintrin.h>
#endif

namespace cpu {
namespace simd {

// Determine if various SIMD instruction sets are available
bool is_avx512_supported() {
    #ifdef HAS_AVX512
    return true;
    #else
    return false;
    #endif
}

bool is_avx2_supported() {
    #ifdef HAS_AVX2
    return true;
    #else
    return false;
    #endif
}

bool is_avx_supported() {
    #ifdef HAS_AVX
    return true;
    #else
    return false;
    #endif
}

bool is_sse_supported() {
    #ifdef HAS_SSE
    return true;
    #else
    return false;
    #endif
}

// Determine optimal block size
int get_optimal_block_size() {
    if (is_avx512_supported()) {
        return 32; // Optimal for AVX-512
    } else if (is_avx2_supported() || is_avx_supported()) {
        return 32; // Optimal for AVX/AVX2
    } else if (is_sse_supported()) {
        return 16; // Optimal for SSE
    } else {
        return 64; // Fallback
    }
}

#ifdef HAS_AVX512
// AVX-512 implementation
void multiply_avx512(const float* A, const float* B, float* C, int m, int n, int k) {
    // Zero out C matrix
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0f;
    }
    
    // Process 16 elements at once using AVX-512
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 16) {
            // Handle boundary condition
            if (j + 16 > n) {
                // Process remaining elements using scalar code
                for (int jj = j; jj < n; jj++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += A[i * k + p] * B[p * n + jj];
                    }
                    C[i * n + jj] = sum;
                }
                break;
            }
            
            __m512 c_vec = _mm512_setzero_ps();
            
            for (int p = 0; p < k; p++) {
                __m512 a_vec = _mm512_set1_ps(A[i * k + p]);
                __m512 b_vec = _mm512_loadu_ps(&B[p * n + j]);
                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
            }
            
            _mm512_storeu_ps(&C[i * n + j], c_vec);
        }
    }
}
#else
// Fallback implementation if AVX-512 is not available
void multiply_avx512(const float* A, const float* B, float* C, int m, int n, int k) {
    // Call the best available implementation
    multiply(A, B, C, m, n, k);
}
#endif

#ifdef HAS_AVX2
// AVX2 implementation
void multiply_avx2(const float* A, const float* B, float* C, int m, int n, int k) {
    // Zero out C matrix
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0f;
    }
    
    // Process 8 elements at once using AVX2
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 8) {
            // Handle boundary condition
            if (j + 8 > n) {
                // Process remaining elements using scalar code
                for (int jj = j; jj < n; jj++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += A[i * k + p] * B[p * n + jj];
                    }
                    C[i * n + jj] = sum;
                }
                break;
            }
            
            __m256 c_vec = _mm256_setzero_ps();
            
            for (int p = 0; p < k; p++) {
                __m256 a_vec = _mm256_set1_ps(A[i * k + p]);
                __m256 b_vec = _mm256_loadu_ps(&B[p * n + j]);
                c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
            }
            
            _mm256_storeu_ps(&C[i * n + j], c_vec);
        }
    }
}
#else
// Fallback implementation if AVX2 is not available
void multiply_avx2(const float* A, const float* B, float* C, int m, int n, int k) {
    // Call the best available implementation
    multiply(A, B, C, m, n, k);
}
#endif

#ifdef HAS_AVX
// AVX implementation
void multiply_avx(const float* A, const float* B, float* C, int m, int n, int k) {
    // Zero out C matrix
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0f;
    }
    
    // Process 8 elements at once using AVX
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 8) {
            // Handle boundary condition
            if (j + 8 > n) {
                // Process remaining elements using scalar code
                for (int jj = j; jj < n; jj++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += A[i * k + p] * B[p * n + jj];
                    }
                    C[i * n + jj] = sum;
                }
                break;
            }
            
            __m256 c_vec = _mm256_setzero_ps();
            
            for (int p = 0; p < k; p++) {
                __m256 a_vec = _mm256_set1_ps(A[i * k + p]);
                __m256 b_vec = _mm256_loadu_ps(&B[p * n + j]);
                c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
            }
            
            _mm256_storeu_ps(&C[i * n + j], c_vec);
        }
    }
}
#else
// Fallback implementation if AVX is not available
void multiply_avx(const float* A, const float* B, float* C, int m, int n, int k) {
    // Call the best available implementation
    multiply(A, B, C, m, n, k);
}
#endif

#ifdef HAS_SSE
// SSE implementation
void multiply_sse(const float* A, const float* B, float* C, int m, int n, int k) {
    // Zero out C matrix
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0f;
    }
    
    // Process 4 elements at once using SSE
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 4) {
            // Handle boundary condition
            if (j + 4 > n) {
                // Process remaining elements using scalar code
                for (int jj = j; jj < n; jj++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += A[i * k + p] * B[p * n + jj];
                    }
                    C[i * n + jj] = sum;
                }
                break;
            }
            
            __m128 c_vec = _mm_setzero_ps();
            
            for (int p = 0; p < k; p++) {
                __m128 a_vec = _mm_set1_ps(A[i * k + p]);
                __m128 b_vec = _mm_loadu_ps(&B[p * n + j]);
                c_vec = _mm_add_ps(c_vec, _mm_mul_ps(a_vec, b_vec));
            }
            
            _mm_storeu_ps(&C[i * n + j], c_vec);
        }
    }
}
#else
// Fallback implementation if SSE is not available
void multiply_sse(const float* A, const float* B, float* C, int m, int n, int k) {
    // Use naive implementation as fallback
    cpu::naive::multiply(A, B, C, m, n, k);
}
#endif

// Generic SIMD implementation that selects the best available instruction set
void multiply(const float* A, const float* B, float* C, int m, int n, int k) {
    if (is_avx512_supported()) {
        multiply_avx512(A, B, C, m, n, k);
    } else if (is_avx2_supported()) {
        multiply_avx2(A, B, C, m, n, k);
    } else if (is_avx_supported()) {
        multiply_avx(A, B, C, m, n, k);
    } else if (is_sse_supported()) {
        multiply_sse(A, B, C, m, n, k);
    } else {
        // Fallback to naive implementation
        cpu::naive::multiply(A, B, C, m, n, k);
    }
}

void multiply_blocked_avx512(const float* A, const float* B, float* C, int m, int n, int k) {
    // Get optimal block size
    int blockSize = get_optimal_block_size();
    
    // Initialize C to zeros
    for (int i = 0; i < m * n; i++) {
        C[i] = 0.0f;
    }
    
    // Blocked matrix multiplication with AVX-512
    for (int i = 0; i < m; i += blockSize) {
        int iLimit = std::min(i + blockSize, m);
        
        for (int j = 0; j < n; j += blockSize) {
            int jLimit = std::min(j + blockSize, n);
            
            for (int p = 0; p < k; p += blockSize) {
                int pLimit = std::min(p + blockSize, k);
                
                // Compute block using AVX-512
                if (is_avx512_supported()) {
                    #ifdef HAS_AVX512
                    for (int ii = i; ii < iLimit; ii++) {
                        for (int jj = j; jj < jLimit; jj += 16) {
                            if (jj + 16 <= jLimit) {
                                __m512 c_vec = _mm512_loadu_ps(&C[ii * n + jj]);
                                
                                for (int pp = p; pp < pLimit; pp++) {
                                    __m512 a_vec = _mm512_set1_ps(A[ii * k + pp]);
                                    __m512 b_vec = _mm512_loadu_ps(&B[pp * n + jj]);
                                    c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                }
                                
                                _mm512_storeu_ps(&C[ii * n + jj], c_vec);
                            } else {
                                // Handle boundary with scalar code
                                for (int j_rem = jj; j_rem < jLimit; j_rem++) {
                                    float sum = C[ii * n + j_rem];
                                    for (int pp = p; pp < pLimit; pp++) {
                                        sum += A[ii * k + pp] * B[pp * n + j_rem];
                                    }
                                    C[ii * n + j_rem] = sum;
                                }
                            }
                        }
                    }
                    #endif
                } else {
                    // Fallback to regular blocked implementation
                    for (int ii = i; ii < iLimit; ii++) {
                        for (int jj = j; jj < jLimit; jj++) {
                            float sum = C[ii * n + jj];
                            for (int pp = p; pp < pLimit; pp++) {
                                sum += A[ii * k + pp] * B[pp * n + jj];
                            }
                            C[ii * n + jj] = sum;
                        }
                    }
                }
            }
        }
    }
}

} // namespace simd
} // namespace cpu
