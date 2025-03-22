#include "cpu/threaded.h"
#include "cpu/simd.h"
#include <omp.h>
#include <algorithm> // Add this include for std::min

namespace cpu {
namespace threaded {

void multiply(const float* A, const float* B, float* C, int m, int n, int k) {
    // Zero out C matrix
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }
    
    // Parallelize the outermost loop
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiply_simd(const float* A, const float* B, float* C, int m, int n, int k) {
    const int BLOCK_SIZE = 64;
    
    // Zero out C matrix
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }
    
    // Blocked matrix multiplication with SIMD and OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        int iLimit = std::min(i + BLOCK_SIZE, m);
        
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            int jLimit = std::min(j + BLOCK_SIZE, n);
            
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                int pLimit = std::min(p + BLOCK_SIZE, k);
                
                // Process block using SIMD
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

} // namespace threaded
} // namespace cpu
