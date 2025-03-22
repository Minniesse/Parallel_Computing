#include "cpu/blocked.h"
#include "common/utils.h"

namespace cpu {
namespace blocked {

// Default block size
constexpr int DEFAULT_BLOCK_SIZE = 64;

void multiply(const float* A, const float* B, float* C, int m, int n, int k) {
    // Get optimal block size
    int block_size = common::getOptimalBlockSize(32 * 1024); // Assuming 32KB L1 cache
    if (block_size <= 0) {
        block_size = DEFAULT_BLOCK_SIZE;
    }
    
    // Initialize C to zeros
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }
    
    // Blocked matrix multiplication
    for (int i = 0; i < m; i += block_size) {
        int i_end = std::min(i + block_size, m);
        
        for (int j = 0; j < n; j += block_size) {
            int j_end = std::min(j + block_size, n);
            
            for (int p = 0; p < k; p += block_size) {
                int p_end = std::min(p + block_size, k);
                
                // Compute block
                for (int ii = i; ii < i_end; ii++) {
                    for (int jj = j; jj < j_end; jj++) {
                        float sum = C[ii * n + jj];
                        
                        for (int pp = p; pp < p_end; pp++) {
                            sum += A[ii * k + pp] * B[pp * n + jj];
                        }
                        
                        C[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

} // namespace blocked
} // namespace cpu
