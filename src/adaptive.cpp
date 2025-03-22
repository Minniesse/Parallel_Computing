#include "adaptive.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"
#include "gpu/cuda_wrapper.h"

namespace MatrixMult {

// Default thresholds based on common hardware
AdaptiveThresholds get_optimal_thresholds() {
    AdaptiveThresholds thresholds;
    
    // Default values - should be tuned for specific hardware
    thresholds.cpu_gpu_crossover = 1024;      // Switch to GPU for matrices >= 1024x1024
    thresholds.naive_blocked_crossover = 256; // Use blocked algorithm for matrices >= 256x256
    thresholds.block_size_small = 64;         // Block size for matrices < 1024x1024
    thresholds.block_size_large = 128;        // Block size for matrices >= 1024x1024
    
    return thresholds;
}

void adaptive_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    size_t matrix_size = A.rows(); // Assuming square matrices for simplicity
    AdaptiveThresholds thresholds = get_optimal_thresholds();
    
    // Check if GPU is available
    bool gpu_available = GPU::check_cuda_available();
    
    // Check if AVX-512 is available
    bool avx512_available = CPU::is_avx512_supported();
    
    // Select the appropriate algorithm based on matrix size and hardware capabilities
    if (gpu_available && matrix_size >= thresholds.cpu_gpu_crossover) {
        // Use GPU for large matrices
        GPU::cuda_shared_multiply(A, B, C);
    } else if (matrix_size < thresholds.naive_blocked_crossover) {
        // Use naive algorithm for very small matrices
        CPU::naive_multiply(A, B, C);
    } else {
        // Use CPU algorithms with appropriate optimizations
        if (avx512_available) {
            // Use threaded SIMD for best CPU performance
            size_t block_size = (matrix_size >= 1024) ? 
                                thresholds.block_size_large : 
                                thresholds.block_size_small;
            CPU::threaded_simd_multiply(A, B, C, block_size);
        } else {
            // Fall back to threaded blocked multiplication
            size_t block_size = (matrix_size >= 1024) ? 
                                thresholds.block_size_large : 
                                thresholds.block_size_small;
            CPU::threaded_multiply(A, B, C, block_size);
        }
    }
}

} // namespace MatrixMult
