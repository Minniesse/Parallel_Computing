#include <string>
#include <iostream>
#include "common/utils.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"

#ifdef HAS_CUDA
#include "gpu/cuda_wrapper.h"
#include "gpu/multi_gpu.h"
#endif

// Adaptive algorithm selection based on matrix size and hardware capabilities
class MatrixMultiplierSelector {
private:
    // Hardware information
    common::HardwareInfo hw_info_;
    bool cuda_available_;
    int num_cuda_devices_;
    
    // Algorithm selection thresholds (based on matrix size)
    const int NAIVE_THRESHOLD = 64;
    const int SIMD_THRESHOLD = 128;
    const int THREADED_THRESHOLD = 256;
    const int GPU_THRESHOLD = 1024;
    
public:
    MatrixMultiplierSelector() {
        // Get hardware information
        hw_info_ = common::getHardwareInfo();
        
        // Check CUDA availability
        #ifdef HAS_CUDA
        cuda_available_ = gpu::isCudaAvailable();
        num_cuda_devices_ = gpu::MultiGpuMatrixMultiplier::getGpuCount();
        #else
        cuda_available_ = false;
        num_cuda_devices_ = 0;
        #endif
    }
    
    // Select the best algorithm based on matrix size and hardware
    std::string selectAlgorithm(int m, int n, int k) {
        // For small matrices, use naive implementation
        if (m <= NAIVE_THRESHOLD && n <= NAIVE_THRESHOLD && k <= NAIVE_THRESHOLD) {
            return "naive";
        }
        
        // For medium matrices with SIMD but without many cores, use SIMD
        if (m <= SIMD_THRESHOLD && n <= SIMD_THRESHOLD && k <= SIMD_THRESHOLD && 
            (hw_info_.avx_supported || hw_info_.avx2_supported || hw_info_.avx512_supported) &&
            hw_info_.num_physical_cpus <= 4) {
            return "simd";
        }
        
        // For larger matrices with multiple CPU cores, use threaded implementation
        if (m <= THREADED_THRESHOLD && n <= THREADED_THRESHOLD && k <= THREADED_THRESHOLD &&
            hw_info_.num_physical_cpus > 4) {
            return "threaded";
        }
        
        // For matrices larger than GPU threshold, use GPU if available
        if (m >= GPU_THRESHOLD && n >= GPU_THRESHOLD && k >= GPU_THRESHOLD && cuda_available_) {
            // If multiple GPUs are available, use multi-GPU implementation
            if (num_cuda_devices_ > 1 && m >= 2 * GPU_THRESHOLD) {
                return "multi_gpu";
            }
            
            // Otherwise use cuBLAS for best single-GPU performance
            return "cublas";
        }
        
        // For everything else, use the combined CPU implementation (SIMD + threaded)
        return "combined";
    }
    
    // Execute matrix multiplication with the best algorithm
    void multiply(const float* A, const float* B, float* C, int m, int n, int k) {
        std::string algorithm = selectAlgorithm(m, n, k);
        
        std::cout << "Adaptive algorithm selection: using " << algorithm << std::endl;
        
        if (algorithm == "naive") {
            cpu::naive::multiply(A, B, C, m, n, k);
        }
        else if (algorithm == "simd") {
            cpu::simd::multiply(A, B, C, m, n, k);
        }
        else if (algorithm == "threaded") {
            cpu::threaded::multiply(A, B, C, m, n, k);
        }
        else if (algorithm == "combined") {
            cpu::threaded::multiply_simd(A, B, C, m, n, k);
        }
        #ifdef HAS_CUDA
        else if (algorithm == "cublas") {
            gpu::CudaMatrixMultiplier multiplier;
            multiplier.initialize(m, n, k);
            multiplier.setMatrixA(A);
            multiplier.setMatrixB(B);
            multiplier.multiplyWithCuBLAS(C);
            multiplier.cleanup();
        }
        else if (algorithm == "multi_gpu") {
            gpu::MultiGpuMatrixMultiplier multiplier;
            multiplier.initialize(m, n, k);
            multiplier.setMatrices(A, B);
            multiplier.multiplyWithCuBLAS(C);
            multiplier.cleanup();
        }
        #endif
        else {
            // Fallback to combined CPU implementation
            cpu::threaded::multiply_simd(A, B, C, m, n, k);
        }
    }
};

// Singleton accessor for the matrix multiplier selector
MatrixMultiplierSelector& getMatrixMultiplierSelector() {
    static MatrixMultiplierSelector selector;
    return selector;
}

// Global function to perform adaptive matrix multiplication
void adaptiveMatrixMultiply(const float* A, const float* B, float* C, int m, int n, int k) {
    getMatrixMultiplierSelector().multiply(A, B, C, m, n, k);
}
