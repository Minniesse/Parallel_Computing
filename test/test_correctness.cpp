#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "common/matrix.h"
#include "common/timing.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"

#ifdef HAS_CUDA
#include "gpu/cuda_wrapper.h"
#endif

bool testAlgorithm(const std::string& name, 
                  void (*algorithm)(const float*, const float*, float*, int, int, int),
                  const common::Matrix<float>& A,
                  const common::Matrix<float>& B,
                  const common::Matrix<float>& expected,
                  float epsilon = 1e-4f) {
    int m = A.rows();
    int n = B.cols();
    int k = A.cols();
    
    common::Matrix<float> C(m, n);
    C.fill(0.0f);
    
    // Run the algorithm
    algorithm(A.data(), B.data(), C.data(), m, n, k);
    
    // Check result
    bool correct = C.equals(expected, epsilon);
    
    // Print result
    std::cout << std::left << std::setw(20) << name << ": "
              << (correct ? "PASSED" : "FAILED") << std::endl;
    
    if (!correct) {
        // Find the maximum difference
        float max_diff = 0.0f;
        int max_i = 0, max_j = 0;
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float diff = std::abs(C(i, j) - expected(i, j));
                if (diff > max_diff) {
                    max_diff = diff;
                    max_i = i;
                    max_j = j;
                }
            }
        }
        
        std::cout << "  Maximum difference: " << max_diff
                  << " at position [" << max_i << "," << max_j << "]" << std::endl;
        std::cout << "  Expected: " << expected(max_i, max_j)
                  << ", Got: " << C(max_i, max_j) << std::endl;
    }
    
    return correct;
}

int main() {
    std::cout << "Running correctness tests..." << std::endl;
    
    // Create test matrices
    const int size = 64;  // Small size for quick testing
    common::Matrix<float> A(size, size);
    common::Matrix<float> B(size, size);
    common::Matrix<float> expected(size, size);
    
    // Fill with random values
    A.randomize();
    B.randomize();
    
    // Compute reference result
    cpu::naive::multiply(A.data(), B.data(), expected.data(), size, size, size);
    
    // Test all algorithms
    bool all_passed = true;
    
    // CPU implementations
    all_passed &= testAlgorithm("Naive", cpu::naive::multiply, A, B, expected);
    all_passed &= testAlgorithm("Blocked", cpu::blocked::multiply, A, B, expected);
    all_passed &= testAlgorithm("SIMD", cpu::simd::multiply, A, B, expected);
    all_passed &= testAlgorithm("Threaded", cpu::threaded::multiply, A, B, expected);
    all_passed &= testAlgorithm("Combined", cpu::threaded::multiply_simd, A, B, expected);
    
    // GPU implementations (if available)
#ifdef HAS_CUDA
    if (gpu::isCudaAvailable()) {
        common::Matrix<float> C_cuda(size, size);
        gpu::CudaMatrixMultiplier multiplier;
        
        // Custom CUDA implementation
        multiplier.initialize(size, size, size);
        multiplier.setMatrixA(A.data());
        multiplier.setMatrixB(B.data());
        multiplier.multiply(C_cuda.data());
        multiplier.cleanup();
        
        bool cuda_correct = C_cuda.equals(expected, 1e-4f);
        std::cout << std::left << std::setw(20) << "CUDA" << ": "
                  << (cuda_correct ? "PASSED" : "FAILED") << std::endl;
        all_passed &= cuda_correct;
        
        // cuBLAS
        multiplier.initialize(size, size, size);
        multiplier.setMatrixA(A.data());
        multiplier.setMatrixB(B.data());
        multiplier.multiplyWithCuBLAS(C_cuda.data());
        multiplier.cleanup();
        
        bool cublas_correct = C_cuda.equals(expected, 1e-4f);
        std::cout << std::left << std::setw(20) << "cuBLAS" << ": "
                  << (cublas_correct ? "PASSED" : "FAILED") << std::endl;
        all_passed &= cublas_correct;
    } else {
        std::cout << "CUDA not available, skipping GPU tests." << std::endl;
    }
#else
    std::cout << "CUDA support not compiled, skipping GPU tests." << std::endl;
#endif
    
    // Summary
    std::cout << std::endl << "All tests " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
}
