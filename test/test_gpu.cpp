#include <gtest/gtest.h>
#include "common/matrix.h"
#include "cpu/naive.h"
#include "gpu/cuda_wrapper.h"

class GPUMatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip all tests if CUDA is not available
        if (!MatrixMult::GPU::check_cuda_available()) {
            GTEST_SKIP() << "CUDA not available, skipping GPU tests";
        }
        
        // Create test matrices
        small_A = Matrix(128, 128);
        small_B = Matrix(128, 128);
        small_C = Matrix(128, 128, 0.0f);
        small_expected = Matrix(128, 128, 0.0f);
        
        large_A = Matrix(1024, 1024);
        large_B = Matrix(1024, 1024);
        large_C = Matrix(1024, 1024, 0.0f);
        large_expected = Matrix(1024, 1024, 0.0f);
        
        // Fill matrices with random values
        small_A.fill_random(-1.0f, 1.0f);
        small_B.fill_random(-1.0f, 1.0f);
        
        large_A.fill_random(-1.0f, 1.0f);
        large_B.fill_random(-1.0f, 1.0f);
        
        // Compute expected results using CPU naive algorithm
        MatrixMult::CPU::naive_multiply(small_A, small_B, small_expected);
        MatrixMult::CPU::naive_multiply(large_A, large_B, large_expected);
    }
    
    // Small matrices (128x128)
    Matrix small_A, small_B, small_C, small_expected;
    
    // Large matrices (1024x1024)
    Matrix large_A, large_B, large_C, large_expected;
    
    // Tolerance for floating-point comparisons (slightly larger for GPU due to potential floating-point differences)
    const float tolerance = 1e-3f;
};

// Test basic CUDA implementation
TEST_F(GPUMatrixMultiplicationTest, CudaMultiplication) {
    // Test with small matrices
    MatrixMult::GPU::cuda_multiply(small_A, small_B, small_C);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Test with large matrices
    MatrixMult::GPU::cuda_multiply(large_A, large_B, large_C);
    EXPECT_TRUE(large_C.equals(large_expected, tolerance));
}

// Test CUDA shared memory implementation
TEST_F(GPUMatrixMultiplicationTest, CudaSharedMultiplication) {
    // Test with small matrices
    MatrixMult::GPU::cuda_shared_multiply(small_A, small_B, small_C);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Test with large matrices
    MatrixMult::GPU::cuda_shared_multiply(large_A, large_B, large_C);
    EXPECT_TRUE(large_C.equals(large_expected, tolerance));
}

// Test cuBLAS implementation
TEST_F(GPUMatrixMultiplicationTest, CublasMultiplication) {
    // Test with small matrices
    MatrixMult::GPU::cublas_multiply(small_A, small_B, small_C);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Test with large matrices
    MatrixMult::GPU::cublas_multiply(large_A, large_B, large_C);
    EXPECT_TRUE(large_C.equals(large_expected, tolerance));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
