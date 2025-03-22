#include <gtest/gtest.h>
#include "common/matrix.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"
#include "gpu/cuda_wrapper.h"
#include "adaptive.h"

class MatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test matrices
        A = Matrix(matrix_size, matrix_size);
        B = Matrix(matrix_size, matrix_size);
        C_expected = Matrix(matrix_size, matrix_size, 0.0f);
        C_actual = Matrix(matrix_size, matrix_size, 0.0f);
        
        // Fill matrices with random values
        A.fill_random(-1.0f, 1.0f);
        B.fill_random(-1.0f, 1.0f);
        
        // Compute expected result using naive algorithm
        MatrixMult::CPU::naive_multiply(A, B, C_expected);
    }
    
    // Test matrices
    Matrix A, B, C_expected, C_actual;
    
    // Matrix size for testing
    const size_t matrix_size = 128;
    
    // Tolerance for floating-point comparisons
    const float tolerance = 1e-4f;
};

// Test naive implementation (baseline)
TEST_F(MatrixMultiplicationTest, NaiveMultiplication) {
    MatrixMult::CPU::naive_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test blocked implementation
TEST_F(MatrixMultiplicationTest, BlockedMultiplication) {
    MatrixMult::CPU::blocked_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test SIMD implementation
TEST_F(MatrixMultiplicationTest, SimdMultiplication) {
    if (MatrixMult::CPU::is_avx512_supported()) {
        MatrixMult::CPU::simd_multiply(A, B, C_actual);
        EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
    } else {
        GTEST_SKIP() << "AVX-512 not supported on this hardware";
    }
}

// Test threaded implementation
TEST_F(MatrixMultiplicationTest, ThreadedMultiplication) {
    MatrixMult::CPU::threaded_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test threaded SIMD implementation
TEST_F(MatrixMultiplicationTest, ThreadedSimdMultiplication) {
    if (MatrixMult::CPU::is_avx512_supported()) {
        MatrixMult::CPU::threaded_simd_multiply(A, B, C_actual);
        EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
    } else {
        GTEST_SKIP() << "AVX-512 not supported on this hardware";
    }
}

// Test CUDA implementation
TEST_F(MatrixMultiplicationTest, CudaMultiplication) {
    if (!MatrixMult::GPU::check_cuda_available()) {
        GTEST_SKIP() << "CUDA not available on this hardware";
    }
    
    MatrixMult::GPU::cuda_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test CUDA shared memory implementation
TEST_F(MatrixMultiplicationTest, CudaSharedMultiplication) {
    if (!MatrixMult::GPU::check_cuda_available()) {
        GTEST_SKIP() << "CUDA not available on this hardware";
    }
    
    MatrixMult::GPU::cuda_shared_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test cuBLAS implementation
TEST_F(MatrixMultiplicationTest, CublasMultiplication) {
    if (!MatrixMult::GPU::check_cuda_available()) {
        GTEST_SKIP() << "CUDA not available on this hardware";
    }
    
    MatrixMult::GPU::cublas_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test adaptive implementation
TEST_F(MatrixMultiplicationTest, AdaptiveMultiplication) {
    MatrixMult::adaptive_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

// Test with non-square matrices
class NonSquareMatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create non-square test matrices
        A = Matrix(m, k);
        B = Matrix(k, n);
        C_expected = Matrix(m, n, 0.0f);
        C_actual = Matrix(m, n, 0.0f);
        
        // Fill matrices with random values
        A.fill_random(-1.0f, 1.0f);
        B.fill_random(-1.0f, 1.0f);
        
        // Compute expected result using naive algorithm
        MatrixMult::CPU::naive_multiply(A, B, C_expected);
    }
    
    // Test matrices
    Matrix A, B, C_expected, C_actual;
    
    // Matrix dimensions for testing
    const size_t m = 100;
    const size_t k = 150;
    const size_t n = 200;
    
    // Tolerance for floating-point comparisons
    const float tolerance = 1e-4f;
};

// Test all implementations with non-square matrices
TEST_F(NonSquareMatrixMultiplicationTest, AllImplementations) {
    // Test blocked implementation
    MatrixMult::CPU::blocked_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
    
    // Test threaded implementation
    C_actual = Matrix(m, n, 0.0f);
    MatrixMult::CPU::threaded_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
    
    // Test SIMD implementation if available
    if (MatrixMult::CPU::is_avx512_supported()) {
        C_actual = Matrix(m, n, 0.0f);
        MatrixMult::CPU::simd_multiply(A, B, C_actual);
        EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
    }
    
    // Test CUDA implementations if available
    if (MatrixMult::GPU::check_cuda_available()) {
        C_actual = Matrix(m, n, 0.0f);
        MatrixMult::GPU::cuda_multiply(A, B, C_actual);
        EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
        
        C_actual = Matrix(m, n, 0.0f);
        MatrixMult::GPU::cuda_shared_multiply(A, B, C_actual);
        EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
        
        C_actual = Matrix(m, n, 0.0f);
        MatrixMult::GPU::cublas_multiply(A, B, C_actual);
        EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
    }
    
    // Test adaptive implementation
    C_actual = Matrix(m, n, 0.0f);
    MatrixMult::adaptive_multiply(A, B, C_actual);
    EXPECT_TRUE(C_actual.equals(C_expected, tolerance));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
