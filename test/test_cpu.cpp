#include <gtest/gtest.h>
#include "common/matrix.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"

class CPUMatrixMultiplicationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test matrices of different sizes
        small_A = Matrix(128, 128);
        small_B = Matrix(128, 128);
        small_C = Matrix(128, 128, 0.0f);
        small_expected = Matrix(128, 128, 0.0f);
        
        medium_A = Matrix(512, 512);
        medium_B = Matrix(512, 512);
        medium_C = Matrix(512, 512, 0.0f);
        medium_expected = Matrix(512, 512, 0.0f);
        
        // Fill matrices with random values
        small_A.fill_random(-1.0f, 1.0f);
        small_B.fill_random(-1.0f, 1.0f);
        
        medium_A.fill_random(-1.0f, 1.0f);
        medium_B.fill_random(-1.0f, 1.0f);
        
        // Compute expected results using naive algorithm
        MatrixMult::CPU::naive_multiply(small_A, small_B, small_expected);
        MatrixMult::CPU::naive_multiply(medium_A, medium_B, medium_expected);
    }
    
    // Small matrices (128x128)
    Matrix small_A, small_B, small_C, small_expected;
    
    // Medium matrices (512x512)
    Matrix medium_A, medium_B, medium_C, medium_expected;
    
    // Tolerance for floating-point comparisons
    const float tolerance = 1e-4f;
};

// Test blocked implementation with different block sizes
TEST_F(CPUMatrixMultiplicationTest, BlockedMultiplicationWithDifferentSizes) {
    // Test with small matrices
    MatrixMult::CPU::blocked_multiply(small_A, small_B, small_C, 16);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Reset result matrix
    small_C = Matrix(128, 128, 0.0f);
    
    // Test with different block size
    MatrixMult::CPU::blocked_multiply(small_A, small_B, small_C, 32);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Test with medium matrices
    MatrixMult::CPU::blocked_multiply(medium_A, medium_B, medium_C, 64);
    EXPECT_TRUE(medium_C.equals(medium_expected, tolerance));
}

// Test auto-tuned block size
TEST_F(CPUMatrixMultiplicationTest, AutoTunedBlockSize) {
    // Test with small matrices
    MatrixMult::CPU::blocked_multiply(small_A, small_B, small_C, 0); // 0 means auto-tune
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Test with medium matrices
    MatrixMult::CPU::blocked_multiply(medium_A, medium_B, medium_C, 0); // 0 means auto-tune
    EXPECT_TRUE(medium_C.equals(medium_expected, tolerance));
}

// Test SIMD implementation if available
TEST_F(CPUMatrixMultiplicationTest, SimdMultiplication) {
    if (MatrixMult::CPU::is_avx512_supported()) {
        // Test with small matrices
        MatrixMult::CPU::simd_multiply(small_A, small_B, small_C);
        EXPECT_TRUE(small_C.equals(small_expected, tolerance));
        
        // Test with medium matrices
        MatrixMult::CPU::simd_multiply(medium_A, medium_B, medium_C);
        EXPECT_TRUE(medium_C.equals(medium_expected, tolerance));
    } else {
        GTEST_SKIP() << "AVX-512 not supported on this hardware";
    }
}

// Test threaded implementation with different thread counts
TEST_F(CPUMatrixMultiplicationTest, ThreadedMultiplication) {
    // Set number of threads for OpenMP
    omp_set_num_threads(2);
    
    // Test with small matrices
    MatrixMult::CPU::threaded_multiply(small_A, small_B, small_C);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Reset matrices
    small_C = Matrix(128, 128, 0.0f);
    
    // Test with more threads
    omp_set_num_threads(4);
    MatrixMult::CPU::threaded_multiply(small_A, small_B, small_C);
    EXPECT_TRUE(small_C.equals(small_expected, tolerance));
    
    // Test with medium matrices and auto thread count
    omp_set_num_threads(omp_get_max_threads());
    MatrixMult::CPU::threaded_multiply(medium_A, medium_B, medium_C);
    EXPECT_TRUE(medium_C.equals(medium_expected, tolerance));
}

// Test combined threaded SIMD implementation
TEST_F(CPUMatrixMultiplicationTest, ThreadedSimdMultiplication) {
    if (MatrixMult::CPU::is_avx512_supported()) {
        // Test with medium matrices
        MatrixMult::CPU::threaded_simd_multiply(medium_A, medium_B, medium_C);
        EXPECT_TRUE(medium_C.equals(medium_expected, tolerance));
    } else {
        GTEST_SKIP() << "AVX-512 not supported on this hardware";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
