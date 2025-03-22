#include "cpu/naive.h"

namespace cpu {
namespace naive {

void multiply(const float* A, const float* B, float* C, int m, int n, int k) {
    // Initialize C to zeros
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

void transpose(const float* A, float* B, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void multiply_transposed(const float* A, const float* B_trans, float* C, int m, int n, int k) {
    // Initialize C to zeros
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }

    // Perform matrix multiplication with transposed B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                // B_trans is stored in column-major order of the original B
                sum += A[i * k + p] * B_trans[j * k + p];
            }
            C[i * n + j] = sum;
        }
    }
}

} // namespace naive
} // namespace cpu
