#pragma once

#include "common/matrix.h"

namespace MatrixMult {

// Automatically selects the best algorithm based on input size and hardware
void adaptive_multiply(const Matrix& A, const Matrix& B, Matrix& C);

// Algorithm selection decision thresholds
struct AdaptiveThresholds {
    size_t cpu_gpu_crossover;     // Threshold to switch from CPU to GPU
    size_t naive_blocked_crossover; // Threshold to switch from naive to blocked
    size_t block_size_small;      // Block size for small matrices
    size_t block_size_large;      // Block size for large matrices
};

// Get the current system's optimal thresholds
AdaptiveThresholds get_optimal_thresholds();

} // namespace MatrixMult
