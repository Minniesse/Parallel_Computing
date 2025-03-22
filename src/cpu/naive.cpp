#include "cpu/naive.h"

namespace MatrixMult {
namespace CPU {

void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Simple triple-loop matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += A.at(i, p) * B.at(p, j);
            }
            C.at(i, j) = sum;
        }
    }
}

} // namespace CPU
} // namespace MatrixMult
