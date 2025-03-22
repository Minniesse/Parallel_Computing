#pragma once

#include "common/matrix.h"

namespace MatrixMult {
namespace CPU {

// Naive triple-loop implementation
void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C);

} // namespace CPU
} // namespace MatrixMult
