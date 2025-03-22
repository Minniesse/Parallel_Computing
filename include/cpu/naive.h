#ifndef CPU_NAIVE_H
#define CPU_NAIVE_H

namespace cpu {
namespace naive {

/**
 * Naive matrix multiplication implementation using triple-nested loops.
 * C = A * B
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Transpose a matrix.
 * B = A^T
 * 
 * @param A Pointer to input matrix data (row-major order)
 * @param B Pointer to output matrix data (row-major order)
 * @param rows Number of rows in A
 * @param cols Number of columns in A
 */
void transpose(const float* A, float* B, int rows, int cols);

/**
 * Naive matrix multiplication with pre-transposed B for better cache locality.
 * C = A * B
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B_trans Pointer to transposed matrix B data (column-major of original B)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_transposed(const float* A, const float* B_trans, float* C, int m, int n, int k);

} // namespace naive
} // namespace cpu

#endif // CPU_NAIVE_H
