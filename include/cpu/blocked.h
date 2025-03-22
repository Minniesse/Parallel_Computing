#ifndef CPU_BLOCKED_H
#define CPU_BLOCKED_H

namespace cpu {
namespace blocked {

/**
 * Cache-blocked matrix multiplication implementation.
 * C = A * B
 * 
 * This implementation divides matrices into cache-sized blocks
 * to improve temporal and spatial cache locality.
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
 * Cache-blocked matrix multiplication with explicit block size.
 * C = A * B
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param block_size Block size to use for blocking
 */
void multiply_with_block_size(const float* A, const float* B, float* C, 
                              int m, int n, int k, int block_size);

/**
 * Get the optimal block size for the current CPU's cache.
 *
 * @return Optimal block size in elements
 */
int get_optimal_block_size();

} // namespace blocked
} // namespace cpu

#endif // CPU_BLOCKED_H
