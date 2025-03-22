#ifndef CPU_THREADED_H
#define CPU_THREADED_H

namespace cpu {
namespace threaded {

/**
 * Multithreaded matrix multiplication implementation using OpenMP.
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
 * Combined SIMD and multithreaded matrix multiplication.
 * C = A * B
 * 
 * This implementation uses both SIMD instructions and OpenMP
 * for maximum parallelism on modern CPUs.
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_simd(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Set the number of threads to use for multiplication.
 * 
 * @param num_threads Number of threads to use (0 means use all available cores)
 */
void set_num_threads(int num_threads);

/**
 * Get the current number of threads being used.
 * 
 * @return Current number of threads
 */
int get_num_threads();

} // namespace threaded
} // namespace cpu

#endif // CPU_THREADED_H
