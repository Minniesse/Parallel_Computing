#ifndef CPU_SIMD_H
#define CPU_SIMD_H

namespace cpu {
namespace simd {

/**
 * Matrix multiplication using AVX-512 SIMD instructions.
 * C = A * B
 * 
 * This implementation uses AVX-512 to process 16 single-precision
 * floating-point elements in parallel.
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_avx512(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Matrix multiplication using AVX2 SIMD instructions.
 * C = A * B
 * 
 * This implementation uses AVX2 to process 8 single-precision
 * floating-point elements in parallel.
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_avx2(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Matrix multiplication using AVX SIMD instructions.
 * C = A * B
 * 
 * This implementation uses AVX to process 8 single-precision
 * floating-point elements in parallel.
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_avx(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Matrix multiplication using SSE SIMD instructions.
 * C = A * B
 * 
 * This implementation uses SSE to process 4 single-precision
 * floating-point elements in parallel.
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_sse(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Detect best SIMD instruction set available and select appropriate implementation.
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
 * Matrix multiplication with register blocking and loop unrolling using AVX-512.
 * C = A * B
 * 
 * This implementation uses advanced optimizations such as:
 * - Register blocking to maximize data reuse within vector registers
 * - Cache blocking to improve temporal locality
 * - Loop unrolling to reduce branch prediction overhead
 * - Memory alignment to 64-byte boundaries for optimal vector operations
 * 
 * @param A Pointer to matrix A data (row-major order)
 * @param B Pointer to matrix B data (row-major order)
 * @param C Pointer to matrix C data (row-major order)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void multiply_blocked_avx512(const float* A, const float* B, float* C, int m, int n, int k);

/**
 * Get the optimal block size for the current CPU architecture.
 *
 * @return Optimal block size in elements
 */
int get_optimal_block_size();

/**
 * Check if AVX-512 instructions are supported on this CPU.
 *
 * @return true if AVX-512 is supported, false otherwise
 */
bool is_avx512_supported();

/**
 * Check if AVX2 instructions are supported on this CPU.
 *
 * @return true if AVX2 is supported, false otherwise
 */
bool is_avx2_supported();

/**
 * Check if AVX instructions are supported on this CPU.
 *
 * @return true if AVX is supported, false otherwise
 */
bool is_avx_supported();

/**
 * Check if SSE instructions are supported on this CPU.
 *
 * @return true if SSE is supported, false otherwise
 */
bool is_sse_supported();

} // namespace simd
} // namespace cpu

#endif // CPU_SIMD_H
