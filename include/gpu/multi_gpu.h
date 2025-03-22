#ifndef GPU_MULTI_GPU_H
#define GPU_MULTI_GPU_H

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace gpu {

/**
 * Multi-GPU matrix multiplication class.
 * This class distributes matrix multiplication across multiple GPUs.
 */
class MultiGpuMatrixMultiplier {
private:
    // Device information
    int num_gpus_;
    std::vector<int> device_ids_;
    
    // Matrix dimensions
    int m_, n_, k_;
    
    // Memory allocation sizes
    size_t bytes_a_;
    size_t bytes_b_;
    size_t bytes_c_;
    
    // Device arrays and streams
    std::vector<float*> d_a_;
    std::vector<float*> d_b_;
    std::vector<float*> d_c_;
    std::vector<cudaStream_t> streams_;
    std::vector<cublasHandle_t> handles_;
    
    // Initialization status
    bool initialized_;
    
public:
    MultiGpuMatrixMultiplier();
    ~MultiGpuMatrixMultiplier();
    
    /**
     * Initialize multi-GPU context for matrix multiplication.
     * 
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param device_ids List of GPU device IDs to use (empty means use all available)
     * @return true if initialization was successful, false otherwise
     */
    bool initialize(int m, int n, int k, const std::vector<int>& device_ids = {});
    
    /**
     * Release device memory and cleanup.
     */
    void cleanup();
    
    /**
     * Transfer matrix data to GPUs.
     * 
     * @param A Pointer to matrix A data (row-major order)
     * @param B Pointer to matrix B data (row-major order)
     */
    void setMatrices(const float* A, const float* B);
    
    /**
     * Perform matrix multiplication using multiple GPUs.
     * C = A * B
     * 
     * @param C Pointer to result matrix C data (row-major order)
     * @param alpha Scalar multiplier for A*B
     * @param beta Scalar multiplier for C
     */
    void multiply(float* C, float alpha = 1.0f, float beta = 0.0f);
    
    /**
     * Perform matrix multiplication using cuBLAS on multiple GPUs.
     * 
     * @param C Pointer to result matrix C data (row-major order)
     * @param alpha Scalar multiplier for A*B
     * @param beta Scalar multiplier for C
     */
    void multiplyWithCuBLAS(float* C, float alpha = 1.0f, float beta = 0.0f);
    
    /**
     * Check if the system supports multi-GPU operations.
     * 
     * @return true if multiple GPUs are available, false otherwise
     */
    static bool isMultiGpuAvailable();
    
    /**
     * Get the number of available GPUs.
     * 
     * @return Number of available CUDA devices
     */
    static int getGpuCount();
};

} // namespace gpu

#endif // GPU_MULTI_GPU_H
