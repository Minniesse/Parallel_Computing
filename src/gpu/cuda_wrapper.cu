#include "gpu/cuda_wrapper.h"
#include <stdio.h>
#include <cublas_v2.h>

// CUDA kernel for matrix multiplication (naive version)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, 
                               int m, int n, int k) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with shared memory
__global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, 
                                     int m, int n, int k) {
    // Shared memory for tile of A and B
    __shared__ float sharedA[32][32];
    __shared__ float sharedB[32][32];
    
    // Calculate global and local indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int p = 0; p < (k + 31) / 32; ++p) {
        // Load A tile
        if (row < m && p * 32 + tx < k) {
            sharedA[ty][tx] = A[row * k + p * 32 + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        // Load B tile
        if (p * 32 + ty < k && col < n) {
            sharedB[ty][tx] = B[(p * 32 + ty) * n + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int i = 0; i < 32; ++i) {
            sum += sharedA[ty][i] * sharedB[i][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// Exported C-style wrapper function for the memory transfer benchmark
extern "C" void runMatrixMulSharedKernel(const float* A, const float* B, float* C, 
                                         int m, int n, int k, dim3 gridDim, dim3 blockDim) {
    matrixMulSharedKernel<<<gridDim, blockDim>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
}

namespace MatrixMult {
namespace GPU {

bool check_cuda_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    return true;
}

void print_cuda_device_info() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    printf("CUDA Devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        printf("  Device %d: %s\n", i, deviceProp.name);
        printf("    Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("    Total Global Memory: %.2f GB\n", 
               static_cast<float>(deviceProp.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
        printf("    Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    }
}

void cuda_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Device memory pointers
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    
    // Copy result from device to host
    cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cuda_shared_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Device memory pointers
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel with shared memory
    matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    
    // Copy result from device to host
    cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cublas_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    // Ensure matrices can be multiplied
    if (!MatrixMult::can_multiply(A, B, C)) {
        return;
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Device memory pointers
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up scaling factors
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Call cuBLAS GEMM (Note: cuBLAS uses column-major order)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
               n, m, k, 
               &alpha, 
               d_B, n,  // B is transposed due to column-major
               d_A, k,  // A is transposed due to column-major
               &beta, 
               d_C, n); // C is transposed due to column-major
    
    // Copy result from device to host
    cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // namespace GPU
} // namespace MatrixMult
