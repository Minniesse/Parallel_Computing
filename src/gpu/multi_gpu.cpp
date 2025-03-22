#include "gpu/multi_gpu.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace gpu {

// Helper function to check CUDA errors
inline void checkCudaErrorsMGPU(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error)
                  << " at " << file << ":" << line << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

#define CUDA_CHECK_MGPU(call) checkCudaErrorsMGPU(call, __FILE__, __LINE__)

MultiGpuMatrixMultiplier::MultiGpuMatrixMultiplier() 
    : num_gpus_(0), m_(0), n_(0), k_(0), 
      bytes_a_(0), bytes_b_(0), bytes_c_(0), 
      initialized_(false) {}

MultiGpuMatrixMultiplier::~MultiGpuMatrixMultiplier() {
    cleanup();
}

bool MultiGpuMatrixMultiplier::initialize(int m, int n, int k, const std::vector<int>& device_ids) {
    // Clean up previous resources if already initialized
    if (initialized_) {
        cleanup();
    }
    
    // Store matrix dimensions
    m_ = m;
    n_ = n;
    k_ = k;
    
    // Calculate memory requirements
    bytes_a_ = m * k * sizeof(float);
    bytes_b_ = k * n * sizeof(float);
    bytes_c_ = m * n * sizeof(float);
    
    // Determine which GPUs to use
    if (device_ids.empty()) {
        // Use all available GPUs
        int device_count;
        CUDA_CHECK_MGPU(cudaGetDeviceCount(&device_count));
        
        if (device_count == 0) {
            std::cerr << "No CUDA devices found." << std::endl;
            return false;
        }
        
        device_ids_.resize(device_count);
        for (int i = 0; i < device_count; ++i) {
            device_ids_[i] = i;
        }
    } else {
        // Use specified GPUs
        device_ids_ = device_ids;
    }
    
    num_gpus_ = static_cast<int>(device_ids_.size());
    
    // Initialize resources for each GPU
    d_a_.resize(num_gpus_, nullptr);
    d_b_.resize(num_gpus_, nullptr);
    d_c_.resize(num_gpus_, nullptr);
    streams_.resize(num_gpus_);
    handles_.resize(num_gpus_);
    
    // Calculate rows per GPU (simple row-wise distribution)
    int rows_per_gpu = (m + num_gpus_ - 1) / num_gpus_;
    
    // Initialize each GPU
    for (int i = 0; i < num_gpus_; ++i) {
        // Set device
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Create stream
        CUDA_CHECK_MGPU(cudaStreamCreate(&streams_[i]));
        
        // Create cuBLAS handle
        cublasCreate(&handles_[i]);
        cublasSetStream(handles_[i], streams_[i]);
        
        // Calculate rows for this GPU
        int gpu_m_start = i * rows_per_gpu;
        int gpu_m_end = std::min(gpu_m_start + rows_per_gpu, m);
        int gpu_m = gpu_m_end - gpu_m_start;
        
        if (gpu_m <= 0) continue;  // Skip if no rows for this GPU
        
        // Allocate device memory
        CUDA_CHECK_MGPU(cudaMalloc(&d_a_[i], gpu_m * k * sizeof(float)));
        CUDA_CHECK_MGPU(cudaMalloc(&d_b_[i], k * n * sizeof(float)));
        CUDA_CHECK_MGPU(cudaMalloc(&d_c_[i], gpu_m * n * sizeof(float)));
    }
    
    initialized_ = true;
    return true;
}

void MultiGpuMatrixMultiplier::cleanup() {
    if (!initialized_) return;
    
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Free device memory
        if (d_a_[i]) CUDA_CHECK_MGPU(cudaFree(d_a_[i]));
        if (d_b_[i]) CUDA_CHECK_MGPU(cudaFree(d_b_[i]));
        if (d_c_[i]) CUDA_CHECK_MGPU(cudaFree(d_c_[i]));
        
        // Destroy cuBLAS handle
        if (handles_[i]) cublasDestroy(handles_[i]);
        
        // Destroy stream
        if (streams_[i]) CUDA_CHECK_MGPU(cudaStreamDestroy(streams_[i]));
    }
    
    d_a_.clear();
    d_b_.clear();
    d_c_.clear();
    streams_.clear();
    handles_.clear();
    device_ids_.clear();
    
    num_gpus_ = 0;
    initialized_ = false;
}

void MultiGpuMatrixMultiplier::setMatrices(const float* A, const float* B) {
    if (!initialized_) {
        throw std::runtime_error("MultiGpuMatrixMultiplier not initialized");
    }
    
    // Calculate rows per GPU
    int rows_per_gpu = (m_ + num_gpus_ - 1) / num_gpus_;
    
    // Copy matrices to each GPU
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Calculate rows for this GPU
        int gpu_m_start = i * rows_per_gpu;
        int gpu_m_end = std::min(gpu_m_start + rows_per_gpu, m_);
        int gpu_m = gpu_m_end - gpu_m_start;
        
        if (gpu_m <= 0) continue;  // Skip if no rows for this GPU
        
        // Copy slice of A
        CUDA_CHECK_MGPU(cudaMemcpyAsync(d_a_[i], 
                                       A + gpu_m_start * k_, 
                                       gpu_m * k_ * sizeof(float), 
                                       cudaMemcpyHostToDevice, 
                                       streams_[i]));
        
        // Copy B (each GPU needs the full B matrix)
        CUDA_CHECK_MGPU(cudaMemcpyAsync(d_b_[i], 
                                       B, 
                                       k_ * n_ * sizeof(float), 
                                       cudaMemcpyHostToDevice, 
                                       streams_[i]));
    }
    
    // Synchronize all devices
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        CUDA_CHECK_MGPU(cudaStreamSynchronize(streams_[i]));
    }
}

void MultiGpuMatrixMultiplier::multiply(float* C, float alpha, float beta) {
    if (!initialized_) {
        throw std::runtime_error("MultiGpuMatrixMultiplier not initialized");
    }
    
    // Forward declaration of kernel launcher
    extern void launchMatrixMultiplyKernel(float* C, const float* A, const float* B, 
                                         int m, int n, int k, float alpha, float beta, 
                                         cudaStream_t stream);
    
    // Calculate rows per GPU
    int rows_per_gpu = (m_ + num_gpus_ - 1) / num_gpus_;
    
    // Launch kernel on each GPU
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Calculate rows for this GPU
        int gpu_m_start = i * rows_per_gpu;
        int gpu_m_end = std::min(gpu_m_start + rows_per_gpu, m_);
        int gpu_m = gpu_m_end - gpu_m_start;
        
        if (gpu_m <= 0) continue;  // Skip if no rows for this GPU
        
        // Launch kernel
        launchMatrixMultiplyKernel(d_c_[i], d_a_[i], d_b_[i], 
                                  gpu_m, n_, k_, alpha, beta, 
                                  streams_[i]);
    }
    
    // Gather results from all GPUs
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Calculate rows for this GPU
        int gpu_m_start = i * rows_per_gpu;
        int gpu_m_end = std::min(gpu_m_start + rows_per_gpu, m_);
        int gpu_m = gpu_m_end - gpu_m_start;
        
        if (gpu_m <= 0) continue;  // Skip if no rows for this GPU
        
        // Copy result back to host
        CUDA_CHECK_MGPU(cudaMemcpyAsync(C + gpu_m_start * n_, 
                                       d_c_[i], 
                                       gpu_m * n_ * sizeof(float), 
                                       cudaMemcpyDeviceToHost, 
                                       streams_[i]));
    }
    
    // Synchronize all devices
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        CUDA_CHECK_MGPU(cudaStreamSynchronize(streams_[i]));
    }
}

void MultiGpuMatrixMultiplier::multiplyWithCuBLAS(float* C, float alpha, float beta) {
    if (!initialized_) {
        throw std::runtime_error("MultiGpuMatrixMultiplier not initialized");
    }
    
    // Calculate rows per GPU
    int rows_per_gpu = (m_ + num_gpus_ - 1) / num_gpus_;
    
    // Launch cuBLAS on each GPU
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Calculate rows for this GPU
        int gpu_m_start = i * rows_per_gpu;
        int gpu_m_end = std::min(gpu_m_start + rows_per_gpu, m_);
        int gpu_m = gpu_m_end - gpu_m_start;
        
        if (gpu_m <= 0) continue;  // Skip if no rows for this GPU
        
        // cuBLAS expects matrices in column-major order
        // So we compute B * A instead of A * B and transpose the result
        // This is using the identity: C = A * B => C^T = B^T * A^T
        cublasSgemm(handles_[i], CUBLAS_OP_N, CUBLAS_OP_N, 
                   n_, gpu_m, k_, 
                   &alpha, 
                   d_b_[i], n_, 
                   d_a_[i], k_, 
                   &beta, 
                   d_c_[i], n_);
    }
    
    // Gather results from all GPUs
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        
        // Calculate rows for this GPU
        int gpu_m_start = i * rows_per_gpu;
        int gpu_m_end = std::min(gpu_m_start + rows_per_gpu, m_);
        int gpu_m = gpu_m_end - gpu_m_start;
        
        if (gpu_m <= 0) continue;  // Skip if no rows for this GPU
        
        // Copy result back to host
        CUDA_CHECK_MGPU(cudaMemcpyAsync(C + gpu_m_start * n_, 
                                       d_c_[i], 
                                       gpu_m * n_ * sizeof(float), 
                                       cudaMemcpyDeviceToHost, 
                                       streams_[i]));
    }
    
    // Synchronize all devices
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK_MGPU(cudaSetDevice(device_ids_[i]));
        CUDA_CHECK_MGPU(cudaStreamSynchronize(streams_[i]));
    }
}

bool MultiGpuMatrixMultiplier::isMultiGpuAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 1);
}

int MultiGpuMatrixMultiplier::getGpuCount() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess) ? device_count : 0;
}

} // namespace gpu
