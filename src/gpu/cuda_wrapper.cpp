#include "gpu/cuda_wrapper.h"
#include <iostream> // Add this include for std::cout, std::cerr, std::endl

namespace gpu {

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

std::vector<CudaDeviceInfo> getAvailableDevices() {
    std::vector<CudaDeviceInfo> devices;
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        return devices;
    }
    
    devices.resize(deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        devices[i].deviceId = i;
        devices[i].name = prop.name;
        devices[i].totalMemory = prop.totalGlobalMem;
        devices[i].multiprocessorCount = prop.multiProcessorCount;
        devices[i].maxThreadsPerBlock = prop.maxThreadsPerBlock;
        devices[i].maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        devices[i].warpSize = prop.warpSize;
        
        // Check if Tensor Cores are available (Volta or newer architecture)
        devices[i].tensorCoresAvailable = (prop.major >= 7);
    }
    
    return devices;
}

void printDeviceProperties(int deviceId) {
    if (!isCudaAvailable()) {
        std::cerr << "CUDA is not available on this system." << std::endl;
        return;
    }
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceId >= deviceCount) {
        std::cerr << "Invalid device ID: " << deviceId << std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    std::cout << "Device: " << prop.name << " (ID: " << deviceId << ")" << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Tensor cores available: " << (prop.major >= 7 ? "Yes" : "No") << std::endl;
}

int getOptimalBlockSize(int matrixSize) {
    // This is a simple heuristic, in practice would be tuned for specific hardware
    return 16;
}

// Matrix Multiplier implementation
CudaMatrixMultiplier::CudaMatrixMultiplier() 
    : handle(nullptr), deviceId(0), initialized(false), 
      d_A(nullptr), d_B(nullptr), d_C(nullptr), 
      m(0), n(0), k(0), bytesA(0), bytesB(0), bytesC(0) {}

CudaMatrixMultiplier::~CudaMatrixMultiplier() {
    cleanup();
}

void CudaMatrixMultiplier::initialize(int m, int n, int k, int device) {
    // Clean up previous resources if already initialized
    if (initialized) {
        cleanup();
    }
    
    // Store dimensions
    this->m = m;
    this->n = n;
    this->k = k;
    
    // Calculate memory requirements
    bytesA = m * k * sizeof(float);
    bytesB = k * n * sizeof(float);
    bytesC = m * n * sizeof(float);
    
    // Set device
    deviceId = device;
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    // Create cuBLAS handle
    cublasCreate(&handle);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    
    initialized = true;
}

void CudaMatrixMultiplier::cleanup() {
    if (!initialized) {
        return;
    }
    
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    // Free device memory
    if (d_A) CUDA_CHECK(cudaFree(d_A));
    if (d_B) CUDA_CHECK(cudaFree(d_B));
    if (d_C) CUDA_CHECK(cudaFree(d_C));
    
    d_A = nullptr;
    d_B = nullptr;
    d_C = nullptr;
    
    // Destroy cuBLAS handle
    if (handle) cublasDestroy(handle);
    handle = nullptr;
    
    initialized = false;
}

void CudaMatrixMultiplier::setMatrixA(const float* A) {
    if (!initialized) {
        throw CudaError("CudaMatrixMultiplier not initialized");
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice));
}

void CudaMatrixMultiplier::setMatrixB(const float* B) {
    if (!initialized) {
        throw CudaError("CudaMatrixMultiplier not initialized");
    }
    
    CUDA_CHECK(cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice));
}

// Forward declaration of the kernel function (implemented in matmul_kernel.cu)
extern "C" void launchMatrixMultiplyKernel(float* C, const float* A, const float* B, 
                                         int m, int n, int k, float alpha, float beta, 
                                         cudaStream_t stream);

void CudaMatrixMultiplier::multiply(float* C, float alpha, float beta) {
    if (!initialized) {
        throw CudaError("CudaMatrixMultiplier not initialized");
    }
    
    // Launch kernel
    launchMatrixMultiplyKernel(d_C, d_A, d_B, m, n, k, alpha, beta, 0);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost));
}

void CudaMatrixMultiplier::multiplyWithCuBLAS(float* C, float alpha, float beta) {
    if (!initialized) {
        throw CudaError("CudaMatrixMultiplier not initialized");
    }
    
    // cuBLAS expects matrices in column-major order, but we're using row-major order
    // So we compute B * A instead of A * B and transpose the result
    // This is using the identity: C = A * B => C^T = B^T * A^T
    
    // Perform multiplication: d_C = alpha * d_B * d_A + beta * d_C
    // Note: cuBLAS uses column-major ordering
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k, 
                &alpha, 
                d_B, n, 
                d_A, k, 
                &beta, 
                d_C, n);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost));
    
    // If needed, transpose the result to get back to row-major order
    // This step is not required if the calling code expects column-major order
}

bool CudaMatrixMultiplier::multiplyWithTensorCores(float* C, float alpha, float beta) {
    if (!initialized) {
        throw CudaError("CudaMatrixMultiplier not initialized");
    }
    
    // Check if tensor cores are available (requires Volta or newer architecture)
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    
    if (props.major < 7) {
        // Tensor cores not available, fall back to regular multiplication
        multiply(C, alpha, beta);
        return false;
    }
    
    // Use tensor cores via cuBLAS
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // Perform multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k, 
                &alpha, 
                d_B, n, 
                d_A, k, 
                &beta, 
                d_C, n);
    
    // Reset math mode
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost));
    
    return true;
}

void CudaMatrixMultiplier::multiplyAsync(float* C, cudaStream_t stream, float alpha, float beta) {
    if (!initialized) {
        throw CudaError("CudaMatrixMultiplier not initialized");
    }
    
    // Launch kernel with specified stream
    launchMatrixMultiplyKernel(d_C, d_A, d_B, m, n, k, alpha, beta, stream);
    
    // Asynchronous copy of result back to host
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, bytesC, cudaMemcpyDeviceToHost, stream));
}

float CudaMatrixMultiplier::getLastOperationTime() const {
    // This would require timing information to be stored during operations
    return 0.0f;
}

float CudaMatrixMultiplier::getGFLOPS() const {
    // This would require timing and FLOP count information
    return 0.0f;
}

float CudaMatrixMultiplier::getBandwidthUsage() const {
    // This would require timing and memory access information
    return 0.0f;
}

} // namespace gpu
