#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace gpu {

// Error handling
class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

// Throw exception on CUDA error
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::string msg = "CUDA error: " + std::string(cudaGetErrorString(error)) + 
                          " at " + std::string(file) + ":" + std::to_string(line);
        throw CudaError(msg);
    }
}

// Macro for error checking
#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

// CUDA Device Information
struct CudaDeviceInfo {
    int deviceId;
    std::string name;
    size_t totalMemory;
    int multiprocessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int warpSize;
    bool tensorCoresAvailable;
};

// Get information about available CUDA devices
std::vector<CudaDeviceInfo> getAvailableDevices();

// CUDA Matrix Multiplication
class CudaMatrixMultiplier {
private:
    cublasHandle_t handle;
    int deviceId;
    bool initialized;
    
    // Device pointers for matrix storage
    float* d_A;
    float* d_B;
    float* d_C;
    
    // Matrix dimensions
    int m, n, k;
    
    // Memory allocation sizes
    size_t bytesA;
    size_t bytesB;
    size_t bytesC;
    
public:
    CudaMatrixMultiplier();
    ~CudaMatrixMultiplier();
    
    // Initialize CUDA context and allocate memory for matrices
    void initialize(int m, int n, int k, int device = 0);
    
    // Release device memory and cleanup
    void cleanup();
    
    // Transfer matrix data to device
    void setMatrixA(const float* A);
    void setMatrixB(const float* B);
    
    // Perform matrix multiplication using various implementations
    // C = alpha*A*B + beta*C
    
    // Using basic CUDA kernel (our implementation)
    void multiply(float* C, float alpha = 1.0f, float beta = 0.0f);
    
    // Using cuBLAS (for comparison)
    void multiplyWithCuBLAS(float* C, float alpha = 1.0f, float beta = 0.0f);
    
    // Using tensor cores (if available)
    bool multiplyWithTensorCores(float* C, float alpha = 1.0f, float beta = 0.0f);
    
    // Asynchronous version for overlapping computation and data transfer
    void multiplyAsync(float* C, cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f);
    
    // Get performance metrics
    float getLastOperationTime() const;
    float getGFLOPS() const;
    float getBandwidthUsage() const;
};

// Utility functions
bool isCudaAvailable();
int getOptimalBlockSize(int matrixSize);
void printDeviceProperties(int deviceId = 0);

} // namespace gpu

#endif // CUDA_WRAPPER_H
