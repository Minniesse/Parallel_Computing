#include <cuda_runtime.h>

// Define block size for CUDA kernel
#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(float* C, const float* A, const float* B, 
                                   int m, int n, int k, float alpha, float beta) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Global row and column in C
    int globalRow = blockRow * BLOCK_SIZE + row;
    int globalCol = blockCol * BLOCK_SIZE + col;
    
    // Check bounds
    if (globalRow >= m || globalCol >= n) return;
    
    // Shared memory for tile of A and B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    // Accumulate product
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tiles into shared memory
        int tileIdx = t * BLOCK_SIZE;
        
        // Load tile of A into shared memory
        if (globalRow < m && tileIdx + col < k) {
            s_A[row][col] = A[globalRow * k + tileIdx + col];
        } else {
            s_A[row][col] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if (tileIdx + row < k && globalCol < n) {
            s_B[row][col] = B[(tileIdx + row) * n + globalCol];
        } else {
            s_B[row][col] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Multiply tiles and accumulate
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += s_A[row][i] * s_B[i][col];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to C with alpha and beta scaling
    if (globalRow < m && globalCol < n) {
        C[globalRow * n + globalCol] = alpha * sum + beta * C[globalRow * n + globalCol];
    }
}

// Host function to launch the kernel
extern "C" void launchMatrixMultiplyKernel(float* C, const float* A, const float* B, 
                                         int m, int n, int k, float alpha, float beta, 
                                         cudaStream_t stream) {
    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    matrixMultiplyKernel<<<gridDim, blockDim, 0, stream>>>(C, A, B, m, n, k, alpha, beta);
}
