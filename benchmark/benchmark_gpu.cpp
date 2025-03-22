#include "common/matrix.h"
#include "common/timing.h"
#include "cpu/naive.h"
#include "gpu/cuda_wrapper.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

// Declare the CUDA kernel used in analyze_memory_transfer
extern "C" __global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, 
                                                int m, int n, int k);

// Benchmark all GPU implementations across different matrix sizes
void run_gpu_benchmarks(const std::vector<size_t>& sizes, int iterations = 5) {
    std::cout << "===========================================" << std::endl;
    std::cout << "GPU Matrix Multiplication Benchmark" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Check if CUDA is available
    if (!MatrixMult::GPU::check_cuda_available()) {
        std::cerr << "CUDA is not available on this system. Exiting benchmark." << std::endl;
        return;
    }
    
    // Create output CSV file
    std::ofstream csv_file("gpu_benchmark_results.csv");
    if (csv_file.is_open()) {
        // Write CSV header
        csv_file << "Size,Implementation,AvgTime,MinTime,AvgGFLOPS,PeakGFLOPS" << std::endl;
    }
    
    // Print GPU information
    std::cout << "\nGPU Information:" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    MatrixMult::GPU::print_cuda_device_info();
    
    // Benchmark each matrix size
    for (auto size : sizes) {
        std::cout << "\n===========================================" << std::endl;
        std::cout << "Matrix Size: " << size << "x" << size << std::endl;
        std::cout << "===========================================" << std::endl;
        
        Matrix A(size, size);
        Matrix B(size, size);
        Matrix C(size, size, 0.0f);
        
        // Initialize matrices with random values
        A.fill_random();
        B.fill_random();
        
        // Create benchmark
        Benchmark benchmark;
        
        // Add CPU naive implementation as baseline for small matrices
        if (size <= 1024) {
            benchmark.add_function("CPU Naive", [&]() {
                MatrixMult::CPU::naive_multiply(A, B, C);
            });
        }
        
        // GPU implementations
        benchmark.add_function("CUDA Basic", [&]() {
            MatrixMult::GPU::cuda_multiply(A, B, C);
        });
        
        benchmark.add_function("CUDA Shared Memory", [&]() {
            MatrixMult::GPU::cuda_shared_multiply(A, B, C);
        });
        
        benchmark.add_function("cuBLAS", [&]() {
            MatrixMult::GPU::cublas_multiply(A, B, C);
        });
        
        // Run benchmarks
        benchmark.run_all(iterations);
        
        // Print GFLOPS
        benchmark.print_gflops(size, size, size);
        
        // Write results to CSV
        if (csv_file.is_open()) {
            for (const auto& result : benchmark.results()) {
                const std::string& name = result.first;
                const std::vector<double>& times = result.second;
                
                // Calculate statistics
                double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                double min_time = *std::min_element(times.begin(), times.end());
                
                // Calculate GFLOPS
                double total_flops = 2.0 * size * size * size;
                double avg_gflops = (total_flops / avg_time) / 1e9;
                double peak_gflops = (total_flops / min_time) / 1e9;
                
                // Write to CSV
                csv_file << size << ","
                        << name << ","
                        << avg_time << ","
                        << min_time << ","
                        << avg_gflops << ","
                        << peak_gflops << std::endl;
            }
        }
    }
    
    if (csv_file.is_open()) {
        csv_file.close();
        std::cout << "\nBenchmark results saved to gpu_benchmark_results.csv" << std::endl;
    }
}

// Analyze GPU memory transfer overhead
void analyze_memory_transfer(const std::vector<size_t>& sizes) {
    std::cout << "\n===========================================" << std::endl;
    std::cout << "GPU Memory Transfer Analysis" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    std::ofstream csv_file("gpu_memory_transfer.csv");
    if (csv_file.is_open()) {
        csv_file << "Size,MatrixBytes,HostToDeviceTime,DeviceToHostTime,ComputeTime,TotalTime,TransferOverhead" << std::endl;
    }
    
    for (auto size : sizes) {
        Matrix A(size, size);
        Matrix B(size, size);
        Matrix C(size, size, 0.0f);
        
        // Initialize matrices with random values
        A.fill_random();
        B.fill_random();
        
        size_t matrix_bytes = size * size * sizeof(float);
        size_t total_bytes = 3 * matrix_bytes; // A, B, and C
        
        // Device memory pointers
        float *d_A, *d_B, *d_C;
        
        // Allocate device memory
        cudaMalloc((void**)&d_A, matrix_bytes);
        cudaMalloc((void**)&d_B, matrix_bytes);
        cudaMalloc((void**)&d_C, matrix_bytes);
        
        // Measure host to device transfer time
        Timer h2d_timer;
        h2d_timer.start();
        cudaMemcpy(d_A, A.data(), matrix_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), matrix_bytes, cudaMemcpyHostToDevice);
        h2d_timer.stop();
        
        // Define grid and block dimensions
        dim3 blockDim(32, 32);
        dim3 gridDim((size + blockDim.x - 1) / blockDim.x, 
                     (size + blockDim.y - 1) / blockDim.y);
        
        // Measure computation time
        Timer compute_timer;
        compute_timer.start();
        // Replace direct kernel call with wrapper function
        runMatrixMulSharedKernel(d_A, d_B, d_C, size, size, size, gridDim, blockDim);
        compute_timer.stop();
        
        // Measure device to host transfer time
        Timer d2h_timer;
        d2h_timer.start();
        cudaMemcpy(C.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost);
        d2h_timer.stop();
        
        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        // Calculate total time and transfer overhead
        double h2d_time = h2d_timer.elapsed_milliseconds();
        double compute_time = compute_timer.elapsed_milliseconds();
        double d2h_time = d2h_timer.elapsed_milliseconds();
        double total_time = h2d_time + compute_time + d2h_time;
        double transfer_overhead = (h2d_time + d2h_time) / total_time * 100.0;
        
        // Print results
        std::cout << "Matrix size: " << size << "x" << size << std::endl;
        std::cout << "  Total bytes: " << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Host to device time: " << h2d_time << " ms" << std::endl;
        std::cout << "  Compute time: " << compute_time << " ms" << std::endl;
        std::cout << "  Device to host time: " << d2h_time << " ms" << std::endl;
        std::cout << "  Total time: " << total_time << " ms" << std::endl;
        std::cout << "  Transfer overhead: " << transfer_overhead << "%" << std::endl;
        
        // Write to CSV
        if (csv_file.is_open()) {
            csv_file << size << ","
                    << matrix_bytes << ","
                    << h2d_time << ","
                    << d2h_time << ","
                    << compute_time << ","
                    << total_time << ","
                    << transfer_overhead << std::endl;
        }
    }
    
    if (csv_file.is_open()) {
        csv_file.close();
        std::cout << "\nMemory transfer analysis saved to gpu_memory_transfer.csv" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default matrix sizes to benchmark
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
    
    // Default iterations
    int iterations = 5;
    
    // Parse command line arguments
    if (argc > 1) {
        // Custom sizes
        sizes.clear();
        for (int i = 1; i < argc - 1; ++i) {
            sizes.push_back(std::stoi(argv[i]));
        }
        
        // Custom iterations (last argument)
        if (argc > 2) {
            iterations = std::stoi(argv[argc - 1]);
        }
    }
    
    // Run GPU benchmarks
    run_gpu_benchmarks(sizes, iterations);
    
    // Analyze memory transfer overhead
    analyze_memory_transfer(sizes);
    
    return 0;
}
