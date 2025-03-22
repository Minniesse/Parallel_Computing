#include "common/matrix.h"
#include "common/timing.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"
#include "gpu/cuda_wrapper.h"
#include "adaptive.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

// Benchmark all implementations across different matrix sizes
void run_benchmarks(const std::vector<size_t>& sizes, int iterations = 3) {
    std::cout << "===========================================" << std::endl;
    std::cout << "Matrix Multiplication Benchmark" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Create output CSV file
    std::ofstream csv_file("benchmark_results.csv");
    if (csv_file.is_open()) {
        // Write CSV header
        csv_file << "Size,Implementation,AvgTime,MinTime,AvgGFLOPS,PeakGFLOPS" << std::endl;
    }
    
    // Print system information
    std::cout << "\nSystem Information:" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "CPU: " << std::thread::hardware_concurrency() << " logical cores" << std::endl;
    std::cout << "AVX-512 Support: " << (MatrixMult::CPU::is_avx512_supported() ? "Yes" : "No") << std::endl;
    
    if (MatrixMult::GPU::check_cuda_available()) {
        MatrixMult::GPU::print_cuda_device_info();
    } else {
        std::cout << "CUDA: Not available" << std::endl;
    }
    
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
        
        // Skip naive implementation for large matrices (too slow)
        if (size <= 1024) {
            benchmark.add_function("Naive", [&]() {
                MatrixMult::CPU::naive_multiply(A, B, C);
            });
        }
        
        benchmark.add_function("Blocked", [&]() {
            MatrixMult::CPU::blocked_multiply(A, B, C);
        });
        
        if (MatrixMult::CPU::is_avx512_supported()) {
            benchmark.add_function("SIMD (AVX-512)", [&]() {
                MatrixMult::CPU::simd_multiply(A, B, C);
            });
        }
        
        benchmark.add_function("Threaded", [&]() {
            MatrixMult::CPU::threaded_multiply(A, B, C);
        });
        
        if (MatrixMult::CPU::is_avx512_supported()) {
            benchmark.add_function("Threaded+SIMD", [&]() {
                MatrixMult::CPU::threaded_simd_multiply(A, B, C);
            });
        }
        
        if (MatrixMult::GPU::check_cuda_available()) {
            benchmark.add_function("CUDA Basic", [&]() {
                MatrixMult::GPU::cuda_multiply(A, B, C);
            });
            
            benchmark.add_function("CUDA Shared", [&]() {
                MatrixMult::GPU::cuda_shared_multiply(A, B, C);
            });
            
            benchmark.add_function("cuBLAS", [&]() {
                MatrixMult::GPU::cublas_multiply(A, B, C);
            });
        }
        
        benchmark.add_function("Adaptive", [&]() {
            MatrixMult::adaptive_multiply(A, B, C);
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
        std::cout << "\nBenchmark results saved to benchmark_results.csv" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default matrix sizes to benchmark
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
    
    // Default iterations
    int iterations = 3;
    
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
    
    run_benchmarks(sizes, iterations);
    
    return 0;
}
