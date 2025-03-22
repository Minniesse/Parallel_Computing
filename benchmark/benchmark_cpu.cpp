#include "common/matrix.h"
#include "common/timing.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

// Benchmark all CPU implementations across different matrix sizes
void run_cpu_benchmarks(const std::vector<size_t>& sizes, int iterations = 5) {
    std::cout << "===========================================" << std::endl;
    std::cout << "CPU Matrix Multiplication Benchmark" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Create output CSV file
    std::ofstream csv_file("cpu_benchmark_results.csv");
    if (csv_file.is_open()) {
        // Write CSV header
        csv_file << "Size,Implementation,AvgTime,MinTime,AvgGFLOPS,PeakGFLOPS" << std::endl;
    }
    
    // Print CPU information
    std::cout << "\nCPU Information:" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Logical cores: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "AVX-512 Support: " << (MatrixMult::CPU::is_avx512_supported() ? "Yes" : "No") << std::endl;
    
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
        
        // Test blocked algorithm with different block sizes
        benchmark.add_function("Blocked-32", [&]() {
            MatrixMult::CPU::blocked_multiply(A, B, C, 32);
        });
        
        benchmark.add_function("Blocked-64", [&]() {
            MatrixMult::CPU::blocked_multiply(A, B, C, 64);
        });
        
        benchmark.add_function("Blocked-128", [&]() {
            MatrixMult::CPU::blocked_multiply(A, B, C, 128);
        });
        
        benchmark.add_function("Blocked-Auto", [&]() {
            MatrixMult::CPU::blocked_multiply(A, B, C, 0); // 0 means auto-tune
        });
        
        if (MatrixMult::CPU::is_avx512_supported()) {
            benchmark.add_function("SIMD (AVX-512)", [&]() {
                MatrixMult::CPU::simd_multiply(A, B, C);
            });
        }
        
        // Test threaded implementation with different thread counts
        int max_threads = std::thread::hardware_concurrency();
        
        for (int threads : {2, 4, 8, max_threads}) {
            if (threads > max_threads) continue;
            
            benchmark.add_function("Threaded-" + std::to_string(threads), [&, threads]() {
                omp_set_num_threads(threads);
                MatrixMult::CPU::threaded_multiply(A, B, C);
            });
        }
        
        if (MatrixMult::CPU::is_avx512_supported()) {
            benchmark.add_function("Threaded+SIMD", [&]() {
                omp_set_num_threads(max_threads);
                MatrixMult::CPU::threaded_simd_multiply(A, B, C);
            });
        }
        
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
        std::cout << "\nBenchmark results saved to cpu_benchmark_results.csv" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default matrix sizes to benchmark
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048};
    
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
    
    run_cpu_benchmarks(sizes, iterations);
    
    return 0;
}
