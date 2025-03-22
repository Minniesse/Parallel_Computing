#include "common/matrix.h"
#include "common/timing.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"
#include "gpu/cuda_wrapper.h"
#include "adaptive.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <functional>

// Function to verify correctness of all implementations
bool verify_implementations(size_t size) {
    Matrix A(size, size);
    Matrix B(size, size);
    Matrix C_naive(size, size, 0.0f);
    Matrix C_blocked(size, size, 0.0f);
    Matrix C_simd(size, size, 0.0f);
    Matrix C_threaded(size, size, 0.0f);
    Matrix C_cuda(size, size, 0.0f);
    
    // Initialize matrices with random values
    A.fill_random();
    B.fill_random();
    
    // Run all implementations
    MatrixMult::CPU::naive_multiply(A, B, C_naive);
    MatrixMult::CPU::blocked_multiply(A, B, C_blocked);
    
    bool all_correct = true;
    
    if (!C_naive.equals(C_blocked)) {
        std::cerr << "Error: Blocked implementation produces different results from naive." << std::endl;
        all_correct = false;
    }
    
    if (MatrixMult::CPU::is_avx512_supported()) {
        MatrixMult::CPU::simd_multiply(A, B, C_simd);
        if (!C_naive.equals(C_simd)) {
            std::cerr << "Error: SIMD implementation produces different results from naive." << std::endl;
            all_correct = false;
        }
    }
    
    MatrixMult::CPU::threaded_multiply(A, B, C_threaded);
    if (!C_naive.equals(C_threaded)) {
        std::cerr << "Error: Threaded implementation produces different results from naive." << std::endl;
        all_correct = false;
    }
    
    if (MatrixMult::GPU::check_cuda_available()) {
        MatrixMult::GPU::cuda_multiply(A, B, C_cuda);
        if (!C_naive.equals(C_cuda, 1e-4)) {  // Use slightly larger tolerance for GPU due to potential floating-point differences
            std::cerr << "Error: CUDA implementation produces different results from naive." << std::endl;
            all_correct = false;
        }
    }
    
    return all_correct;
}

// Benchmark all implementations for given matrix size
void benchmark_size(size_t size, int iterations = 5) {
    std::cout << "\nBenchmarking matrix size: " << size << "x" << size << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    Matrix A(size, size);
    Matrix B(size, size);
    Matrix C(size, size, 0.0f);
    
    // Initialize matrices with random values
    A.fill_random();
    B.fill_random();
    
    // Create benchmark
    Benchmark benchmark;
    
    // Add implementations to benchmark
    benchmark.add_function("Naive", [&]() {
        MatrixMult::CPU::naive_multiply(A, B, C);
    });
    
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
    
    benchmark.add_function("Threaded+SIMD", [&]() {
        MatrixMult::CPU::threaded_simd_multiply(A, B, C);
    });
    
    if (MatrixMult::GPU::check_cuda_available()) {
        benchmark.add_function("CUDA", [&]() {
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
}

int main(int argc, char* argv[]) {
    std::cout << "===========================================" << std::endl;
    std::cout << "High-Performance Parallel Matrix Multiplication" << std::endl;
    std::cout << "===========================================" << std::endl;
    
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
    
    // Verify implementations
    std::cout << "\nVerifying implementations..." << std::endl;
    if (verify_implementations(512)) {
        std::cout << "All implementations produce correct results!" << std::endl;
    } else {
        std::cout << "Warning: Some implementations produce different results!" << std::endl;
    }
    
    // Run benchmarks for various matrix sizes
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048, 4096};
    
    for (auto size : sizes) {
        benchmark_size(size);
    }
    
    return 0;
}
