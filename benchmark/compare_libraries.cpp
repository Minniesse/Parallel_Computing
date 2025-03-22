#include "common/matrix.h"
#include "common/timing.h"
#include "cpu/naive.h"
#include "cpu/threaded.h"
#include "gpu/cuda_wrapper.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

// If MKL is available, include MKL headers
#ifdef USE_MKL
#include <mkl.h>
#endif

// Function to perform matrix multiplication using Intel MKL
void mkl_multiply(const Matrix& A, const Matrix& B, Matrix& C) {
#ifdef USE_MKL
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Call MKL SGEMM (single precision general matrix multiply)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A.rows(), B.cols(), A.cols(),
                alpha, A.data(), A.cols(), B.data(), B.cols(),
                beta, C.data(), C.cols());
#else
    // Fallback to our threaded implementation if MKL is not available
    MatrixMult::CPU::threaded_multiply(A, B, C);
    std::cerr << "Warning: MKL not available, using threaded implementation instead." << std::endl;
#endif
}

// Run comparison benchmark across libraries
void run_library_comparison(const std::vector<size_t>& sizes, int iterations = 3) {
    std::cout << "===========================================" << std::endl;
    std::cout << "Matrix Multiplication Library Comparison" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Create output CSV file
    std::ofstream csv_file("library_comparison.csv");
    if (csv_file.is_open()) {
        csv_file << "Size,Library,AvgTime,MinTime,AvgGFLOPS,PeakGFLOPS" << std::endl;
    }
    
    // Print system information
    std::cout << "\nSystem Information:" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "CPU: " << std::thread::hardware_concurrency() << " logical cores" << std::endl;
    std::cout << "AVX-512 Support: " << (MatrixMult::CPU::is_avx512_supported() ? "Yes" : "No") << std::endl;
    
    bool cuda_available = MatrixMult::GPU::check_cuda_available();
    if (cuda_available) {
        MatrixMult::GPU::print_cuda_device_info();
    } else {
        std::cout << "CUDA: Not available" << std::endl;
    }
    
#ifdef USE_MKL
    std::cout << "Intel MKL: Available (Version " << mkl_get_version_string() << ")" << std::endl;
#else
    std::cout << "Intel MKL: Not available" << std::endl;
#endif
    
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
        
        // Add our best CPU implementation
        benchmark.add_function("Our Threaded+SIMD", [&]() {
            MatrixMult::CPU::threaded_simd_multiply(A, B, C);
        });
        
        // Add MKL implementation
#ifdef USE_MKL
        benchmark.add_function("Intel MKL", [&]() {
            mkl_multiply(A, B, C);
        });
#endif
        
        // Add cuBLAS implementation
        if (cuda_available) {
            benchmark.add_function("NVIDIA cuBLAS", [&]() {
                MatrixMult::GPU::cublas_multiply(A, B, C);
            });
        }
        
        // Add our adaptive implementation
        benchmark.add_function("Our Adaptive", [&]() {
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
        std::cout << "\nLibrary comparison results saved to library_comparison.csv" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default matrix sizes to benchmark
    std::vector<size_t> sizes = {512, 1024, 2048, 4096};
    
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
    
    run_library_comparison(sizes, iterations);
    
    return 0;
}
