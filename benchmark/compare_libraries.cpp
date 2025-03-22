#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>
#include "../include/cpu/naive.h"
#include "../include/cpu/blocked.h"
#include "../include/cpu/simd.h"
#include "../include/cpu/threaded.h"
#include "../include/common/timing.h"
#include "../include/common/matrix.h"
#include "../include/common/utils.h"

// Include GPU header if CUDA is available
#ifdef HAS_CUDA
#include "../include/gpu/cuda_wrapper.h"
#endif

// Include OpenBLAS if available
#ifdef HAS_OPENBLAS
#include <cblas.h>
#endif

// Include Intel MKL if available
#ifdef HAS_MKL
#include <mkl.h>
#endif

// Benchmark configuration
struct LibraryComparisonConfig {
    std::vector<int> matrixSizes;
    int repetitions;
    bool runOurBest;
    bool runOpenBLAS;
    bool runMKL;
    bool runCuBLAS;
    std::string outputFile;
};

// Benchmark results
struct LibraryComparisonResult {
    int matrixSize;
    double timeOurBest;
    double timeOpenBLAS;
    double timeMKL;
    double timeCuBLAS;
    double gflopsOurBest;
    double gflopsOpenBLAS;
    double gflopsMKL;
    double gflopsCuBLAS;
    double percentageOfOpenBLAS;
    double percentageOfMKL;
    double percentageOfCuBLAS;
};

// Function prototypes for library wrappers
#ifdef HAS_OPENBLAS
void multiply_openblas(const float* A, const float* B, float* C, int m, int n, int k);
#endif

#ifdef HAS_MKL
void multiply_mkl(const float* A, const float* B, float* C, int m, int n, int k);
#endif

// Run library comparison benchmarks
std::vector<LibraryComparisonResult> runLibraryComparison(const LibraryComparisonConfig& configInput) {
    // Create a local copy we can modify
    LibraryComparisonConfig config = configInput;
    
    std::vector<LibraryComparisonResult> results;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    // Check if CUDA is available
    bool cudaAvailable = false;
    #ifdef HAS_CUDA
    cudaAvailable = gpu::isCudaAvailable();
    if (!cudaAvailable && config.runCuBLAS) {
        std::cerr << "CUDA is not available on this system. Skipping cuBLAS benchmarks." << std::endl;
        config.runCuBLAS = false;
    }
    
    // Print device information
    if (cudaAvailable) {
        gpu::printDeviceProperties();
    }
    
    // Create CUDA multiplier
    gpu::CudaMatrixMultiplier multiplier;
    #endif
    
    // Run benchmarks for each matrix size
    for (int size : config.matrixSizes) {
        std::cout << "Benchmarking matrix size: " << size << "x" << size << std::endl;
        
        // Initialize matrices
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C(size * size, 0.0f);
        std::vector<float> C_openblas(size * size, 0.0f);
        std::vector<float> C_mkl(size * size, 0.0f);
        std::vector<float> C_cublas(size * size, 0.0f);
        
        // Fill matrices with random values
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(A.begin(), A.end(), [&]() { return dist(gen); });
        std::generate(B.begin(), B.end(), [&]() { return dist(gen); });
        
        // Initialize CUDA multiplier if available
        #ifdef HAS_CUDA
        if (cudaAvailable && config.runCuBLAS) {
            multiplier.initialize(size, size, size);
            multiplier.setMatrixA(A.data());
            multiplier.setMatrixB(B.data());
        }
        #endif
        
        // Create timer
        Timer timer;
        
        // Initialize result
        LibraryComparisonResult result;
        result.matrixSize = size;
        
        // Calculate FLOPS for this matrix size
        double flops = 2.0 * size * size * size; // 2*n^3 operations for matrix multiplication
        
        // Benchmark our best implementation
        if (config.runOurBest) {
            double totalTimeOurBest = 0.0;
            
            // Warmup
            cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), size, size, size);
                timer.stop();
                totalTimeOurBest += timer.elapsedSeconds();
            }
            
            result.timeOurBest = totalTimeOurBest / config.repetitions;
            result.gflopsOurBest = flops / (result.timeOurBest * 1e9);
            
            std::cout << "  Our Best: " << std::fixed << std::setprecision(4) 
                      << result.timeOurBest * 1000 << " ms, " 
                      << std::setprecision(2) << result.gflopsOurBest << " GFLOPS" << std::endl;
        } else {
            result.timeOurBest = 0.0;
            result.gflopsOurBest = 0.0;
        }
        
        // Benchmark OpenBLAS
        #ifdef HAS_OPENBLAS
        if (config.runOpenBLAS) {
            double totalTimeOpenBLAS = 0.0;
            
            // Warmup
            multiply_openblas(A.data(), B.data(), C_openblas.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                multiply_openblas(A.data(), B.data(), C_openblas.data(), size, size, size);
                timer.stop();
                totalTimeOpenBLAS += timer.elapsedSeconds();
            }
            
            result.timeOpenBLAS = totalTimeOpenBLAS / config.repetitions;
            result.gflopsOpenBLAS = flops / (result.timeOpenBLAS * 1e9);
            
            std::cout << "  OpenBLAS: " << std::fixed << std::setprecision(4) 
                      << result.timeOpenBLAS * 1000 << " ms, " 
                      << std::setprecision(2) << result.gflopsOpenBLAS << " GFLOPS" << std::endl;
        } else {
            result.timeOpenBLAS = 0.0;
            result.gflopsOpenBLAS = 0.0;
        }
        #else
        result.timeOpenBLAS = 0.0;
        result.gflopsOpenBLAS = 0.0;
        #endif
        
        // Benchmark Intel MKL
        #ifdef HAS_MKL
        if (config.runMKL) {
            double totalTimeMKL = 0.0;
            
            // Warmup
            multiply_mkl(A.data(), B.data(), C_mkl.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                multiply_mkl(A.data(), B.data(), C_mkl.data(), size, size, size);
                timer.stop();
                totalTimeMKL += timer.elapsedSeconds();
            }
            
            result.timeMKL = totalTimeMKL / config.repetitions;
            result.gflopsMKL = flops / (result.timeMKL * 1e9);
            
            std::cout << "  Intel MKL: " << std::fixed << std::setprecision(4) 
                      << result.timeMKL * 1000 << " ms, " 
                      << std::setprecision(2) << result.gflopsMKL << " GFLOPS" << std::endl;
        } else {
            result.timeMKL = 0.0;
            result.gflopsMKL = 0.0;
        }
        #else
        result.timeMKL = 0.0;
        result.gflopsMKL = 0.0;
        #endif
        
        // Benchmark cuBLAS
        #ifdef HAS_CUDA
        if (config.runCuBLAS && cudaAvailable) {
            double totalTimeCuBLAS = 0.0;
            
            // Warmup
            multiplier.multiplyWithCuBLAS(C_cublas.data());
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                multiplier.multiplyWithCuBLAS(C_cublas.data());
                timer.stop();
                totalTimeCuBLAS += timer.elapsedSeconds();
            }
            
            result.timeCuBLAS = totalTimeCuBLAS / config.repetitions;
            result.gflopsCuBLAS = flops / (result.timeCuBLAS * 1e9);
            
            std::cout << "  cuBLAS: " << std::fixed << std::setprecision(4) 
                      << result.timeCuBLAS * 1000 << " ms, " 
                      << std::setprecision(2) << result.gflopsCuBLAS << " GFLOPS" << std::endl;
        } else {
            result.timeCuBLAS = 0.0;
            result.gflopsCuBLAS = 0.0;
        }
        #else
        result.timeCuBLAS = 0.0;
        result.gflopsCuBLAS = 0.0;
        #endif
        
        // Calculate percentages
        result.percentageOfOpenBLAS = (result.gflopsOpenBLAS > 0) ? 
            (result.gflopsOurBest / result.gflopsOpenBLAS) * 100.0 : 0.0;
        result.percentageOfMKL = (result.gflopsMKL > 0) ? 
            (result.gflopsOurBest / result.gflopsMKL) * 100.0 : 0.0;
        result.percentageOfCuBLAS = (result.gflopsCuBLAS > 0) ? 
            (result.gflopsOurBest / result.gflopsCuBLAS) * 100.0 : 0.0;
        
        results.push_back(result);
        
        // Cleanup
        #ifdef HAS_CUDA
        if (cudaAvailable && config.runCuBLAS) {
            multiplier.cleanup();
        }
        #endif
    }
    
    // Save results to file if requested
    if (!config.outputFile.empty()) {
        std::ofstream outFile(config.outputFile);
        outFile << "Matrix Size,Our Best (ms),OpenBLAS (ms),Intel MKL (ms),cuBLAS (ms),"
                << "Our Best (GFLOPS),OpenBLAS (GFLOPS),Intel MKL (GFLOPS),cuBLAS (GFLOPS),"
                << "Percentage of OpenBLAS,Percentage of Intel MKL,Percentage of cuBLAS\n";
        
        for (const auto& result : results) {
            outFile << result.matrixSize << ","
                    << result.timeOurBest * 1000 << ","
                    << result.timeOpenBLAS * 1000 << ","
                    << result.timeMKL * 1000 << ","
                    << result.timeCuBLAS * 1000 << ","
                    << result.gflopsOurBest << ","
                    << result.gflopsOpenBLAS << ","
                    << result.gflopsMKL << ","
                    << result.gflopsCuBLAS << ","
                    << result.percentageOfOpenBLAS << ","
                    << result.percentageOfMKL << ","
                    << result.percentageOfCuBLAS << "\n";
        }
    }
    
    return results;
}

// Implementation of OpenBLAS wrapper (if available)
#ifdef HAS_OPENBLAS
void multiply_openblas(const float* A, const float* B, float* C, int m, int n, int k) {
    // Set all elements of C to zero
    std::fill(C, C + m * n, 0.0f);
    
    // Using SGEMM: C = alpha * A * B + beta * C
    // Note: OpenBLAS uses column-major order, so we compute B * A instead
    // and interpret the result in column-major order (which gives us A * B in row-major)
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Call OpenBLAS SGEMM
    cblas_sgemm(CblasRowMajor,  // Row-major order
                CblasNoTrans,    // A is not transposed
                CblasNoTrans,    // B is not transposed
                m,               // Rows of A and C
                n,               // Columns of B and C
                k,               // Columns of A and rows of B
                alpha,           // Alpha multiplier
                A,               // Matrix A
                k,               // Leading dimension of A
                B,               // Matrix B
                n,               // Leading dimension of B
                beta,            // Beta multiplier
                C,               // Result matrix C
                n);              // Leading dimension of C
}
#endif

// Implementation of Intel MKL wrapper (if available)
#ifdef HAS_MKL
void multiply_mkl(const float* A, const float* B, float* C, int m, int n, int k) {
    // Set all elements of C to zero
    std::fill(C, C + m * n, 0.0f);
    
    // Using MKL SGEMM: C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Call MKL SGEMM
    cblas_sgemm(CblasRowMajor,  // Row-major order
                CblasNoTrans,    // A is not transposed
                CblasNoTrans,    // B is not transposed
                m,               // Rows of A and C
                n,               // Columns of B and C
                k,               // Columns of A and rows of B
                alpha,           // Alpha multiplier
                A,               // Matrix A
                k,               // Leading dimension of A
                B,               // Matrix B
                n,               // Leading dimension of B
                beta,            // Beta multiplier
                C,               // Result matrix C
                n);              // Leading dimension of C
}
#endif

int main(int argc, char* argv[]) {
    LibraryComparisonConfig config;
    
    // Default configuration
    config.matrixSizes = {128, 256, 512, 1024, 2048, 4096};
    config.repetitions = 5;
    config.runOurBest = true;
    
    #ifdef HAS_OPENBLAS
    config.runOpenBLAS = true;
    #else
    config.runOpenBLAS = false;
    #endif
    
    #ifdef HAS_MKL
    config.runMKL = true;
    #else
    config.runMKL = false;
    #endif
    
    #ifdef HAS_CUDA
    config.runCuBLAS = true;
    #else
    config.runCuBLAS = false;
    #endif
    
    config.outputFile = "library_comparison_results.csv";
    
    // Command line argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--sizes" && i + 1 < argc) {
            config.matrixSizes.clear();
            std::string sizes_str = argv[++i];
            std::stringstream ss(sizes_str);
            std::string size;
            
            while (std::getline(ss, size, ',')) {
                config.matrixSizes.push_back(std::stoi(size));
            }
        } 
        else if (arg == "--reps" && i + 1 < argc) {
            config.repetitions = std::stoi(argv[++i]);
        }
        else if (arg == "--no-ourbest") {
            config.runOurBest = false;
        }
        else if (arg == "--no-openblas") {
            config.runOpenBLAS = false;
        }
        else if (arg == "--no-mkl") {
            config.runMKL = false;
        }
        else if (arg == "--no-cublas") {
            config.runCuBLAS = false;
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.outputFile = argv[++i];
        }
        else if (arg == "--help") {
            std::cout << "Library Comparison Benchmark\n"
                      << "Options:\n"
                      << "  --sizes N1,N2,...   Comma-separated list of matrix sizes to test\n"
                      << "  --reps N            Number of repetitions for each benchmark\n"
                      << "  --no-ourbest        Skip our best implementation\n"
                      << "  --no-openblas       Skip OpenBLAS benchmarks\n"
                      << "  --no-mkl            Skip Intel MKL benchmarks\n"
                      << "  --no-cublas         Skip cuBLAS benchmarks\n"
                      << "  --output FILE       Output file for results (CSV format)\n"
                      << "  --help              Display this help message\n";
            return 0;
        }
    }
    
    // Run benchmarks
    std::vector<LibraryComparisonResult> results = runLibraryComparison(config);
    
    // Print summary of results
    std::cout << "\nSummary of Results:" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Print table header
    std::cout << std::setw(10) << "Size" << " | "
              << std::setw(15) << "Our Best" << " | "
              << std::setw(15) << "OpenBLAS" << " | "
              << std::setw(15) << "Intel MKL" << " | "
              << std::setw(15) << "cuBLAS" << std::endl;
    
    std::cout << std::string(10 + 15*4 + 9, '-') << std::endl;
    
    // Print performance in GFLOPS for each implementation and matrix size
    for (const auto& result : results) {
        std::cout << std::setw(10) << result.matrixSize << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gflopsOurBest << " | "
                  << std::setw(15) << result.gflopsOpenBLAS << " | "
                  << std::setw(15) << result.gflopsMKL << " | "
                  << std::setw(15) << result.gflopsCuBLAS << std::endl;
    }
    
    std::cout << "\nPercentage of Our Implementation Relative to Libraries:" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Print table header for percentages
    std::cout << std::setw(10) << "Size" << " | "
              << std::setw(15) << "vs OpenBLAS" << " | "
              << std::setw(15) << "vs Intel MKL" << " | "
              << std::setw(15) << "vs cuBLAS" << std::endl;
    
    std::cout << std::string(10 + 15*3 + 7, '-') << std::endl;
    
    // Print percentages for each matrix size
    for (const auto& result : results) {
        std::cout << std::setw(10) << result.matrixSize << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.percentageOfOpenBLAS << "% | "
                  << std::setw(15) << result.percentageOfMKL << "% | "
                  << std::setw(15) << result.percentageOfCuBLAS << "%" << std::endl;
    }
    
    // Print output file location
    if (!config.outputFile.empty()) {
        std::cout << "\nDetailed results saved to: " << config.outputFile << std::endl;
    }
    
    return 0;
}