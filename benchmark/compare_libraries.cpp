#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include "../include/cpu/naive.h"
#include "../include/cpu/blocked.h"
#include "../include/cpu/simd.h"
#include "../include/cpu/threaded.h"
#include "../include/gpu/cuda_wrapper.h"
#include "../include/common/timing.h"
#include "../include/common/matrix.h"
#include "../include/common/utils.h"

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
    float timeOurBest;
    float timeOpenBLAS;
    float timeMKL;
    float timeCuBLAS;
    float gflopsOurBest;
    float gflopsOpenBLAS;
    float gflopsMKL;
    float gflopsCuBLAS;
    float percentageOfOpenBLAS;
    float percentageOfMKL;
    float percentageOfCuBLAS;
};

// Run library comparison benchmarks
std::vector<LibraryComparisonResult> runLibraryComparison(const LibraryComparisonConfig& configInput) {
    // Create a local copy we can modify
    LibraryComparisonConfig config = configInput;
    
    std::vector<LibraryComparisonResult> results;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    // Check if CUDA is available
    bool cudaAvailable = gpu::isCudaAvailable();
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
    
    // Run benchmarks for each matrix size
    for (int size : config.matrixSizes) {
        std::cout << "Benchmarking matrix size: " << size << "x" << size << std::endl;
        
        // Initialize matrices
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C(size * size);
        std::vector<float> C_openblas(size * size);
        std::vector<float> C_mkl(size * size);
        std::vector<float> C_cublas(size * size);
        
        // Fill matrices with random values
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(A.begin(), A.end(), [&]() { return dist(gen); });
        std::generate(B.begin(), B.end(), [&]() { return dist(gen); });
        
        // Initialize CUDA multiplier
        if (cudaAvailable) {
            multiplier.initialize(size, size, size);
            multiplier.setMatrixA(A.data());
            multiplier.setMatrixB(B.data());
        }
        
        // Create timer
        Timer timer;
        
        // Initialize result
        LibraryComparisonResult result;
        result.matrixSize = size;
        
        // Benchmark our best implementation
        if (config.runOurBest) {
            float totalTimeOurBest = 0.0f;
            
            // Warmup
            cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), size, size, size);
                timer.stop();
                totalTimeOurBest += timer.elapsedMilliseconds();
            }
            
            result.timeOurBest = totalTimeOurBest / config.repetitions;
            result.gflopsOurBest = (2.0f * size * size * size) / (result.timeOurBest * 1e-3) / 1e9;
            
            std::cout << "  Our Best: " << std::fixed << std::setprecision(2) 
                      << result.timeOurBest << " ms, " 
                      << result.gflopsOurBest << " GFLOPS" << std::endl;
        } else {
            result.timeOurBest = 0.0f;
            result.gflopsOurBest = 0.0f;
        }
        
        // Benchmark OpenBLAS
        if (config.runOpenBLAS) {
            float totalTimeOpenBLAS = 0.0f;
            
            // Warmup
            // openblas::multiply(A.data(), B.data(), C_openblas.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                // openblas::multiply(A.data(), B.data(), C_openblas.data(), size, size, size);
                timer.stop();
                totalTimeOpenBLAS += timer.elapsedMilliseconds();
            }
            
            result.timeOpenBLAS = totalTimeOpenBLAS / config.repetitions;
            result.gflopsOpenBLAS = (2.0f * size * size * size) / (result.timeOpenBLAS * 1e-3) / 1e9;
            
            std::cout << "  OpenBLAS: " << result.timeOpenBLAS << " ms, " 
                      << result.gflopsOpenBLAS << " GFLOPS" << std::endl;
        } else {
            result.timeOpenBLAS = 0.0f;
            result.gflopsOpenBLAS = 0.0f;
        }
        
        // Benchmark Intel MKL
        if (config.runMKL) {
            float totalTimeMKL = 0.0f;
            
            // Warmup
            // mkl::multiply(A.data(), B.data(), C_mkl.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                // mkl::multiply(A.data(), B.data(), C_mkl.data(), size, size, size);
                timer.stop();
                totalTimeMKL += timer.elapsedMilliseconds();
            }
            
            result.timeMKL = totalTimeMKL / config.repetitions;
            result.gflopsMKL = (2.0f * size * size * size) / (result.timeMKL * 1e-3) / 1e9;
            
            std::cout << "  Intel MKL: " << result.timeMKL << " ms, " 
                      << result.gflopsMKL << " GFLOPS" << std::endl;
        } else {
            result.timeMKL = 0.0f;
            result.gflopsMKL = 0.0f;
        }
        
        // Benchmark cuBLAS
        if (config.runCuBLAS && cudaAvailable) {
            float totalTimeCuBLAS = 0.0f;
            
            // Warmup
            multiplier.multiplyWithCuBLAS(C_cublas.data());
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                multiplier.multiplyWithCuBLAS(C_cublas.data());
                timer.stop();
                totalTimeCuBLAS += timer.elapsedMilliseconds();
            }
            
            result.timeCuBLAS = totalTimeCuBLAS / config.repetitions;
            result.gflopsCuBLAS = (2.0f * size * size * size) / (result.timeCuBLAS * 1e-3) / 1e9;
            
            std::cout << "  cuBLAS: " << result.timeCuBLAS << " ms, " 
                      << result.gflopsCuBLAS << " GFLOPS" << std::endl;
        } else {
            result.timeCuBLAS = 0.0f;
            result.gflopsCuBLAS = 0.0f;
        }
        
        // Calculate percentages
        result.percentageOfOpenBLAS = (result.gflopsOpenBLAS > 0) ? (result.gflopsOurBest / result.gflopsOpenBLAS) * 100.0f : 0.0f;
        result.percentageOfMKL = (result.gflopsMKL > 0) ? (result.gflopsOurBest / result.gflopsMKL) * 100.0f : 0.0f;
        result.percentageOfCuBLAS = (result.gflopsCuBLAS > 0) ? (result.gflopsOurBest / result.gflopsCuBLAS) * 100.0f : 0.0f;
        
        results.push_back(result);
        
        // Cleanup
        if (cudaAvailable) {
            multiplier.cleanup();
        }
    }
    
    // Save results to file if requested
    if (!config.outputFile.empty()) {
        std::ofstream outFile(config.outputFile);
        outFile << "Matrix Size,Our Best (ms),OpenBLAS (ms),Intel MKL (ms),cuBLAS (ms),"
                << "Our Best (GFLOPS),OpenBLAS (GFLOPS),Intel MKL (GFLOPS),cuBLAS (GFLOPS),"
                << "Percentage of OpenBLAS,Percentage of Intel MKL,Percentage of cuBLAS\n";
        
        for (const auto& result : results) {
            outFile << result.matrixSize << ","
                    << result.timeOurBest << ","
                    << result.timeOpenBLAS << ","
                    << result.timeMKL << ","
                    << result.timeCuBLAS << ","
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

int main(int argc, char* argv[]) {
    LibraryComparisonConfig config;
    
    // Default configuration
    config.matrixSizes = {128, 256, 512};
    config.repetitions = 5;
    config.runOurBest = true;
    config.runOpenBLAS = true;
    config.runMKL = true;
    config.runCuBLAS = true;
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