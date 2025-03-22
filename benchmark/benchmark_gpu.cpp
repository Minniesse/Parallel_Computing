#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>
#include "../include/gpu/cuda_wrapper.h"
#include "../include/common/timing.h"
#include "../include/common/matrix.h"
#include "../include/common/utils.h"

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<int> matrixSizes;
    int repetitions;
    bool compareToCuBLAS;
    bool useTensorCores;
    std::string outputFile;
};

// Benchmark results
struct BenchmarkResult {
    int matrixSize;
    float timeCustom;  // Our CUDA implementation
    float timeCuBLAS;  // cuBLAS implementation
    float timeTensorCores; // Tensor cores implementation
    float gflopsCustom;
    float gflopsCuBLAS;
    float gflopsTensorCores;
    float speedupVsCuBLAS;
    bool tensorCoresAvailable;
};

// Run GPU benchmarks
std::vector<BenchmarkResult> runGpuBenchmarks(const BenchmarkConfig& config) {
    std::vector<BenchmarkResult> results;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    // Check if CUDA is available
    if (!gpu::isCudaAvailable()) {
        std::cerr << "CUDA is not available on this system. Skipping GPU benchmarks." << std::endl;
        return results;
    }
    
    // Print device information
    gpu::printDeviceProperties();
    
    // Create CUDA multiplier
    gpu::CudaMatrixMultiplier multiplier;
    
    // Run benchmarks for each matrix size
    for (int size : config.matrixSizes) {
        std::cout << "Benchmarking matrix size: " << size << "x" << size << std::endl;
        
        // Initialize matrices
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C(size * size);
        std::vector<float> C_cublas(size * size);
        std::vector<float> C_tensor(size * size);
        
        // Fill matrices with random values
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(A.begin(), A.end(), [&]() { return dist(gen); });
        std::generate(B.begin(), B.end(), [&]() { return dist(gen); });
        
        // Initialize CUDA multiplier
        multiplier.initialize(size, size, size);
        multiplier.setMatrixA(A.data());
        multiplier.setMatrixB(B.data());
        
        // Warmup
        multiplier.multiply(C.data());
        
        // Benchmark our CUDA implementation
        Timer timer;
        float totalTimeCustom = 0.0f;
        
        for (int i = 0; i < config.repetitions; i++) {
            timer.start();
            multiplier.multiply(C.data());
            timer.stop();
            totalTimeCustom += timer.elapsedMilliseconds();
        }
        
        float avgTimeCustom = totalTimeCustom / config.repetitions;
        float gflopsCustom = (2.0f * size * size * size) / (avgTimeCustom * 1e-3) / 1e9;
        
        // Benchmark cuBLAS
        float avgTimeCuBLAS = 0.0f;
        float gflopsCuBLAS = 0.0f;
        
        if (config.compareToCuBLAS) {
            float totalTimeCuBLAS = 0.0f;
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                multiplier.multiplyWithCuBLAS(C_cublas.data());
                timer.stop();
                totalTimeCuBLAS += timer.elapsedMilliseconds();
            }
            
            avgTimeCuBLAS = totalTimeCuBLAS / config.repetitions;
            gflopsCuBLAS = (2.0f * size * size * size) / (avgTimeCuBLAS * 1e-3) / 1e9;
        }
        
        // Benchmark tensor cores if available
        float avgTimeTensorCores = 0.0f;
        float gflopsTensorCores = 0.0f;
        bool tensorCoresAvailable = false;
        
        if (config.useTensorCores) {
            // Try to use tensor cores
            tensorCoresAvailable = multiplier.multiplyWithTensorCores(C_tensor.data());
            
            if (tensorCoresAvailable) {
                float totalTimeTensorCores = 0.0f;
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    multiplier.multiplyWithTensorCores(C_tensor.data());
                    timer.stop();
                    totalTimeTensorCores += timer.elapsedMilliseconds();
                }
                
                avgTimeTensorCores = totalTimeTensorCores / config.repetitions;
                gflopsTensorCores = (2.0f * size * size * size) / (avgTimeTensorCores * 1e-3) / 1e9;
            }
        }
        
        // Calculate speedup relative to cuBLAS
        float speedupVsCuBLAS = (avgTimeCuBLAS > 0) ? avgTimeCuBLAS / avgTimeCustom : 0.0f;
        
        // Store results
        BenchmarkResult result;
        result.matrixSize = size;
        result.timeCustom = avgTimeCustom;
        result.timeCuBLAS = avgTimeCuBLAS;
        result.timeTensorCores = avgTimeTensorCores;
        result.gflopsCustom = gflopsCustom;
        result.gflopsCuBLAS = gflopsCuBLAS;
        result.gflopsTensorCores = gflopsTensorCores;
        result.speedupVsCuBLAS = speedupVsCuBLAS;
        result.tensorCoresAvailable = tensorCoresAvailable;
        
        results.push_back(result);
        
        // Cleanup
        multiplier.cleanup();
        
        std::cout << "  Custom CUDA: " << std::fixed << std::setprecision(2) << avgTimeCustom << " ms, " 
                  << gflopsCustom << " GFLOPS" << std::endl;
        
        if (config.compareToCuBLAS) {
            std::cout << "  cuBLAS: " << avgTimeCuBLAS << " ms, " 
                      << gflopsCuBLAS << " GFLOPS" << std::endl;
            std::cout << "  Speedup vs cuBLAS: " << speedupVsCuBLAS << "x" << std::endl;
        }
        
        if (tensorCoresAvailable) {
            std::cout << "  Tensor Cores: " << avgTimeTensorCores << " ms, " 
                      << gflopsTensorCores << " GFLOPS" << std::endl;
        }
    }
    
    // Save results to file if requested
    if (!config.outputFile.empty()) {
        std::ofstream outFile(config.outputFile);
        outFile << "Matrix Size,Custom CUDA (ms),cuBLAS (ms),Tensor Cores (ms),Custom CUDA (GFLOPS),cuBLAS (GFLOPS),Tensor Cores (GFLOPS),Speedup vs cuBLAS\n";
        
        for (const auto& result : results) {
            outFile << result.matrixSize << ","
                    << result.timeCustom << ","
                    << result.timeCuBLAS << ","
                    << result.timeTensorCores << ","
                    << result.gflopsCustom << ","
                    << result.gflopsCuBLAS << ","
                    << result.gflopsTensorCores << ","
                    << result.speedupVsCuBLAS << "\n";
        }
    }
    
    return results;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    // Default configuration
    config.matrixSizes = {128, 256, 512, 1024};
    config.repetitions = 10;
    config.compareToCuBLAS = true;
    config.useTensorCores = true;
    config.outputFile = "gpu_benchmark_results.csv";
    
    // Parse command line arguments
    // ...
    
    // Run benchmarks
    std::vector<BenchmarkResult> results = runGpuBenchmarks(config);
    
    return 0;
}
