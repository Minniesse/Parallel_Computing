#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <omp.h>
#include "../include/cpu/naive.h"
#include "../include/cpu/blocked.h"
#include "../include/cpu/simd.h"
#include "../include/cpu/threaded.h"
#include "../include/gpu/cuda_wrapper.h"
#include "../include/common/timing.h"
#include "../include/common/matrix.h"
#include "../include/common/utils.h"

// Combined benchmark configuration
struct AllBenchmarkConfig {
    std::vector<int> matrixSizes;
    int repetitions;
    bool runCPU;
    bool runGPU;
    bool runNaive;
    bool runBlocked;
    bool runSIMD;
    bool runThreaded;
    bool runCombinedCPU;
    bool runCUDA;
    bool runCuBLAS;
    int numThreads;
    std::string outputFile;
};

// Combined benchmark results
struct AllBenchmarkResult {
    int matrixSize;
    float timeNaive;
    float timeBlocked;
    float timeSIMD;
    float timeThreaded;
    float timeCombinedCPU;
    float timeCUDA;
    float timeCuBLAS;
    float gflopsNaive;
    float gflopsBlocked;
    float gflopsSIMD;
    float gflopsThreaded;
    float gflopsCombinedCPU;
    float gflopsCUDA;
    float gflopsCuBLAS;
    float bestCPUTime;
    float bestGPUTime;
    float speedupGPUvsCPU;
};

// Run all benchmarks
std::vector<AllBenchmarkResult> runAllBenchmarks(const AllBenchmarkConfig& config) {
    std::vector<AllBenchmarkResult> results;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    // Check if CUDA is available
    bool cudaAvailable = false;
    if (config.runGPU) {
        cudaAvailable = gpu::isCudaAvailable();
        if (!cudaAvailable) {
            std::cerr << "CUDA is not available on this system. Skipping GPU benchmarks." << std::endl;
        } else {
            gpu::printDeviceProperties();
        }
    }
    
    // Set number of threads
    if (config.numThreads > 0) {
        omp_set_num_threads(config.numThreads);
    }
    
    std::cout << "Running benchmarks with " << omp_get_max_threads() << " CPU threads" << std::endl;
    
    // Create CUDA multiplier if GPU benchmarks are enabled
    gpu::CudaMatrixMultiplier multiplier;
    
    // Run benchmarks for each matrix size
    for (int size : config.matrixSizes) {
        std::cout << "Benchmarking matrix size: " << size << "x" << size << std::endl;
        
        // Initialize matrices
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C(size * size, 0.0f);
        
        // Fill matrices with random values
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(A.begin(), A.end(), [&]() { return dist(gen); });
        std::generate(B.begin(), B.end(), [&]() { return dist(gen); });
        
        // Create timer
        Timer timer;
        
        // Initialize result
        AllBenchmarkResult result;
        result.matrixSize = size;
        
        // CPU benchmarks
        if (config.runCPU) {
            // Benchmark naive implementation
            if (config.runNaive && size <= 2048) {  // Skip naive for large matrices
                float totalTimeNaive = 0.0f;
                
                // Warmup
                cpu::naive::multiply(A.data(), B.data(), C.data(), size, size, size);
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    cpu::naive::multiply(A.data(), B.data(), C.data(), size, size, size);
                    timer.stop();
                    totalTimeNaive += timer.elapsedMilliseconds();
                }
                
                result.timeNaive = totalTimeNaive / config.repetitions;
                result.gflopsNaive = (2.0f * size * size * size) / (result.timeNaive * 1e-3) / 1e9;
                
                std::cout << "  CPU Naive: " << std::fixed << std::setprecision(2) 
                          << result.timeNaive << " ms, " 
                          << result.gflopsNaive << " GFLOPS" << std::endl;
            } else {
                result.timeNaive = 0.0f;
                result.gflopsNaive = 0.0f;
            }
            
            // Benchmark cache-blocked implementation
            if (config.runBlocked) {
                float totalTimeBlocked = 0.0f;
                
                // Warmup
                cpu::blocked::multiply(A.data(), B.data(), C.data(), size, size, size);
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    cpu::blocked::multiply(A.data(), B.data(), C.data(), size, size, size);
                    timer.stop();
                    totalTimeBlocked += timer.elapsedMilliseconds();
                }
                
                result.timeBlocked = totalTimeBlocked / config.repetitions;
                result.gflopsBlocked = (2.0f * size * size * size) / (result.timeBlocked * 1e-3) / 1e9;
                
                std::cout << "  CPU Blocked: " << result.timeBlocked << " ms, " 
                          << result.gflopsBlocked << " GFLOPS" << std::endl;
            } else {
                result.timeBlocked = 0.0f;
                result.gflopsBlocked = 0.0f;
            }
            
            // Benchmark SIMD implementation
            if (config.runSIMD) {
                float totalTimeSIMD = 0.0f;
                
                // Warmup
                cpu::simd::multiply_avx512(A.data(), B.data(), C.data(), size, size, size);
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    cpu::simd::multiply_avx512(A.data(), B.data(), C.data(), size, size, size);
                    timer.stop();
                    totalTimeSIMD += timer.elapsedMilliseconds();
                }
                
                result.timeSIMD = totalTimeSIMD / config.repetitions;
                result.gflopsSIMD = (2.0f * size * size * size) / (result.timeSIMD * 1e-3) / 1e9;
                
                std::cout << "  CPU SIMD: " << result.timeSIMD << " ms, " 
                          << result.gflopsSIMD << " GFLOPS" << std::endl;
            } else {
                result.timeSIMD = 0.0f;
                result.gflopsSIMD = 0.0f;
            }
            
            // Benchmark multithreaded implementation
            if (config.runThreaded) {
                float totalTimeThreaded = 0.0f;
                
                // Warmup
                cpu::threaded::multiply(A.data(), B.data(), C.data(), size, size, size);
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    cpu::threaded::multiply(A.data(), B.data(), C.data(), size, size, size);
                    timer.stop();
                    totalTimeThreaded += timer.elapsedMilliseconds();
                }
                
                result.timeThreaded = totalTimeThreaded / config.repetitions;
                result.gflopsThreaded = (2.0f * size * size * size) / (result.timeThreaded * 1e-3) / 1e9;
                
                std::cout << "  CPU Threaded: " << result.timeThreaded << " ms, " 
                          << result.gflopsThreaded << " GFLOPS" << std::endl;
            } else {
                result.timeThreaded = 0.0f;
                result.gflopsThreaded = 0.0f;
            }
            
            // Benchmark combined (SIMD + multithreaded) implementation
            if (config.runCombinedCPU) {
                float totalTimeCombinedCPU = 0.0f;
                
                // Warmup
                cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), size, size, size);
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), size, size, size);
                    timer.stop();
                    totalTimeCombinedCPU += timer.elapsedMilliseconds();
                }
                
                result.timeCombinedCPU = totalTimeCombinedCPU / config.repetitions;
                result.gflopsCombinedCPU = (2.0f * size * size * size) / (result.timeCombinedCPU * 1e-3) / 1e9;
                
                std::cout << "  CPU Combined: " << result.timeCombinedCPU << " ms, " 
                          << result.gflopsCombinedCPU << " GFLOPS" << std::endl;
            } else {
                result.timeCombinedCPU = 0.0f;
                result.gflopsCombinedCPU = 0.0f;
            }
        } else {
            // Skip all CPU benchmarks
            result.timeNaive = 0.0f;
            result.timeBlocked = 0.0f;
            result.timeSIMD = 0.0f;
            result.timeThreaded = 0.0f;
            result.timeCombinedCPU = 0.0f;
            result.gflopsNaive = 0.0f;
            result.gflopsBlocked = 0.0f;
            result.gflopsSIMD = 0.0f;
            result.gflopsThreaded = 0.0f;
            result.gflopsCombinedCPU = 0.0f;
        }
        
        // GPU benchmarks
        if (config.runGPU && cudaAvailable) {
            // Initialize CUDA multiplier
            multiplier.initialize(size, size, size);
            multiplier.setMatrixA(A.data());
            multiplier.setMatrixB(B.data());
            
            // Benchmark our CUDA implementation
            if (config.runCUDA) {
                float totalTimeCUDA = 0.0f;
                
                // Warmup
                multiplier.multiply(C.data());
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    multiplier.multiply(C.data());
                    timer.stop();
                    totalTimeCUDA += timer.elapsedMilliseconds();
                }
                
                result.timeCUDA = totalTimeCUDA / config.repetitions;
                result.gflopsCUDA = (2.0f * size * size * size) / (result.timeCUDA * 1e-3) / 1e9;
                
                std::cout << "  GPU CUDA: " << result.timeCUDA << " ms, " 
                          << result.gflopsCUDA << " GFLOPS" << std::endl;
            } else {
                result.timeCUDA = 0.0f;
                result.gflopsCUDA = 0.0f;
            }
            
            // Benchmark cuBLAS
            if (config.runCuBLAS) {
                float totalTimeCuBLAS = 0.0f;
                
                // Warmup
                multiplier.multiplyWithCuBLAS(C.data());
                
                for (int i = 0; i < config.repetitions; i++) {
                    timer.start();
                    multiplier.multiplyWithCuBLAS(C.data());
                    timer.stop();
                    totalTimeCuBLAS += timer.elapsedMilliseconds();
                }
                
                result.timeCuBLAS = totalTimeCuBLAS / config.repetitions;
                result.gflopsCuBLAS = (2.0f * size * size * size) / (result.timeCuBLAS * 1e-3) / 1e9;
                
                std::cout << "  GPU cuBLAS: " << result.timeCuBLAS << " ms, " 
                          << result.gflopsCuBLAS << " GFLOPS" << std::endl;
            } else {
                result.timeCuBLAS = 0.0f;
                result.gflopsCuBLAS = 0.0f;
            }
            
            // Cleanup
            multiplier.cleanup();
        } else {
            // Skip all GPU benchmarks
            result.timeCUDA = 0.0f;
            result.timeCuBLAS = 0.0f;
            result.gflopsCUDA = 0.0f;
            result.gflopsCuBLAS = 0.0f;
        }
        
        // Find best CPU and GPU times
        result.bestCPUTime = std::min({
            result.timeNaive > 0 ? result.timeNaive : std::numeric_limits<float>::max(),
            result.timeBlocked > 0 ? result.timeBlocked : std::numeric_limits<float>::max(),
            result.timeSIMD > 0 ? result.timeSIMD : std::numeric_limits<float>::max(),
            result.timeThreaded > 0 ? result.timeThreaded : std::numeric_limits<float>::max(),
            result.timeCombinedCPU > 0 ? result.timeCombinedCPU : std::numeric_limits<float>::max()
        });
        
        if (result.bestCPUTime == std::numeric_limits<float>::max()) {
            result.bestCPUTime = 0.0f;
        }
        
        result.bestGPUTime = std::min({
            result.timeCUDA > 0 ? result.timeCUDA : std::numeric_limits<float>::max(),
            result.timeCuBLAS > 0 ? result.timeCuBLAS : std::numeric_limits<float>::max()
        });
        
        if (result.bestGPUTime == std::numeric_limits<float>::max()) {
            result.bestGPUTime = 0.0f;
        }
        
        // Calculate speedup of GPU over CPU
        if (result.bestCPUTime > 0 && result.bestGPUTime > 0) {
            result.speedupGPUvsCPU = result.bestCPUTime / result.bestGPUTime;
            std::cout << "  GPU vs CPU Speedup: " << result.speedupGPUvsCPU << "x" << std::endl;
        } else {
            result.speedupGPUvsCPU = 0.0f;
        }
        
        results.push_back(result);
    }
    
    // Save results to file if requested
    if (!config.outputFile.empty()) {
        std::ofstream outFile(config.outputFile);
        outFile << "Matrix Size,"
                << "CPU Naive (ms),CPU Blocked (ms),CPU SIMD (ms),CPU Threaded (ms),CPU Combined (ms),"
                << "GPU CUDA (ms),GPU cuBLAS (ms),"
                << "CPU Naive (GFLOPS),CPU Blocked (GFLOPS),CPU SIMD (GFLOPS),CPU Threaded (GFLOPS),CPU Combined (GFLOPS),"
                << "GPU CUDA (GFLOPS),GPU cuBLAS (GFLOPS),"
                << "Best CPU (ms),Best GPU (ms),GPU vs CPU Speedup\n";
        
        for (const auto& result : results) {
            outFile << result.matrixSize << ","
                    << result.timeNaive << ","
                    << result.timeBlocked << ","
                    << result.timeSIMD << ","
                    << result.timeThreaded << ","
                    << result.timeCombinedCPU << ","
                    << result.timeCUDA << ","
                    << result.timeCuBLAS << ","
                    << result.gflopsNaive << ","
                    << result.gflopsBlocked << ","
                    << result.gflopsSIMD << ","
                    << result.gflopsThreaded << ","
                    << result.gflopsCombinedCPU << ","
                    << result.gflopsCUDA << ","
                    << result.gflopsCuBLAS << ","
                    << result.bestCPUTime << ","
                    << result.bestGPUTime << ","
                    << result.speedupGPUvsCPU << "\n";
        }
    }
    
    return results;
}

int main(int argc, char* argv[]) {
    AllBenchmarkConfig config;
    
    // Default configuration
    config.matrixSizes = {128, 256, 512, 1024};
    config.repetitions = 5;
    config.runCPU = true;
    config.runGPU = true;
    config.runNaive = true;
    config.runBlocked = true;
    config.runSIMD = true;
    config.runThreaded = true;
    config.runCombinedCPU = true;
    config.runCUDA = true;
    config.runCuBLAS = true;
    config.numThreads = omp_get_max_threads();
    config.outputFile = "all_benchmark_results.csv";
    
    // Parse command line arguments
    // ...
    
    // Run benchmarks
    std::vector<AllBenchmarkResult> results = runAllBenchmarks(config);
    
    return 0;
}
