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
#include "../include/common/timing.h"
#include "../include/common/matrix.h"
#include "../include/common/utils.h"

// Benchmark configuration
struct CPUBenchmarkConfig {
    std::vector<int> matrixSizes;
    int repetitions;
    bool runNaive;
    bool runBlocked;
    bool runSIMD;
    bool runThreaded;
    bool runCombined;
    int numThreads;
    std::string outputFile;
};

// Benchmark results
struct CPUBenchmarkResult {
    int matrixSize;
    float timeNaive;
    float timeBlocked;
    float timeSIMD;
    float timeThreaded;
    float timeCombined;
    float gflopsNaive;
    float gflopsBlocked;
    float gflopsSIMD;
    float gflopsThreaded;
    float gflopsCombined;
    float speedupBlockedVsNaive;
    float speedupSIMDVsNaive;
    float speedupThreadedVsNaive;
    float speedupCombinedVsNaive;
};

// Run CPU benchmarks
std::vector<CPUBenchmarkResult> runCpuBenchmarks(const CPUBenchmarkConfig& config) {
    std::vector<CPUBenchmarkResult> results;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    // Set number of threads
    if (config.numThreads > 0) {
        omp_set_num_threads(config.numThreads);
    }
    
    std::cout << "Running CPU benchmarks with " << omp_get_max_threads() << " threads" << std::endl;
    
    // Run benchmarks for each matrix size
    for (int size : config.matrixSizes) {
        std::cout << "Benchmarking matrix size: " << size << "x" << size << std::endl;
        
        // Initialize matrices
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C_naive(size * size, 0.0f);
        std::vector<float> C_blocked(size * size, 0.0f);
        std::vector<float> C_simd(size * size, 0.0f);
        std::vector<float> C_threaded(size * size, 0.0f);
        std::vector<float> C_combined(size * size, 0.0f);
        
        // Fill matrices with random values
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::generate(A.begin(), A.end(), [&]() { return dist(gen); });
        std::generate(B.begin(), B.end(), [&]() { return dist(gen); });
        
        // Create timer
        Timer timer;
        
        // Initialize result
        CPUBenchmarkResult result;
        result.matrixSize = size;
        
        // Benchmark naive implementation
        if (config.runNaive && size <= 2048) {  // Skip naive for large matrices
            float totalTimeNaive = 0.0f;
            
            // Warmup
            cpu::naive::multiply(A.data(), B.data(), C_naive.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::naive::multiply(A.data(), B.data(), C_naive.data(), size, size, size);
                timer.stop();
                totalTimeNaive += timer.elapsedMilliseconds();
            }
            
            result.timeNaive = totalTimeNaive / config.repetitions;
            result.gflopsNaive = (2.0f * size * size * size) / (result.timeNaive * 1e-3) / 1e9;
            
            std::cout << "  Naive: " << std::fixed << std::setprecision(2) 
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
            cpu::blocked::multiply(A.data(), B.data(), C_blocked.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::blocked::multiply(A.data(), B.data(), C_blocked.data(), size, size, size);
                timer.stop();
                totalTimeBlocked += timer.elapsedMilliseconds();
            }
            
            result.timeBlocked = totalTimeBlocked / config.repetitions;
            result.gflopsBlocked = (2.0f * size * size * size) / (result.timeBlocked * 1e-3) / 1e9;
            result.speedupBlockedVsNaive = (result.timeNaive > 0) ? result.timeNaive / result.timeBlocked : 0.0f;
            
            std::cout << "  Blocked: " << result.timeBlocked << " ms, " 
                      << result.gflopsBlocked << " GFLOPS" << std::endl;
            if (result.speedupBlockedVsNaive > 0) {
                std::cout << "  Speedup vs Naive: " << result.speedupBlockedVsNaive << "x" << std::endl;
            }
        } else {
            result.timeBlocked = 0.0f;
            result.gflopsBlocked = 0.0f;
            result.speedupBlockedVsNaive = 0.0f;
        }
        
        // Benchmark SIMD implementation
        if (config.runSIMD) {
            float totalTimeSIMD = 0.0f;
            
            // Warmup
            cpu::simd::multiply_avx512(A.data(), B.data(), C_simd.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::simd::multiply_avx512(A.data(), B.data(), C_simd.data(), size, size, size);
                timer.stop();
                totalTimeSIMD += timer.elapsedMilliseconds();
            }
            
            result.timeSIMD = totalTimeSIMD / config.repetitions;
            result.gflopsSIMD = (2.0f * size * size * size) / (result.timeSIMD * 1e-3) / 1e9;
            result.speedupSIMDVsNaive = (result.timeNaive > 0) ? result.timeNaive / result.timeSIMD : 0.0f;
            
            std::cout << "  SIMD: " << result.timeSIMD << " ms, " 
                      << result.gflopsSIMD << " GFLOPS" << std::endl;
            if (result.speedupSIMDVsNaive > 0) {
                std::cout << "  Speedup vs Naive: " << result.speedupSIMDVsNaive << "x" << std::endl;
            }
        } else {
            result.timeSIMD = 0.0f;
            result.gflopsSIMD = 0.0f;
            result.speedupSIMDVsNaive = 0.0f;
        }
        
        // Benchmark multithreaded implementation
        if (config.runThreaded) {
            float totalTimeThreaded = 0.0f;
            
            // Warmup
            cpu::threaded::multiply(A.data(), B.data(), C_threaded.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::threaded::multiply(A.data(), B.data(), C_threaded.data(), size, size, size);
                timer.stop();
                totalTimeThreaded += timer.elapsedMilliseconds();
            }
            
            result.timeThreaded = totalTimeThreaded / config.repetitions;
            result.gflopsThreaded = (2.0f * size * size * size) / (result.timeThreaded * 1e-3) / 1e9;
            result.speedupThreadedVsNaive = (result.timeNaive > 0) ? result.timeNaive / result.timeThreaded : 0.0f;
            
            std::cout << "  Threaded: " << result.timeThreaded << " ms, " 
                      << result.gflopsThreaded << " GFLOPS" << std::endl;
            if (result.speedupThreadedVsNaive > 0) {
                std::cout << "  Speedup vs Naive: " << result.speedupThreadedVsNaive << "x" << std::endl;
            }
        } else {
            result.timeThreaded = 0.0f;
            result.gflopsThreaded = 0.0f;
            result.speedupThreadedVsNaive = 0.0f;
        }
        
        // Benchmark combined (SIMD + multithreaded) implementation
        if (config.runCombined) {
            float totalTimeCombined = 0.0f;
            
            // Warmup
            cpu::threaded::multiply_simd(A.data(), B.data(), C_combined.data(), size, size, size);
            
            for (int i = 0; i < config.repetitions; i++) {
                timer.start();
                cpu::threaded::multiply_simd(A.data(), B.data(), C_combined.data(), size, size, size);
                timer.stop();
                totalTimeCombined += timer.elapsedMilliseconds();
            }
            
            result.timeCombined = totalTimeCombined / config.repetitions;
            result.gflopsCombined = (2.0f * size * size * size) / (result.timeCombined * 1e-3) / 1e9;
            result.speedupCombinedVsNaive = (result.timeNaive > 0) ? result.timeNaive / result.timeCombined : 0.0f;
            
            std::cout << "  Combined: " << result.timeCombined << " ms, " 
                      << result.gflopsCombined << " GFLOPS" << std::endl;
            if (result.speedupCombinedVsNaive > 0) {
                std::cout << "  Speedup vs Naive: " << result.speedupCombinedVsNaive << "x" << std::endl;
            }
        } else {
            result.timeCombined = 0.0f;
            result.gflopsCombined = 0.0f;
            result.speedupCombinedVsNaive = 0.0f;
        }
        
        results.push_back(result);
    }
    
    // Save results to file if requested
    if (!config.outputFile.empty()) {
        std::ofstream outFile(config.outputFile);
        outFile << "Matrix Size,Naive (ms),Blocked (ms),SIMD (ms),Threaded (ms),Combined (ms),"
                << "Naive (GFLOPS),Blocked (GFLOPS),SIMD (GFLOPS),Threaded (GFLOPS),Combined (GFLOPS),"
                << "Speedup Blocked,Speedup SIMD,Speedup Threaded,Speedup Combined\n";
        
        for (const auto& result : results) {
            outFile << result.matrixSize << ","
                    << result.timeNaive << ","
                    << result.timeBlocked << ","
                    << result.timeSIMD << ","
                    << result.timeThreaded << ","
                    << result.timeCombined << ","
                    << result.gflopsNaive << ","
                    << result.gflopsBlocked << ","
                    << result.gflopsSIMD << ","
                    << result.gflopsThreaded << ","
                    << result.gflopsCombined << ","
                    << result.speedupBlockedVsNaive << ","
                    << result.speedupSIMDVsNaive << ","
                    << result.speedupThreadedVsNaive << ","
                    << result.speedupCombinedVsNaive << "\n";
        }
    }
    
    return results;
}

int main(int argc, char* argv[]) {
    CPUBenchmarkConfig config;
    
    // Default configuration
    config.matrixSizes = {128, 256, 512, 1024};
    config.repetitions = 5;
    config.runNaive = true;
    config.runBlocked = true;
    config.runSIMD = true;
    config.runThreaded = true;
    config.runCombined = true;
    config.numThreads = omp_get_max_threads();
    config.outputFile = "cpu_benchmark_results.csv";
    
    // Parse command line arguments
    // ...
    
    // Run benchmarks
    std::vector<CPUBenchmarkResult> results = runCpuBenchmarks(config);
    
    return 0;
}
