#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "common/matrix.h"
#include "common/timing.h"
#include "common/utils.h"
#include "cpu/naive.h"
#include "cpu/blocked.h"
#include "cpu/simd.h"
#include "cpu/threaded.h"
#include <omp.h>

#ifdef HAS_CUDA
#include "gpu/cuda_wrapper.h"
#include "gpu/multi_gpu.h"
#endif

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --size N                 Matrix size (N x N)" << std::endl;
    std::cout << "  --algo ALGORITHM         Algorithm to use (naive, blocked, simd, threaded, combined, cuda, cublas)" << std::endl;
    std::cout << "  --verify                 Verify correctness of result" << std::endl;
    std::cout << "  --repeat N               Number of repetitions for timing" << std::endl;
    std::cout << "  --threads N              Number of threads to use (for CPU implementations)" << std::endl;
    std::cout << "  --device N               GPU device ID to use (for GPU implementations)" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int matrixSize = 1024;
    std::string algorithm = "combined";
    bool verify = false;
    int repetitions = 3;
    int numThreads = omp_get_max_threads();
    int deviceId = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--size" && i + 1 < argc) {
            matrixSize = std::stoi(argv[++i]);
        }
        else if (arg == "--algo" && i + 1 < argc) {
            algorithm = argv[++i];
        }
        else if (arg == "--verify") {
            verify = true;
        }
        else if (arg == "--repeat" && i + 1 < argc) {
            repetitions = std::stoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
            omp_set_num_threads(numThreads);
        }
        else if (arg == "--device" && i + 1 < argc) {
            deviceId = std::stoi(argv[++i]);
        }
        else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Print configuration
    std::cout << "Matrix Multiplication Configuration:" << std::endl;
    std::cout << "  Matrix size: " << matrixSize << " x " << matrixSize << std::endl;
    std::cout << "  Algorithm: " << algorithm << std::endl;
    std::cout << "  Verification: " << (verify ? "enabled" : "disabled") << std::endl;
    std::cout << "  Repetitions: " << repetitions << std::endl;
    std::cout << "  Threads: " << numThreads << std::endl;
    
    // Create matrices
    common::Matrix<float> A(matrixSize, matrixSize);
    common::Matrix<float> B(matrixSize, matrixSize);
    common::Matrix<float> C(matrixSize, matrixSize);
    common::Matrix<float> C_reference(matrixSize, matrixSize);
    
    // Initialize matrices with random values
    A.randomize();
    B.randomize();
    
    // Create timer
    Timer timer;
    
    // Select algorithm and run benchmark
    double totalTime = 0.0;
    
    for (int i = 0; i < repetitions; i++) {
        // Reset result matrix
        C.fill(0.0f);
        
        if (algorithm == "naive") {
            timer.start();
            cpu::naive::multiply(A.data(), B.data(), C.data(), matrixSize, matrixSize, matrixSize);
            timer.stop();
        }
        else if (algorithm == "blocked") {
            timer.start();
            cpu::blocked::multiply(A.data(), B.data(), C.data(), matrixSize, matrixSize, matrixSize);
            timer.stop();
        }
        else if (algorithm == "simd") {
            timer.start();
            cpu::simd::multiply(A.data(), B.data(), C.data(), matrixSize, matrixSize, matrixSize);
            timer.stop();
        }
        else if (algorithm == "threaded") {
            timer.start();
            cpu::threaded::multiply(A.data(), B.data(), C.data(), matrixSize, matrixSize, matrixSize);
            timer.stop();
        }
        else if (algorithm == "combined") {
            timer.start();
            cpu::threaded::multiply_simd(A.data(), B.data(), C.data(), matrixSize, matrixSize, matrixSize);
            timer.stop();
        }
#ifdef HAS_CUDA
        else if (algorithm == "cuda") {
            gpu::CudaMatrixMultiplier multiplier;
            multiplier.initialize(matrixSize, matrixSize, matrixSize, deviceId);
            multiplier.setMatrixA(A.data());
            multiplier.setMatrixB(B.data());
            
            timer.start();
            multiplier.multiply(C.data());
            timer.stop();
            
            multiplier.cleanup();
        }
        else if (algorithm == "cublas") {
            gpu::CudaMatrixMultiplier multiplier;
            multiplier.initialize(matrixSize, matrixSize, matrixSize, deviceId);
            multiplier.setMatrixA(A.data());
            multiplier.setMatrixB(B.data());
            
            timer.start();
            multiplier.multiplyWithCuBLAS(C.data());
            timer.stop();
            
            multiplier.cleanup();
        }
#endif
        else {
            std::cerr << "Error: Unknown algorithm '" << algorithm << "'" << std::endl;
            return 1;
        }
        
        totalTime += timer.elapsedSeconds();
        
        std::cout << "Run " << (i + 1) << ": " << std::fixed << std::setprecision(4) 
                  << timer.elapsedMilliseconds() << " ms" << std::endl;
    }
    
    // Calculate average time and GFLOPS
    double avgTime = totalTime / repetitions;
    double gflops = (2.0 * matrixSize * matrixSize * matrixSize) / (avgTime * 1e9);
    
    std::cout << "Average time: " << std::fixed << std::setprecision(4) 
              << (avgTime * 1000.0) << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS" << std::endl;
    
    // Verify result if requested
    if (verify) {
        std::cout << "Verifying result... ";
        
        // Compute reference result
        cpu::naive::multiply(A.data(), B.data(), C_reference.data(), matrixSize, matrixSize, matrixSize);
        
        // Compare results
        bool correct = common::verify_multiplication(A, B, C);
        
        std::cout << (correct ? "PASSED" : "FAILED") << std::endl;
    }
    
    return 0;
}
