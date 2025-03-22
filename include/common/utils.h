#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <thread>
#include <functional>
#include <memory>
#include <sys/stat.h>
#include <chrono>

namespace common {

// Hardware information
struct HardwareInfo {
    int num_cpus;                // Number of CPU cores
    int num_physical_cpus;       // Number of physical CPU cores
    int cache_line_size;         // Cache line size in bytes
    int l1_cache_size;           // L1 cache size in bytes
    int l2_cache_size;           // L2 cache size in bytes
    int l3_cache_size;           // L3 cache size in bytes
    bool avx_supported;          // Is AVX supported
    bool avx2_supported;         // Is AVX2 supported
    bool avx512_supported;       // Is AVX-512 supported
    std::string cpu_brand;       // CPU brand string
    
    // Default constructor with reasonable fallback values
    HardwareInfo() 
        : num_cpus(std::thread::hardware_concurrency()),
          num_physical_cpus(std::thread::hardware_concurrency()),
          cache_line_size(64),
          l1_cache_size(32 * 1024),      // 32KB
          l2_cache_size(256 * 1024),     // 256KB
          l3_cache_size(8 * 1024 * 1024), // 8MB
          avx_supported(false),
          avx2_supported(false),
          avx512_supported(false),
          cpu_brand("Unknown CPU") {}
};

// Get hardware information
HardwareInfo getHardwareInfo();

// Check if a SIMD instruction set is available
bool isAVXAvailable();
bool isAVX2Available();
bool isAVX512Available();

// Calculate optimal block size for cache blocking
int getOptimalBlockSize(int cacheSize);

// Memory utility functions
void* allocateAlignedMemory(size_t size, size_t alignment);
void freeAlignedMemory(void* ptr);

// Matrix generation and validation
template<typename T>
std::vector<T> generateRandomMatrix(int rows, int cols, T min_val = T(-1), T max_val = T(1)) {
    std::vector<T> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    std::generate(matrix.begin(), matrix.end(), [&]() { return static_cast<T>(dist(gen)); });
    return matrix;
}

// Check if two matrices are approximately equal
template<typename T>
bool matricesAreEqual(const T* A, const T* B, int rows, int cols, T epsilon = T(1e-4)) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            if (std::abs(A[idx] - B[idx]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

// Data loading and saving functions
template<typename T>
bool saveMatrixToFile(const std::string& filename, const T* matrix, int rows, int cols) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    // Write data
    file.write(reinterpret_cast<const char*>(matrix), rows * cols * sizeof(T));
    
    return file.good();
}

template<typename T>
bool loadMatrixFromFile(const std::string& filename, std::vector<T>& matrix, int& rows, int& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read dimensions
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    // Resize and read data
    matrix.resize(rows * cols);
    file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(T));
    
    return file.good();
}

// String utility functions
std::vector<std::string> splitString(const std::string& s, char delimiter);
std::string trim(const std::string& s);

// File and directory utilities
bool fileExists(const std::string& filename);
bool createDirectory(const std::string& path);
std::string getFileExtension(const std::string& filename);

// Performance calculation
double calculateGFLOPS(double operations, double seconds);
double calculateMemoryBandwidth(double bytesAccessed, double seconds);

// Calculate theoretical peak performance
double getTheorecticalPeakGFLOPS(const HardwareInfo& info);
double getTheorecticalMemoryBandwidth();

// Parse command line arguments
class CommandLineParser {
private:
    std::vector<std::string> args_;
    
public:
    CommandLineParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            args_.push_back(argv[i]);
        }
    }
    
    bool hasOption(const std::string& option) const {
        return std::find(args_.begin(), args_.end(), option) != args_.end();
    }
    
    std::string getOptionValue(const std::string& option, const std::string& defaultValue = "") const {
        auto it = std::find(args_.begin(), args_.end(), option);
        if (it != args_.end() && ++it != args_.end()) {
            return *it;
        }
        return defaultValue;
    }
    
    std::vector<int> getIntVector(const std::string& option, const std::vector<int>& defaultValue = {}) const {
        std::string value = getOptionValue(option);
        if (value.empty()) {
            return defaultValue;
        }
        
        std::vector<int> result;
        std::stringstream ss(value);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            result.push_back(std::stoi(item));
        }
        
        return result;
    }
    
    int getIntValue(const std::string& option, int defaultValue = 0) const {
        std::string value = getOptionValue(option);
        if (value.empty()) {
            return defaultValue;
        }
        return std::stoi(value);
    }
    
    double getDoubleValue(const std::string& option, double defaultValue = 0.0) const {
        std::string value = getOptionValue(option);
        if (value.empty()) {
            return defaultValue;
        }
        return std::stod(value);
    }
};

} // namespace common

#endif // UTILS_H
