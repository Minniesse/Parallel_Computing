#include "common/utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace common {

// Implementation of hardware info functions
HardwareInfo getHardwareInfo() {
    HardwareInfo info;
    
    // In a real implementation, we would detect CPU features here
    // For now, use the default values in the constructor
    
    #ifdef HAS_AVX
    info.avx_supported = true;
    #endif
    
    #ifdef HAS_AVX2
    info.avx2_supported = true;
    #endif
    
    #ifdef HAS_AVX512
    info.avx512_supported = true;
    #endif
    
    return info;
}

bool isAVXAvailable() {
    #ifdef HAS_AVX
    return true;
    #else
    return false;
    #endif
}

bool isAVX2Available() {
    #ifdef HAS_AVX2
    return true;
    #else
    return false;
    #endif
}

bool isAVX512Available() {
    #ifdef HAS_AVX512
    return true;
    #else
    return false;
    #endif
}

int getOptimalBlockSize(int cacheSize) {
    // A simple heuristic for block size based on L1 cache size
    // For a matrix of floats, we aim to fit three blocks in cache:
    // One block from A, one from B, and one from C
    // For a block size of b, we need (3 * b^2 * sizeof(float)) bytes
    
    int sizeOfFloat = sizeof(float);
    int maxElements = cacheSize / (3 * sizeOfFloat);
    int blockSize = static_cast<int>(std::sqrt(maxElements));
    
    // Round down to a multiple of 16 for AVX-512 alignment
    blockSize = (blockSize / 16) * 16;
    
    // Ensure minimum block size
    return std::max(16, blockSize);
}

void* allocateAlignedMemory(size_t size, size_t alignment) {
    void* ptr = nullptr;
    int result = posix_memalign(&ptr, alignment, size);
    if (result != 0) {
        return nullptr;
    }
    return ptr;
}

void freeAlignedMemory(void* ptr) {
    free(ptr);
}

// String utility functions
std::vector<std::string> splitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        start++;
    }
    
    auto end = s.end();
    while (end != start && std::isspace(*(end - 1))) {
        end--;
    }
    
    return std::string(start, end);
}

// File and directory utilities
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

bool createDirectory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0;
}

std::string getFileExtension(const std::string& filename) {
    size_t lastDot = filename.find_last_of(".");
    if (lastDot == std::string::npos) {
        return "";
    }
    return filename.substr(lastDot + 1);
}

// Performance calculation
double calculateGFLOPS(double operations, double seconds) {
    return operations / (seconds * 1e9);
}

double calculateMemoryBandwidth(double bytesAccessed, double seconds) {
    return bytesAccessed / (seconds * 1e9);
}

// Calculate theoretical peak performance
double getTheorecticalPeakGFLOPS(const HardwareInfo& info) {
    // This is a simplification and would need to be customized for specific hardware
    double flopsPerCycle = 16.0; // Assuming AVX-512 with FMA (16 float operations per cycle)
    double clockSpeedGHz = 3.0;  // Assuming 3 GHz CPU
    return info.num_physical_cpus * flopsPerCycle * clockSpeedGHz;
}

double getTheorecticalMemoryBandwidth() {
    // This is a simplification and would need to be customized for specific hardware
    return 50.0; // Assuming 50 GB/s memory bandwidth
}

} // namespace common
