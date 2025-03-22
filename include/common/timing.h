#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <functional>
#include <utility>

// Simple timer class for performance measurements
class Timer {
public:
    // Start the timer
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }
    
    // Stop the timer
    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }
    
    // Get elapsed time in seconds
    double elapsed_seconds() const {
        auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
        std::chrono::duration<double> elapsed = end - start_time_;
        return elapsed.count();
    }
    
    // Get elapsed time in milliseconds
    double elapsed_milliseconds() const {
        return elapsed_seconds() * 1000.0;
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
    bool running_ = false;
};

// Benchmark class to run and collect multiple timing measurements
class Benchmark {
public:
    // Add a function to benchmark
    void add_function(std::string name, std::function<void()> func) {
        functions_.push_back({std::move(name), std::move(func)});
    }
    
    // Run all benchmarks for specified iterations
    void run_all(int iterations = 5);
    
    // Calculate and print GFLOPS for matrix multiplication
    void print_gflops(size_t m, size_t n, size_t k) const;
    
    // Get the results
    const std::vector<std::pair<std::string, std::vector<double>>>& results() const {
        return results_;
    }
    
private:
    std::vector<std::pair<std::string, std::function<void()>>> functions_;
    std::vector<std::pair<std::string, std::vector<double>>> results_;
};
