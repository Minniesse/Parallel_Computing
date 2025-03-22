#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath> // Add this include for std::sqrt

namespace common {

// Basic timer class
class Timer {
private:
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using duration_t = std::chrono::duration<double>;
    
    time_point_t start_time_;
    time_point_t end_time_;
    bool running_;
    
public:
    Timer() : running_(false) {}
    
    void start() {
        start_time_ = clock_t::now();
        running_ = true;
    }
    
    void stop() {
        end_time_ = clock_t::now();
        running_ = false;
    }
    
    void reset() {
        running_ = false;
    }
    
    // Get elapsed time in seconds
    double elapsedSeconds() const {
        if (running_) {
            auto now = clock_t::now();
            return std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        } else {
            return std::chrono::duration_cast<std::chrono::duration<double>>(end_time_ - start_time_).count();
        }
    }
    
    // Get elapsed time in milliseconds
    double elapsedMilliseconds() const {
        return elapsedSeconds() * 1000.0;
    }
    
    // Get elapsed time in microseconds
    double elapsedMicroseconds() const {
        return elapsedSeconds() * 1000000.0;
    }
    
    // Check if timer is currently running
    bool isRunning() const {
        return running_;
    }
};

// Performance benchmark class for multiple runs
class Benchmark {
private:
    std::string name_;
    std::vector<double> times_;
    double total_gflops_;
    int operations_per_run_;
    
public:
    Benchmark(const std::string& name = "", int operations_per_run = 0)
        : name_(name), total_gflops_(0.0), operations_per_run_(operations_per_run) {}
    
    void setName(const std::string& name) {
        name_ = name;
    }
    
    void setOperationsPerRun(int operations) {
        operations_per_run_ = operations;
    }
    
    // Add a timing result in seconds
    void addResult(double seconds) {
        times_.push_back(seconds);
        
        // Calculate GFLOPS if operations count is provided
        if (operations_per_run_ > 0) {
            double gflops = (operations_per_run_ / 1e9) / seconds;
            total_gflops_ += gflops;
        }
    }
    
    // Get statistics
    double minTime() const {
        return !times_.empty() ? *std::min_element(times_.begin(), times_.end()) : 0.0;
    }
    
    double maxTime() const {
        return !times_.empty() ? *std::max_element(times_.begin(), times_.end()) : 0.0;
    }
    
    double avgTime() const {
        return !times_.empty() ? 
            std::accumulate(times_.begin(), times_.end(), 0.0) / times_.size() : 0.0;
    }
    
    double medianTime() const {
        if (times_.empty()) return 0.0;
        
        std::vector<double> sorted_times = times_;
        std::sort(sorted_times.begin(), sorted_times.end());
        
        size_t n = sorted_times.size();
        if (n % 2 == 0) {
            return (sorted_times[n/2 - 1] + sorted_times[n/2]) / 2.0;
        } else {
            return sorted_times[n/2];
        }
    }
    
    double stddevTime() const {
        if (times_.size() <= 1) return 0.0;
        
        double mean = avgTime();
        double sq_sum = std::accumulate(times_.begin(), times_.end(), 0.0,
            [mean](double sum, double x) { return sum + (x - mean) * (x - mean); });
        
        return std::sqrt(sq_sum / (times_.size() - 1));
    }
    
    double avgGflops() const {
        return !times_.empty() && operations_per_run_ > 0 ? 
            total_gflops_ / times_.size() : 0.0;
    }
    
    int numRuns() const {
        return static_cast<int>(times_.size());
    }
    
    const std::string& name() const {
        return name_;
    }
    
    // Reset the benchmark
    void reset() {
        times_.clear();
        total_gflops_ = 0.0;
    }
    
    // Print results
    void printResults(std::ostream& os = std::cout) const {
        os << "Benchmark: " << name_ << std::endl;
        os << "  Runs: " << numRuns() << std::endl;
        os << "  Min time: " << std::fixed << std::setprecision(4) << minTime() * 1000 << " ms" << std::endl;
        os << "  Max time: " << std::fixed << std::setprecision(4) << maxTime() * 1000 << " ms" << std::endl;
        os << "  Avg time: " << std::fixed << std::setprecision(4) << avgTime() * 1000 << " ms" << std::endl;
        os << "  Median:   " << std::fixed << std::setprecision(4) << medianTime() * 1000 << " ms" << std::endl;
        os << "  Std dev:  " << std::fixed << std::setprecision(4) << stddevTime() * 1000 << " ms" << std::endl;
        
        if (operations_per_run_ > 0) {
            os << "  Avg perf: " << std::fixed << std::setprecision(2) << avgGflops() << " GFLOPS" << std::endl;
        }
    }
};

// Benchmark manager for multiple benchmarks
class BenchmarkManager {
private:
    std::map<std::string, Benchmark> benchmarks_;
    
public:
    // Add or get a benchmark
    Benchmark& getBenchmark(const std::string& name) {
        return benchmarks_[name];
    }
    
    // Check if a benchmark exists
    bool hasBenchmark(const std::string& name) const {
        return benchmarks_.find(name) != benchmarks_.end();
    }
    
    // Get all benchmark names
    std::vector<std::string> getBenchmarkNames() const {
        std::vector<std::string> names;
        for (const auto& pair : benchmarks_) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    // Print all benchmark results
    void printAllResults(std::ostream& os = std::cout) const {
        for (const auto& pair : benchmarks_) {
            pair.second.printResults(os);
            os << std::endl;
        }
    }
    
    // Save results to CSV file
    void saveToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
            return;
        }
        
        // Write header
        file << "Benchmark,Runs,Min (ms),Max (ms),Avg (ms),Median (ms),Std Dev (ms),Avg GFLOPS" << std::endl;
        
        // Write data
        for (const auto& pair : benchmarks_) {
            const Benchmark& bench = pair.second;
            file << bench.name() << ","
                 << bench.numRuns() << ","
                 << bench.minTime() * 1000 << ","
                 << bench.maxTime() * 1000 << ","
                 << bench.avgTime() * 1000 << ","
                 << bench.medianTime() * 1000 << ","
                 << bench.stddevTime() * 1000 << ","
                 << bench.avgGflops() << std::endl;
        }
        
        file.close();
    }
    
    // Clear all benchmarks
    void clearAll() {
        benchmarks_.clear();
    }
};

} // namespace common

// Global alias for backward compatibility
using Timer = common::Timer;
using Benchmark = common::Benchmark;
using BenchmarkManager = common::BenchmarkManager;

#endif // TIMING_H
