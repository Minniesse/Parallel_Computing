#include "common/timing.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

void Benchmark::run_all(int iterations) {
    // Clear previous results
    results_.clear();
    
    // Run each function for the specified number of iterations
    for (const auto& func_pair : functions_) {
        const std::string& name = func_pair.first;
        const auto& func = func_pair.second;
        
        std::vector<double> times;
        
        std::cout << "Running " << name << "... ";
        std::cout.flush();
        
        // Warm-up run
        Timer timer;
        timer.start();
        func();
        timer.stop();
        
        // Benchmark runs
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            func();
            timer.stop();
            times.push_back(timer.elapsed_seconds());
        }
        
        // Calculate statistics
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_time = *std::min_element(times.begin(), times.end());
        
        std::cout << "Avg: " << std::fixed << std::setprecision(4) << avg_time << "s, "
                  << "Min: " << min_time << "s" << std::endl;
        
        // Save results
        results_.push_back({name, times});
    }
}

void Benchmark::print_gflops(size_t m, size_t n, size_t k) const {
    // Total floating-point operations for matrix multiplication: 2*m*n*k
    double total_flops = 2.0 * m * n * k;
    
    std::cout << "\nPerformance in GFLOPS (higher is better):" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << std::setw(20) << "Implementation" << " | "
              << std::setw(10) << "Avg GFLOPS" << " | "
              << std::setw(10) << "Peak GFLOPS" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    // Print GFLOPS for each implementation
    for (const auto& result : results_) {
        const std::string& name = result.first;
        const std::vector<double>& times = result.second;
        
        // Calculate average time
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        // Calculate minimum time (peak performance)
        double min_time = *std::min_element(times.begin(), times.end());
        
        // Calculate GFLOPS
        double avg_gflops = (total_flops / avg_time) / 1e9;
        double peak_gflops = (total_flops / min_time) / 1e9;
        
        std::cout << std::setw(20) << name << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << avg_gflops << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << peak_gflops << std::endl;
    }
    
    std::cout << "------------------------------------------" << std::endl;
}
