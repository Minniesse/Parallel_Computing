#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <cstdlib>
#include <random>
#include <cassert>

// Matrix class to store and manipulate 2D floating-point data
class Matrix {
public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, float init_val);
    
    // Copy and move constructors
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    
    // Assignment operators
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    
    // Destructor
    ~Matrix() = default;
    
    // Element access
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    // Matrix properties
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    
    // Raw data access for optimized implementations
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    // Memory alignment methods
    void align(size_t alignment);
    bool is_aligned(size_t alignment) const;
    
    // Fill the matrix with random values
    void fill_random(float min_val = -1.0f, float max_val = 1.0f);
    
    // Check if two matrices are equal within a tolerance
    bool equals(const Matrix& other, float tolerance = 1e-5) const;
    
    // Print the matrix
    void print(std::ostream& os = std::cout, size_t precision = 4) const;
    
private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
    size_t leading_dimension_;  // For potential padding
};

// Matrix multiplication interface
namespace MatrixMult {
    // Function pointer type for multiplication implementations
    using MultFunc = void (*)(const Matrix& A, const Matrix& B, Matrix& C);
    
    // Verify that matrices can be multiplied (dimensions check)
    bool can_multiply(const Matrix& A, const Matrix& B, const Matrix& C);
}
