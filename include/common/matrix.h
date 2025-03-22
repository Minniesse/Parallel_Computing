#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <string>
#include <memory>
#include <cstring> // Add this include for std::memset and std::memcpy

namespace common {

// Forward declaration
template<typename T>
class Matrix;

// Matrix class template
template<typename T>
class Matrix {
private:
    int rows_;
    int cols_;
    std::vector<T> data_;
    bool aligned_;
    void* aligned_data_;

public:
    // Constructors
    Matrix() : rows_(0), cols_(0), aligned_(false), aligned_data_(nullptr) {}
    
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), aligned_(false), aligned_data_(nullptr) {
        if (rows < 0 || cols < 0) {
            throw std::invalid_argument("Matrix dimensions cannot be negative");
        }
        data_.resize(rows * cols, T(0));
    }
    
    Matrix(int rows, int cols, const T& value) : rows_(rows), cols_(cols), aligned_(false), aligned_data_(nullptr) {
        if (rows < 0 || cols < 0) {
            throw std::invalid_argument("Matrix dimensions cannot be negative");
        }
        data_.resize(rows * cols, value);
    }
    
    // Create aligned matrix (for SIMD operations)
    Matrix(int rows, int cols, bool aligned) : rows_(rows), cols_(cols), aligned_(aligned), aligned_data_(nullptr) {
        if (rows < 0 || cols < 0) {
            throw std::invalid_argument("Matrix dimensions cannot be negative");
        }
        
        if (aligned) {
            // Allocate aligned memory (64-byte alignment for AVX-512)
            int result = posix_memalign(&aligned_data_, 64, rows * cols * sizeof(T));
            if (result != 0) {
                throw std::runtime_error("Failed to allocate aligned memory");
            }
            // Initialize memory to zero
            std::memset(aligned_data_, 0, rows * cols * sizeof(T));
        } else {
            // Use standard vector
            data_.resize(rows * cols, T(0));
        }
    }
    
    // Copy constructor
    Matrix(const Matrix<T>& other) : rows_(other.rows_), cols_(other.cols_), aligned_(other.aligned_) {
        if (aligned_) {
            // Allocate and copy aligned memory
            int result = posix_memalign(&aligned_data_, 64, rows_ * cols_ * sizeof(T));
            if (result != 0) {
                throw std::runtime_error("Failed to allocate aligned memory");
            }
            std::memcpy(aligned_data_, other.aligned_data_, rows_ * cols_ * sizeof(T));
        } else {
            // Copy vector data
            data_ = other.data_;
        }
    }
    
    // Move constructor
    Matrix(Matrix<T>&& other) noexcept 
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)), 
          aligned_(other.aligned_), aligned_data_(other.aligned_data_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.aligned_ = false;
        other.aligned_data_ = nullptr;
    }
    
    // Destructor
    ~Matrix() {
        if (aligned_ && aligned_data_) {
            free(aligned_data_);
            aligned_data_ = nullptr;
        }
    }
    
    // Assignment operator
    Matrix<T>& operator=(const Matrix<T>& other) {
        if (this != &other) {
            // Clean up existing aligned memory if needed
            if (aligned_ && aligned_data_) {
                free(aligned_data_);
                aligned_data_ = nullptr;
            }
            
            rows_ = other.rows_;
            cols_ = other.cols_;
            aligned_ = other.aligned_;
            
            if (aligned_) {
                // Allocate and copy aligned memory
                int result = posix_memalign(&aligned_data_, 64, rows_ * cols_ * sizeof(T));
                if (result != 0) {
                    throw std::runtime_error("Failed to allocate aligned memory");
                }
                std::memcpy(aligned_data_, other.aligned_data_, rows_ * cols_ * sizeof(T));
            } else {
                // Copy vector data
                data_ = other.data_;
            }
        }
        return *this;
    }
    
    // Move assignment operator
    Matrix<T>& operator=(Matrix<T>&& other) noexcept {
        if (this != &other) {
            // Clean up existing aligned memory if needed
            if (aligned_ && aligned_data_) {
                free(aligned_data_);
            }
            
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            aligned_ = other.aligned_;
            aligned_data_ = other.aligned_data_;
            
            other.rows_ = 0;
            other.cols_ = 0;
            other.aligned_ = false;
            other.aligned_data_ = nullptr;
        }
        return *this;
    }
    
    // Accessors
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    bool is_aligned() const { return aligned_; }
    
    // Element access
    T& operator()(int i, int j) {
        if (i < 0 || i >= rows_ || j < 0 || j >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        if (aligned_) {
            return ((T*)aligned_data_)[i * cols_ + j];
        } else {
            return data_[i * cols_ + j];
        }
    }
    
    const T& operator()(int i, int j) const {
        if (i < 0 || i >= rows_ || j < 0 || j >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        if (aligned_) {
            return ((T*)aligned_data_)[i * cols_ + j];
        } else {
            return data_[i * cols_ + j];
        }
    }
    
    // Get raw pointer to data
    T* data() {
        if (aligned_) {
            return static_cast<T*>(aligned_data_);
        } else {
            return data_.data();
        }
    }
    
    const T* data() const {
        if (aligned_) {
            return static_cast<T*>(aligned_data_);
        } else {
            return data_.data();
        }
    }
    
    // Fill matrix with random values
    void randomize(T min_val = T(-1), T max_val = T(1)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        if (aligned_) {
            T* data_ptr = static_cast<T*>(aligned_data_);
            for (int i = 0; i < rows_ * cols_; ++i) {
                data_ptr[i] = static_cast<T>(dist(gen));
            }
        } else {
            std::generate(data_.begin(), data_.end(), [&]() { return static_cast<T>(dist(gen)); });
        }
    }
    
    // Fill matrix with a specific value
    void fill(const T& value) {
        if (aligned_) {
            T* data_ptr = static_cast<T*>(aligned_data_);
            for (int i = 0; i < rows_ * cols_; ++i) {
                data_ptr[i] = value;
            }
        } else {
            std::fill(data_.begin(), data_.end(), value);
        }
    }
    
    // Check for equality with another matrix
    bool equals(const Matrix<T>& other, T epsilon = T(1e-6)) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (std::abs((*this)(i, j) - other(i, j)) > epsilon) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // Print matrix to stream
    void print(std::ostream& os = std::cout, int width = 8, int precision = 2) const {
        os << std::fixed << std::setprecision(precision);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                os << std::setw(width) << (*this)(i, j) << " ";
            }
            os << std::endl;
        }
    }
};

// Function to create an identity matrix
template<typename T>
Matrix<T> identity(int size) {
    Matrix<T> result(size, size);
    for (int i = 0; i < size; ++i) {
        result(i, i) = T(1);
    }
    return result;
}

// Function to verify the result of matrix multiplication
template<typename T>
bool verify_multiplication(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C, T epsilon = T(1e-4)) {
    if (A.cols() != B.rows() || A.rows() != C.rows() || B.cols() != C.cols()) {
        return false;
    }
    
    // Compute reference result using naive algorithm
    Matrix<T> reference(C.rows(), C.cols());
    
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j) {
            T sum = T(0);
            for (int k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            reference(i, j) = sum;
        }
    }
    
    // Compare with the provided result
    return C.equals(reference, epsilon);
}

} // namespace common

#endif // MATRIX_H
