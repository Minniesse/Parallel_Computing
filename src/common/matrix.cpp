#include "common/matrix.h"
#include <cstring>
#include <iomanip>
#include <random>
#include <cmath>

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), leading_dimension_(cols) {
    data_.resize(rows * cols, 0.0f);
}

Matrix::Matrix(size_t rows, size_t cols, float init_val)
    : rows_(rows), cols_(cols), leading_dimension_(cols) {
    data_.resize(rows * cols, init_val);
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), 
      leading_dimension_(other.leading_dimension_), data_(other.data_) {
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), 
      leading_dimension_(other.leading_dimension_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.leading_dimension_ = 0;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        leading_dimension_ = other.leading_dimension_;
        data_ = other.data_;
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        leading_dimension_ = other.leading_dimension_;
        data_ = std::move(other.data_);
        
        other.rows_ = 0;
        other.cols_ = 0;
        other.leading_dimension_ = 0;
    }
    return *this;
}

float& Matrix::at(size_t row, size_t col) {
    return data_[row * leading_dimension_ + col];
}

const float& Matrix::at(size_t row, size_t col) const {
    return data_[row * leading_dimension_ + col];
}

void Matrix::align(size_t alignment) {
    // This is a simple implementation that pads each row to the alignment boundary
    // A more sophisticated implementation would use aligned allocators
    size_t padded_cols = ((cols_ + alignment - 1) / alignment) * alignment;
    leading_dimension_ = padded_cols;
    
    std::vector<float> aligned_data(rows_ * padded_cols, 0.0f);
    
    // Copy data row by row
    for (size_t i = 0; i < rows_; ++i) {
        std::memcpy(&aligned_data[i * padded_cols], 
                   &data_[i * cols_], 
                   cols_ * sizeof(float));
    }
    
    data_ = std::move(aligned_data);
}

bool Matrix::is_aligned(size_t alignment) const {
    return leading_dimension_ % alignment == 0;
}

void Matrix::fill_random(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            at(i, j) = dist(gen);
        }
    }
}

bool Matrix::equals(const Matrix& other, float tolerance) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (std::abs(at(i, j) - other.at(i, j)) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

void Matrix::print(std::ostream& os, size_t precision) const {
    os << "Matrix " << rows_ << " x " << cols_ << ":" << std::endl;
    
    // Limit the number of rows and columns to print
    size_t max_rows = std::min(rows_, size_t(10));
    size_t max_cols = std::min(cols_, size_t(10));
    
    for (size_t i = 0; i < max_rows; ++i) {
        for (size_t j = 0; j < max_cols; ++j) {
            os << std::fixed << std::setprecision(precision) << at(i, j) << " ";
        }
        
        if (cols_ > max_cols) {
            os << "...";
        }
        
        os << std::endl;
    }
    
    if (rows_ > max_rows) {
        os << "..." << std::endl;
    }
}

namespace MatrixMult {
    bool can_multiply(const Matrix& A, const Matrix& B, const Matrix& C) {
        // Check dimensions
        if (A.cols() != B.rows() || A.rows() != C.rows() || B.cols() != C.cols()) {
            std::cerr << "Error: Matrix dimensions do not match for multiplication." << std::endl;
            std::cerr << "A: " << A.rows() << "x" << A.cols() << ", "
                     << "B: " << B.rows() << "x" << B.cols() << ", "
                     << "C: " << C.rows() << "x" << C.cols() << std::endl;
            return false;
        }
        
        return true;
    }
}
