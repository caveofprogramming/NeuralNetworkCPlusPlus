#pragma once

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <functional>
#include <sstream>
#include <initializer_list>
#include <exception>

namespace cop
{
    template <class T>
    class Matrix
    {
    private:
        T *v_{nullptr};
        size_t rows_{0};
        size_t cols_{0};

    protected:
        T multiplyRowColumn(const Matrix<T> multiplier, size_t row, size_t col)
        {
            T result = 0;

            for (auto c = 0; c < cols_; c++)
            {
                result += (*this)[row][c] * multiplier[c][col];
            }

            return result;
        }

    public:
        size_t rows() { return rows_; };
        size_t cols() { return cols_; };

        Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols)
        {
            if (rows == 0 || cols == 0)
            {
                throw std::runtime_error("rows or columns specified to Matrix constructor is zero");
            }

            if (rows > 1E6 || cols > 1E6)
            {
                throw std::runtime_error("rows or columns specified to Matrix constructor are greater than max allowed");
            }

            v_ = new T[rows * cols]{0};
        }

        Matrix(size_t rows, size_t cols, std::function<double(int, int)> init) : rows_(rows), cols_(cols)
        {
            v_ = new T[rows * cols]{0};

            for (auto row = 0; row < rows_; row++)
            {
                for (auto col = 0; col < cols_; col++)
                {
                    v_[cols_ * row + col] = init(row, col);
                }
            }
        }

        Matrix(std::initializer_list<std::initializer_list<T>> init)
        {
            rows_ = init.size();
            cols_ = init.begin()->size();
            v_ = new T[rows_ * cols_];

            int index = 0;

            for (auto row : init)
            {
                for (auto value : row)
                {
                    v_[index++] = value;
                }
            }
        }

        Matrix(const std::initializer_list<T> init)
        {
            rows_ = 1;
            cols_ = init.size();
            v_ = new T[rows_ * cols_];

            int index = 0;

            for (auto value : init)
            {
                v_[index++] = value;
            }
        }

        Matrix(Matrix<T> &&source)
        {
            swap(*this, source);
        }

        double columns() const
        {
            return cols_;
        }

        double rows() const
        {
            return rows_;
        }

        static double signedRandomUnit(double row, double col)
        {
            return 1.0 - (2.0 * rand() / RAND_MAX);
        }

        static double identity(double row, double col)
        {
            return row == col ? 1.0 : 0.0;
        }

        T *operator[](size_t row)
        {
            return v_ + (row * cols_);
        }

        T *const operator[](size_t row) const
        {
            return v_ + (row * cols_);
        }

        Matrix<T> operator+(const Matrix<T> &addend)
        {
            if (rows_ != addend.rows_ || cols_ != addend.cols_)
            {
                throw std::runtime_error("Cannot add matrixes. Different sizes.");
            }

            Matrix<T> result(rows_, cols_);

            int entries = rows_ * cols_;

            for (int i = 0; i < entries; i++)
            {
                result.v_[i] = v_[i] + addend.v_[i];
            }

            return result;
        }

        Matrix<T> operator%(const Matrix<T> &multiplier)
        {
            if (rows_ != multiplier.rows_ || cols_ != multiplier.cols_)
            {
                throw std::runtime_error("Cannot calculate Hadamard product for matrixes. Different sizes.");
            }

            Matrix<T> result(rows_, cols_);

            int entries = rows_ * cols_;

            for (int i = 0; i < entries; i++)
            {
                result.v_[i] = v_[i] * multiplier.v_[i];
            }

            return result;
        }

        Matrix<T> operator-(const Matrix<T> &subtrahend)
        {
            if (rows_ != subtrahend.rows_ || cols_ != subtrahend.cols_)
            {
                throw std::runtime_error("Cannot subtract matrixes. Different sizes.");
            }

            Matrix<T> result(rows_, cols_);

            int entries = rows_ * cols_;

            for (int i = 0; i < entries; i++)
            {
                result.v_[i] = v_[i] - subtrahend.v_[i];
            }

            return result;
        }

        void operator-=(const Matrix<T> &subtrahend)
        {
            if (rows_ != subtrahend.rows_ || cols_ != subtrahend.cols_)
            {
                throw std::runtime_error("Cannot subtract matrixes. Different sizes.");
            }

            int entries = rows_ * cols_;

            for (int i = 0; i < entries; i++)
            {
                v_[i] -= subtrahend.v_[i];
            }
        }

        Matrix<T> operator*(const Matrix<T> &multiplier)
        {
            if (cols_ != multiplier.rows_)
            {
                std::stringstream ss;
                ss << "Cannot multiply matrixes. Column and row sizes differ: ";
                ss << rows_ << "x" << cols_;
                ss << ", " << multiplier.rows_ << "x" << multiplier.cols_;
                throw std::runtime_error(ss.str());
            }

            Matrix<T> result(rows_, multiplier.cols_);

            const int multiplicationsPerEntry = cols_;
            const int multiplicandSize = rows_ * cols_;
            const int multiplierSize = multiplier.rows_ * multiplier.cols_;
            const int outputCols = multiplier.cols_;
            const int outputSize = rows_ * outputCols;
            const int totalMultiplications = outputSize * multiplicationsPerEntry;

            double sum = 0;
            int multiplicandOffset = 0;
            int multiplierOffset = 0;
            int outputOffset = 0;
            int multiplications = 0;

            int outputRow = 0;

            for (int i = 0; i < totalMultiplications; ++i)
            {
                sum += v_[multiplicandOffset] * multiplier.v_[multiplierOffset];

                ++multiplications;

                if(++multiplicandOffset == multiplicandSize)
                {
                    ++multiplierOffset;
                    multiplicandOffset = 0;
                }

                multiplierOffset += outputCols;

                if(multiplications == multiplicationsPerEntry)
                {
                    multiplications = 0;
                    result.v_[outputOffset] = sum;
                    sum = 0;
                    outputOffset += outputCols;
                }

                if(outputOffset >= outputSize)
                {
                    ++outputOffset -= outputSize;
                }

                if(multiplierOffset >= multiplierSize)
                {
                    multiplierOffset -= multiplierSize;
                }
            }

            return result;
        }

        ~Matrix()
        {
            delete[] v_;
        }

        Matrix(const Matrix<T> &source) : rows_(source.rows_), cols_(source.cols_)
        {
            v_ = new T[rows_ * cols_];
            std::copy_n(source.v_, rows_ * cols_, v_);
        }

        void swap(Matrix<T> &left, Matrix<T> &right)
        {
            std::swap(left.rows_, right.rows_);
            std::swap(left.cols_, right.cols_);
            std::swap(left.v_, right.v_);
        }

        Matrix<T> &operator=(const Matrix<T> &source)
        {
            Matrix<T> m(source);
            swap(*this, m);
            return *this;
        }

        Matrix<T> &operator=(Matrix<T> &&source)
        {
            swap(*this, source);
            return *this;
        }

        Matrix<T> operator~() const
        {
            Matrix<T> result(cols_, rows_);

            for (size_t row = 0; row < rows_; row++)
            {
                for (size_t col = 0; col < cols_; col++)
                {
                    result[col][row] = (*this)[row][col];
                }
            }

            return result;
        }

        double magnitude() const
        {
            if (rows_ != 1 && cols_ != 1)
            {
                throw std::runtime_error("Cannot get magnitude of non-vector matrix.");
            }

            auto m = *this;

            if (rows_ == 1)
            {
                return (m * ~m)[0][0];
            }
            else
            {
                return (~m * m)[0][0];
            }
        }

        Matrix<T>
        transform(std::function<T(T)> transformation) const
        {
            Matrix<T> result(rows_, cols_);

            for (size_t row = 0; row < rows_; row++)
            {
                for (size_t col = 0; col < cols_; col++)
                {
                    result[row][col] = transformation((*this)[row][col]);
                }
            }

            return result;
        }

        friend std::ostream &operator<<(std::ostream &out, const Matrix<T> &m)
        {
            out << std::showpos << std::fixed << std::setprecision(5);

            int index = 0;

            for (auto row = 0; row < m.rows_; row++)
            {
                for (auto col = 0; col < m.cols_; col++)
                {
                    out << std::setw(12) << m.v_[index++];
                }

                out << std::endl;
            }

            return out;
        }

        friend Matrix<T> operator*(T value, const Matrix<T> &m)
        {
            Matrix<T> result(m.rows_, m.cols_);

            const int size = m.rows_ * m.cols_;

            for(int i = 0; i < size; i++)
            {
                result.v_[i] = m.v_[i] * value;
            }

            return result;
        }
    };
} // namespace cop