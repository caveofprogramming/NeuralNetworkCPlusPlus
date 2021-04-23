#pragma once

#include <iostream>
#include <initializer_list>
#include <iomanip>
#include <functional>
#include <sstream>
#include <exception>
#include <algorithm>
#include <utility>
#include <vector>

namespace cop
{
    class Matrix
    {
    private:
        std::vector<float> e_;

        int rows_ = 0;
        int cols_ = 0;

    public:
        Matrix(const Matrix &other) = delete;

        Matrix(Matrix &&other)
        {
            *this = std::move(other);
        }

        void operator=(Matrix &&other)
        {
            e_ = std::move(other.e_);

            rows_ = other.rows_;
            cols_ = other.cols_;
            other.rows_ = 0;
            other.cols_ = 0;
        }

        Matrix(int rows, int cols, std::function<float(int, int)> init) : rows_(rows), cols_(cols)
        {
            e_.resize(rows_ * cols_);

            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    e_.data()[row * cols_ + col] = init(row, col);
                }
            }
        }

        Matrix(int rows, int cols) : rows_(rows), cols_(cols)
        {
            e_.resize(rows * cols);
        }

        Matrix(int rows, float *pData) : rows_(rows), cols_(1)
        {
            e_.resize(rows);
            memcpy(e_.data(), pData, rows);
        }

        Matrix(std::initializer_list<std::initializer_list<float>> init)
        {
            rows_ = init.size();
            cols_ = init.begin()->size();

            e_.resize(rows_ * cols_);

            int index = 0;

            for (auto row : init)
            {
                for (auto value : row)
                {
                    e_.data()[index++] = value;
                }
            }
        }

        ~Matrix()
        {
        }

        void setData(float *pData, int nBytes)
        {
            memcpy(e_.data(), pData, nBytes);
        }

        int rows()
        {
            return rows_;
        }

        int cols()
        {
            return cols_;
        }

        float *data()
        {
            return e_.data();
        }

        const float *operator[](int index) const
        {
            return e_.data() + (index * cols_);
        }

        float *operator[](int index)
        {
            return e_.data() + (index * cols_);
        }

        /*
         * Transpose
         */
        Matrix operator~() const
        {
            Matrix transposed(cols_, rows_);

            for (int row = 0; row < rows_; ++row)
            {
                for (int col = 0; col < cols_; ++col)
                {
                    transposed[col][row] = (*this)[row][col];
                }
            }

            return transposed;
        }

        Matrix operator+(const Matrix &addend) const
        {
            if (addend.rows_ != rows_ || addend.cols_ != cols_)
            {
                std::stringstream message;
                message << "Cannot add matrixes." << std::endl;
                message << "Addend 1 (" << rows_ << "," << cols_ << ")" << std::endl;
                message << "Addend 2 (" << addend.rows_ << "," << addend.cols_ << ")" << std::endl;

                throw std::runtime_error(message.str());
            }

            Matrix result(rows_, cols_);

            const float *pThisData = e_.data();
            const float *pAddendData = addend.e_.data();
            float *pResultData = result.e_.data();

            for (int i = 0; i < rows_ * cols_; i++)
            {
                *pResultData++ = *pThisData++ + *pAddendData++;
            }

            return result;
        }

        Matrix operator*(const Matrix &multiplier)
        {
            Matrix result(rows_, multiplier.cols_);

            Matrix transposed = ~multiplier;

            float sum = 0.0;

            for (int row = 0; row < rows_; row++)
            {
                for (int col = 0; col < multiplier.cols_; col++)
                {
                    sum = 0.0;

                    for (int n = 0; n < cols_; n++)
                    {
                        sum += (*this)[row][n] * transposed[col][n];
                    }

                    result[row][col] = sum;
                }
            }

            return result;
        }

        std::string toString() const
        {
            std::stringstream ss;

            ss << "(" << rows_ << "," << cols_ << ")";

            return ss.str();
        }

        friend std::ostream &operator<<(std::ostream &out, const Matrix &m)
        {
            out << std::showpos << std::fixed << std::setprecision(5);

            for (auto row = 0; row < m.rows_; row++)
            {
                for (auto col = 0; col < m.cols_; col++)
                {
                    out << std::setw(12) << m[row][col];
                }

                out << std::endl;
            }

            return out;
        }
    };
}