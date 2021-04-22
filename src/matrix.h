#pragma once

#include <iostream>
#include <initializer_list>
#include <iomanip>
#include <functional>
#include <sstream>
#include <exception>

namespace cop
{
    class Matrix
    {
    private:
        float *v_ = nullptr;
        int rows_ = 0;
        int cols_ = 0;

    public:
        Matrix(const Matrix &other) = delete;

        Matrix(Matrix &&other)
        {
            v_ = other.v_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.v_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }

        Matrix(int rows, int cols, std::function<float(int, int)> init) : rows_(rows), cols_(cols)
        {
            v_ = new float[rows_ * cols_];

            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    v_[row * cols_ + col] = init(row, col);
                }
            }
        }

        Matrix(int rows, int cols) : rows_(rows), cols_(cols)
        {
            v_ = new float[rows * cols];
        }

        Matrix(int rows, float *pData) : rows_(rows), cols_(1)
        {
            v_ = new float[rows];
            memcpy(v_, pData, rows);
        }

        Matrix(std::initializer_list<std::initializer_list<float>> init)
        {
            rows_ = init.size();
            cols_ = init.begin()->size();

            v_ = new float[rows_ * cols_];

            int index = 0;

            for (auto row : init)
            {
                for (auto value : row)
                {
                    v_[index++] = value;
                }
            }
        }

        ~Matrix()
        {
            delete[] v_;
        }

        void setData(float *pData, int nBytes)
        {
            memcpy(v_, pData, nBytes);
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
            return v_;
        }

        void addTo(const Matrix &addend)
        {
            if (addend.rows_ != rows_ || addend.cols_ != cols_)
            {
                std::stringstream message;
                message << "Cannot add matrixes." << std::endl;
                message << "Addend 1 (" << rows_ << "," << cols_ << ")" << std::endl;
                message << "Addend 2 (" << addend.rows_ << "," << addend.cols_ << ")" << std::endl;

                throw std::runtime_error(message.str());
            }

            for (int i = 0; i < rows_ * cols_; i++)
            {
                v_[i] += addend.v_[i];
            }
        }

        void operator=(Matrix &&other)
        {
            v_ = other.v_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.v_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }

        /*
         * Transpose
         */
        void operator~()
        {
            for(int row = 0; row < rows_; ++row)
            {
                for(int col = 0; col < cols_; ++col)
                {
                   
                }
            }
        }

        Matrix operator*(const Matrix &multiplier)
        {
            Matrix result(rows_, multiplier.cols_);

            float *pThis = v_;
            float *pMultiplier = multiplier.v_;
            float *pResult = result.v_;
            float sum = 0;

            for(int row = 0; row < rows_; row++)
            {
                for(int col = 0; col < multiplier.cols_; col++)
                {
                    sum = 0;

                    for(int n = 0; n < cols_; n++)
                    {
                        sum += pThis[row * cols_ + n] * pMultiplier[multiplier.cols_ * n + col];
                    }

                    pResult[multiplier.cols_ * row + col] = sum;
                }
            }

            return result;
        }

        void multiply(const Matrix &result, const Matrix &multiplier)
        {
            if (result.rows_ != rows_ || result.cols_ != multiplier.cols_ || cols_ != multiplier.rows_)
            {
                std::stringstream message;
                message << "Cannot multiply matrixes." << std::endl;
                message << "Multiplicand: " << toString() << std::endl;
                message << "Multiplier: " << multiplier.toString() << std::endl;
                message << "Result: " << result.toString() << std::endl;

                throw std::runtime_error(message.str());
            }
            
            float sum = 0.0;

            float *pThisData = v_;
            float *pResultData = result.v_;

            for (int row = 0; row < rows_; ++row)
            {
                for (int col = 0; col < multiplier.cols_; ++col)
                {
                    sum = 0.0;

                    for (int n = 0; n < cols_; n++)
                    {
                        sum += pThisData[n] * multiplier.v_[n * multiplier.cols_ + col];
                    }

                    pResultData[col] = sum;
                }

                pThisData += cols_;
                pResultData += multiplier.cols_;
            }
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
    };
}