#pragma once

#include <iostream>
#include <initializer_list>
#include <iomanip>
#include <functional>

namespace cop
{
    class Matrix
    {
    private:
        double *v_ = nullptr;
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

        Matrix(int rows, int cols, std::function<double(int, int)> init) : rows_(rows), cols_(cols)
        {
            v_ = new double[rows_ * cols_];

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
            v_ = new double[rows * cols];
        }

        Matrix(std::initializer_list<std::initializer_list<double>> init)
        {
            rows_ = init.size();
            cols_ = init.begin()->size();

            v_ = new double[rows_ * cols_];

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

        void setData(double *pData, int nBytes)
        {
            memcpy(v_, pData, nBytes);
        }

        int rows()
        {
            return rows_;
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