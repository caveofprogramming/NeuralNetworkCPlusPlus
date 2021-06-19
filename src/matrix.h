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
#include <sstream>

namespace cop
{
    class Matrix
    {
    private:
        std::vector<double> e_;

        int rows_ = 0;
        int cols_ = 0;

    public:
        Matrix(const Matrix &other) = delete;

        Matrix()
        {

        }

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

        Matrix(int rows, int cols, std::function<double(int, int)> init) : rows_(rows), cols_(cols)
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

        Matrix operator-(const Matrix &other)
        {
            Matrix result(rows_, cols_);

            for (int i = 0; i < rows_ * cols_; i++)
            {
                result.e_[i] = e_[i] - other.e_[i];
            }

            return result;
        }

        Matrix(int rows, int cols) : rows_(rows), cols_(cols)
        {
            e_.resize(rows * cols);
        }

        Matrix(int rows, int cols, double *pData) : rows_(rows), cols_(cols)
        {
            e_.resize(rows * cols);
            memcpy(e_.data(), pData, rows * cols * sizeof(double));
        }

        Matrix(int rows, double *pData) : rows_(rows), cols_(1)
        {
            e_.resize(rows);
            memcpy(e_.data(), pData, rows * sizeof(double));
        }

        Matrix(std::initializer_list<std::initializer_list<double>> init)
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

            // Initialise column vectors by default,
            // not row vectors.
            if (rows_ == 1)
            {
                rows_ = cols_;
                cols_ = 1;
            }
        }

        ~Matrix()
        {
        }

        void serialize(std::ostream &out)
        {
            out.write(reinterpret_cast<char *>(&rows_), sizeof(rows_));
            out.write(reinterpret_cast<char *>(&cols_), sizeof(cols_));
            out.write(reinterpret_cast<char *>(e_.data()), rows_ * cols_ * sizeof(e_[0]));
        }

        void deserialize(std::istream &in)
        {
            in.read(reinterpret_cast<char *>(&rows_), sizeof(rows_));
            in.read(reinterpret_cast<char *>(&cols_), sizeof(cols_));

            e_.resize(rows_ * cols_);

            in.read(reinterpret_cast<char *>(e_.data()), rows_ * cols_ * sizeof(e_[0]));
        }

        void setData(double *pData, int nBytes)
        {
            memcpy(e_.data(), pData, nBytes);
        }

        int rows() const
        {
            return rows_;
        }

        int cols() const
        {
            return cols_;
        }

        double *data()
        {
            return e_.data();
        }

        const double *operator[](int index) const
        {
            return e_.data() + (index * cols_);
        }

        double *operator[](int index)
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

            const double *pThisData = e_.data();
            const double *pAddendData = addend.e_.data();
            double *pResultData = result.e_.data();

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

            double sum = 0.0;

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

        Matrix augment(Matrix &m)
        {
             if (rows() != m.rows())
            {
                std::stringstream message;
                message << "Cannot augment matrix." << std::endl;
                message << rows() << " vs. " << m.rows()  << std::endl;

                throw std::runtime_error(message.str());
            }

            Matrix result(rows(), cols() + m.cols());

            for(int i = 0; i < rows(); i++)
            {
                for(int col = 0; col < cols(); col++)
                {
                    result[i][col] = (*this)[i][col];
                }

                for(int col = 0; col < m.cols(); col++)
                {
                    result[i][col + cols()] = m[i][col];
                }
            }

            return result;
        }

        std::string toString()
        {
            std::stringstream ss;

            for(int row = 0; row < rows_; row++)
            {
                for(int col = 0; col < cols_; col++)
                {
                    ss << std::setw(12) << (*this)[row][col];
                }

                ss << "\n";
            }

            return ss.str();
        }
    };


}