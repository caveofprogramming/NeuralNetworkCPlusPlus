#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <functional>
#include <sstream>

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
        Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols)
        {
            v_ = new T[rows * cols]{0};
        }

        Matrix(size_t rows, size_t cols, std::function<int()> init) : rows_(rows), cols_(cols)
        {
            v_ = new T[rows * cols]{0};

            for (auto row = 0; row < rows_; row++)
            {
                for (auto col = 0; col < cols_; col++)
                {
                    v_[cols_ * row + col] = init();
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

        static int init()
        {
            return 5.0 - static_cast<int>(10.0 * rand() / RAND_MAX);
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

            for (size_t row = 0; row < rows_; row++)
            {
                for (size_t col = 0; col < multiplier.cols_; col++)
                {
                    result[row][col] = multiplyRowColumn(multiplier, row, col);
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

        Matrix<T> operator!() const
        {
            if (cols_ == 1)
            {
                Matrix result(rows_, rows_);

                for (auto index = 0; index < rows_; index++)
                {
                    result[index][index] = (*this)[index][0];
                }

                return result;
            }

            if (rows_ == 1)
            {
                Matrix result(cols_, cols_);
                for (auto index = 0; index < cols_; index++)
                {
                    result[index][index] = (*this)[0][index];
                }

                return result;
            }

            throw std::runtime_error("Cannot create diagonal matrix; neither rows nor columns is 1.");

            return Matrix(0, 0);
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
            return m.transform([&value](double entry) {
                return value * entry;
            });
        }
    };
} // namespace cop