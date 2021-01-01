#include <iostream>
#include "matrix.h"

using namespace std;

typedef cop::Matrix<double> Matrix;

int main()
{
    Matrix m1 = ~Matrix{2, 3};
    Matrix m2 = {{3, 1}, {2, 3}};

    Matrix expected = ~Matrix{1, 3};

    cout << "input:\n" << m1 << endl;
    cout << "layer:\n" << m2 << endl;

    cop::Matrix<double> result(1, 2);

    auto calculateCost = [&]() {
        cop::Matrix<double> difference = result - expected;

        double sum = 0.0;

        for (auto i = 0; i < expected.columns(); i++)
        {
            sum += difference[0][i] * difference[0][i];
        }

        return sum;
    };

    auto calculateCostGradient = [&]() {
        cop::Matrix<double> difference = 2 * result - expected;

        return difference;
    };

    const double inc = 0.000001;

    result = m2 * m1;

    cout << "result:\n" << result << endl;
    cout << "expected:\n" << expected << endl;
    double cost1 = calculateCost();

    m2[0][0] += inc;

    result = m2 * m1;
    double cost2 = calculateCost();

    auto costGradients = calculateCostGradient();

    cout << "cost gradients:\n" << costGradients << endl;

    double weightGradient = (cost2 - cost1) / inc;

    cout << "weight gradient:\n" << weightGradient << endl;

    return 0;
}