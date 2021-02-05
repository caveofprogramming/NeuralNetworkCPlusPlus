#include <iostream>
#include <vector>
#include "matrix.h"

using namespace std;

typedef cop::Matrix<double> Matrix;

double calculateCost(Matrix &result, Matrix &expected)
{
     cop::Matrix<double> difference = result - expected;
     return (~difference * difference)[0][0];
};

Matrix calculateCostGradient(Matrix &result, Matrix &expected)
{
     cop::Matrix<double> difference = 2 * (result - expected);

     return difference;
};

Matrix getOutput(std::vector<cop::Matrix<double>> &layers, Matrix inputs)
{
     cop::Matrix<double> result = inputs;

     for(auto m: layers)
     {
          result = m * result;
     }

     return result;
     //return layer2 * layer1 * (layer0 * inputs);
}

double rateOfCostChangeWrt(std::vector<cop::Matrix<double>> &layers, Matrix &inputs, Matrix &expected, double &variable)
{
     const double inc = 0.0000001;

     auto result1 = getOutput(layers, inputs);
     double cost1 = calculateCost(result1, expected);

     variable += inc;

     auto result2 = getOutput(layers, inputs);
     double cost2 = calculateCost(result2, expected);

     double weightGradient = (cost2 - cost1) / inc;
     return weightGradient;
}


int main()
{
    Matrix inputs = ~Matrix{2, 3};
    Matrix layer0 = {{3, 1}, {2, 3}};
    Matrix layer1 = {{4, 5}, {3, 2}};
    Matrix layer2 = {{1, -3}, {-2, 2}};

    std::vector<cop::Matrix<double>> layers;

    layers.push_back(layer0);
    layers.push_back(layer1);
    layers.push_back(layer2);

    Matrix expected = ~Matrix{10, 11};

    for (auto i = 0; i < layers.size(); i++)
    {
        cout << "layer" << i << "\n"
             << layers[i] << endl;
    }

    cout << "input:\n"
         << inputs << endl;

    auto result = getOutput(layers, inputs);

    cout << "result:\n"
         << result << endl;
    cout << "expected:\n"
         << expected << endl;

    auto costGradients = calculateCostGradient(result, expected);

    cout << "cost gradients:\n"
         << costGradients << endl;

    double &ref = layers[0][0][0];

    double gradient = rateOfCostChangeWrt(layers, inputs, expected, ref);

    cout << "Cost gradient: " << gradient << endl;

    cout << "\nTest:\n"
         << ~layer1 * ~layer2 * costGradients * ~inputs << endl;

    cout << "\nTest:\n"
         << ~layer2 * costGradients * ~(layer0 * inputs) << endl;

    // layer0 gradients
    // ~layer1 * ~layer2 * ~layer3 * costGradients * ~input0

    // layer1 gradients
    // ~layer2 * ~layer3 * costGradients * ~input1

    // layer2 gradients
    // ~layer3 * costGradients * ~input2
}