#pragma once

#include <vector>

#include "matrix.h"

namespace cop
{
    class NeuralNetwork
    {
    private:
        std::vector<Matrix<double>> w_;
        std::vector<Matrix<double>> b_;

        double learningRate{0.01};

    public:
        NeuralNetwork(std::initializer_list<int> layerSizes);
        void fit(double *pInput, int numberInputs, double *pExpected);
    };
}