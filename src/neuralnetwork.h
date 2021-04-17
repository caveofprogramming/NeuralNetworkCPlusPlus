#pragma once

#include <vector>

#include "matrix.h"

namespace cop
{
    class NeuralNetwork
    {
    private:
        std::vector<Matrix> w_;
        std::vector<Matrix> b_;

        double learningRate{0.01};

    public:
        NeuralNetwork(std::initializer_list<int> layerSizes);
        void fit(double *pInput, int numberInputVectors, double *pExpected);
    };
}