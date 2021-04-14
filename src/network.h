#pragma once

#include <vector>

#include "matrix.h"

namespace cop
{
    class Network
    {
    private:
        std::vector<Matrix<double>> w_;
        std::vector<Matrix<double>> b_;

        double learningRate{0.01};

    public:
        Network(std::initializer_list<int> layerSizes);
    };
}