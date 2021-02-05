#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>
#include "matrix.h"

namespace cop
{
    class Network
    {
    private:
        std::vector<Matrix<double>> w_;
        std::vector<Matrix<double>> b_;
        std::vector<Matrix<double>> outputs;

    protected:
        static double random();
        double calculateCost(cop::Matrix<double> &input, cop::Matrix<double> &expected);

    public:
        void rateOfCostChangeWrt(cop::Matrix<double> &input, cop::Matrix<double> &expected);

        Network(std::initializer_list<size_t> layerSizes);
        void calculateLayerOutputs(cop::Matrix<double> *input, cop::Matrix<double> *expected = nullptr);
        void run(cop::Matrix<double> &input, cop::Matrix<double> &expected);
        cop::Matrix<double> calculateCostGradients(cop::Matrix<double> &input, cop::Matrix<double> &expected);

        friend std::ostream &operator<<(std::ostream &out, const cop::Network &network);
    };

} // namespace cop