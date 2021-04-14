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

        double learningRate{0.01};

    protected:
        static double random(int row, int col);
        static double sigmoid(double input);
        double calculateCost(cop::Matrix<double> &input, cop::Matrix<double> &expected);
        void learn(std::vector<Matrix<double>> &outputs, cop::Matrix<double> *input, cop::Matrix<double> *expected);
        void calculateOutput(std::vector<cop::Matrix<double>> &output, cop::Matrix<double> &input);

    public:
        void setLearningRate(double rate) { learningRate = rate; };
        void rateOfCostChangeWrt(cop::Matrix<double> &input, cop::Matrix<double> &expected);

        Network(std::initializer_list<size_t> layerSizes);
        cop::Matrix<double>  run(cop::Matrix<double> *input, cop::Matrix<double> *expected = nullptr);
        cop::Matrix<double> calculateCostGradients(cop::Matrix<double> &input, cop::Matrix<double> &expected);

        friend std::ostream &operator<<(std::ostream &out, const cop::Network &network);
    };

} // namespace cop