#include "network.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>

std::ostream &cop::operator<<(std::ostream &out, const cop::Network &network)
{
    for (auto i = 0; i < network.w_.size(); i++)
    {
        out << std::noshowpos << "LAYER " << i << "\n";
        out << network.w_[i];
    }

    return out;
}

cop::Network::Network(std::initializer_list<size_t> layerSizes)
{
    srand(time(NULL));

    size_t n = 0;

    for (auto m : layerSizes)
    {
        if (n != 0)
        {
            cop::Matrix<double> layer(m, n, random);
            w_.push_back(layer);
        }

        n = m;
    }
}

double cop::Network::calculateCost(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
    auto difference = input - expected;

    return (~difference * difference)[0][0];
}

cop::Matrix<double> cop::Network::calculateCostGradients(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
    return 2 * (input - expected);
}

double cop::Network::random()
{
    return (int)(((5.0 * rand()) / RAND_MAX)) - 2.0;
    //return ((2.0 * rand()) / RAND_MAX) - 1.0;
}

void cop::Network::rateOfCostChangeWrt(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
    const double inc = 0.000001;
    double &wrt = w_[0][0][0];

    auto result1 = input;

    for (auto m : w_)
    {
        result1 = m * result1;
    }

    double cost1 = !(result1 - expected);

    wrt += inc;

    auto result2 = input;

    for (auto m : w_)
    {
        result2 = m * result2;
    }

    double cost2 = !(result2 - expected);

    std::cout << "Cost gradient: " << ((cost2 - cost1) / inc) << std::endl;
}

void cop::Network::run(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
}

void cop::Network::calculateLayerOutputs(cop::Matrix<double> *input, cop::Matrix<double> *expected)
{
    cop::Matrix<double> result = *input;

    outputs.clear();

    for (auto m : w_)
    {
        result = m * result;
        outputs.push_back(result);
    }

    auto &output = outputs.back();

    for (auto i = w_.size() - 1; i >= 0; i--)
    {
        auto m = w_[i];

        auto layerInput = i == 0 ? *input : w_[i - 1];

        std::cout << layerInput << std::endl;
    }
}
