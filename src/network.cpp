#include "network.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <deque>
#include <math.h>

std::ostream &cop::operator<<(std::ostream &out, const cop::Network &network)
{
    for (auto i = 0; i < network.w_.size(); i++)
    {
        out << std::noshowpos << "LAYER " << i << "\n";
        out << network.w_[i] << std::endl;
        out << network.b_[i] << std::endl;
    }

    return out;
}

double cop::Network::sigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

cop::Network::Network(std::initializer_list<size_t> layerSizes)
{
    //srand(time(NULL));

    size_t n = 0;

    for (auto m : layerSizes)
    {
        if (n != 0)
        {
            cop::Matrix<double> layer(m, n, cop::Matrix<double>::signedRandomUnit);
            cop::Matrix<double> biases(m, 1, cop::Matrix<double>::signedRandomUnit);

            w_.push_back(layer);
            b_.push_back(biases);
        }

        n = m;
    }
}

double cop::Network::calculateCost(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
    auto difference = input - expected;

    return 0.5 * ((~difference * difference)[0][0]);
}

cop::Matrix<double> cop::Network::calculateCostGradients(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
    return (input - expected);
}

void cop::Network::rateOfCostChangeWrt(cop::Matrix<double> &input, cop::Matrix<double> &expected)
{
    const double inc = 0.00001;
    double &wrt = w_[0][0][0];

    std::vector<Matrix<double>> outputs;

    calculateOutput(outputs, input);
    double cost1 = (outputs.back() - expected).magnitude() / 2.0;
    auto activations1 = outputs.back();

    wrt += inc;

    calculateOutput(outputs, input);

    double cost2 = (outputs.back() - expected).magnitude() / 2.0;

    auto rate = (cost2 - cost1) / inc;

    std::cout.precision(5);
    std::cout << std::fixed;

    std::cout << "Cost...:\n"
              << rate << std::endl;
}

void cop::Network::calculateOutput(std::vector<cop::Matrix<double>> &outputs, cop::Matrix<double> &input)
{
    auto result = input;

    outputs.clear();

    for (auto i = 0; i < w_.size(); i++)
    {
        auto &m = w_[i];
        auto &b = b_[i];

        result = (m * result + b).transform(Network::sigmoid);
        outputs.push_back(result);
    }
}

void cop::Network::learn(std::vector<Matrix<double>> &outputs, cop::Matrix<double> *input, cop::Matrix<double> *expected)
{
    auto outputRateTransform = [](double value) { return value * (1.0 - value); };

    auto layerError = (outputs.back() - *expected) % outputs.back().transform((outputRateTransform));

    for (int i = w_.size() - 1; i >= 0; i--)
    {
        auto layerOutput = outputs[i];
        auto layerInput = i == 0 ? *input : outputs[i - 1];
        auto &weights = w_[i];
        auto &biases = b_[i];

        auto layerActivationRates = layerInput.transform(outputRateTransform);

        auto weightGradients = layerError * ~layerInput;
        
        biases -= learningRate * layerError;

        if (i > 0)
        {
            layerError = (~weights * layerError) % layerActivationRates;
        }

        weights -= learningRate * weightGradients;
    }
}

cop::Matrix<double> cop::Network::run(cop::Matrix<double> *input, cop::Matrix<double> *expected)
{
    std::vector<Matrix<double>> outputs;

    calculateOutput(outputs, *input);

    if (expected != nullptr)
    {
        learn(outputs, input, expected);
    }

    return outputs.back();
}
