#include <random>
#include <time.h>

#include "neuralnetwork.h"

cop::NeuralNetwork::NeuralNetwork(std::initializer_list<int> layerSizes)
{
    srand(time(NULL));

    auto init = [](int, int) {
        return 0.0;
    };

    size_t n = 0;

    for (auto m : layerSizes)
    {
        if (n != 0)
        {
            w_.push_back(cop::Matrix(m, n, init));
            b_.push_back(cop::Matrix(m, 1, init));
        }

        n = m;
    }
}

void cop::NeuralNetwork::fit(double *pInput, int numberInputVectors, double *pExpected)
{
    int inputRows = w_[0].cols();

    double *pInputVector = pInput;

    std::vector<cop::Matrix> layerIo;

    for(int i = 0; i < w_.size(); i++)
    {
        layerIo.push_back(cop::Matrix(w_[i].cols(), 1));
    }

    layerIo[0].setData(pInput, inputRows * sizeof(double));
    layerIo.push_back(cop::Matrix(w_.back().rows(), 1));

    for (int i = 0; i < numberInputVectors; i++)
    {
        if(i % 100 == 0) std::cout << "." << std::flush;
        for(int layer = 0; layer < w_.size(); layer++)
        {
            auto &weights = w_[layer];
            auto &biases = b_[layer];
            auto &input = layerIo[layer];
            auto &output = layerIo[layer + 1];

            weights.multiply(output, input);
        }

        pInputVector += inputRows;
    }
}
