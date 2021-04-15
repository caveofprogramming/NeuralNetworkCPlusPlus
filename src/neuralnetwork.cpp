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

void cop::NeuralNetwork::fit(double *pInput, int numberInputs, double *pExpected)
{
    int inputVectorSize = w_[0].rows();

    cop::Matrix input(inputVectorSize, 1);

    double *pInputVector = pInput;
    int numberInputBytes = sizeof(double) * inputVectorSize;

    for (int i = 0; i < numberInputs; i++)
    {
        input.setData(pInputVector, numberInputBytes);

        for(int layer = 0; layer < w_.size(); layer++)
        {
            auto &weights = w_[layer];
            auto &biases = b_[layer];
        }

        pInputVector += inputVectorSize;
    }
}
