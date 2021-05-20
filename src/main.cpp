#include <iostream>
#include <random>
#include <time.h>
#include <thread>
#include <sstream>
#include <string>
#include <functional>
#include <cmath>

#include "logger.h"

#include "neuralnetwork.h"
#include "imageloader.h"
#include "threadpool.h"
#include "matrix.h"

typedef std::vector<std::string> strings;

using namespace std::chrono_literals;

int main()
{
    const int items = 10000;
    const int inputSize = 2;
    const int outputSize = 4;

    std::vector<float> input;
    std::vector<float> expected(items * 4);

    for (int i = 0; i < items; i++)
    {
        double sum = 0.0;

        for (int k = 0; k < inputSize; k++)
        {
            double value = float(rand()) / RAND_MAX;

            input.push_back(value);

            sum += value;
        }

        int category = int(sum * 2);
        expected[(i * 4) + category] = 1;
    }

    cop::NeuralNetwork network{inputSize, 4, outputSize};

    network.setBatchSize(8);
    network.setWorkers(std::thread::hardware_concurrency());
    network.setWorkers(1);

    network.fit(input.data(), items, expected.data());
}

/*
int main()
{
    cop::ImageLoader imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";
    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    cop::NeuralNetwork network{imageData.getPixelsPerImage(), 256, 10};

    network.setBatchSize(128);
    network.setWorkers(std::thread::hardware_concurrency());
    network.setWorkers(6);

    network.fit(imageData.getImageData(), imageData.getNumberImages(), imageData.getLabelData());

    return 0;
}

*/
