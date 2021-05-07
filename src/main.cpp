#include <iostream>
#include <random>
#include <time.h>
#include <thread>
#include <sstream>
#include <string>
#include <functional>
#include <cmath>

#include "neuralnetwork.h"
#include "imageloader.h"
#include "threadpool.h"
#include "matrix.h"

typedef std::vector<std::string> strings;

using namespace std::chrono_literals;

/*
int main()
{
    const int items = 10000;
    const int inputSize = 4;
    const int outputSize = 3;

    std::vector<float> input;
    std::vector<float> expected;

    for (int i = 0; i < items; i++)
    {
        double sum = 0.0;

        for (int k = 0; k < inputSize; k++)
        {
            double value = float(rand()) / RAND_MAX;

            input.push_back(value);

            sum += value;
        }

        for (int k = 0; k < outputSize; k++)
        {
            expected.push_back(cos(k * sum));
        }
    }

    cop::NeuralNetwork network{inputSize, 4, outputSize};

    network.setBatchSize(8);
    network.setWorkers(std::thread::hardware_concurrency());
    network.setWorkers(6);

    network.fit(input.data(), inputSize, expected.data());
}
*/

int main()
{
    cop::ImageLoader imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";
    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    for (int i = 0; i < 10; i++)
    {
        int index = (imageData.getNumberImages() - 1) - i;
        std::cout << "Writing " << index << std::endl;
        imageData.save(index);
    }

    return 0;

    cop::NeuralNetwork network{imageData.getPixelsPerImage(), 256, 10};

    network.setBatchSize(128);
    network.setWorkers(std::thread::hardware_concurrency());
    network.setWorkers(6);

    network.fit(imageData.getImageData(), imageData.getNumberImages(), imageData.getLabelData());

    return 0;
}
