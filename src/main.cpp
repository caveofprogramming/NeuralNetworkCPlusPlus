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

/*

std::vector<double> calculateActivations(std::vector<double> zs)
{
    // Softmax
    double total = 0.0;

    for (auto z : zs)
    {
        total += exp(z);
    }

    std::vector<double> as;

    for (auto z : zs)
    {
        as.push_back(exp(z) / total);
    }

    return as;
}

double calculateLoss(std::vector<double> zs, std::vector<double> expected)
{
    auto as = calculateActivations(zs);

    // Categorical cross-entropy
    double loss = 0.0;

    for (int i = 0; i < as.size(); i++)
    {
        loss += -expected[i] * log(as[i]);
    }

    return loss;
}

std::vector<double> normalisedVector(int entries)
{
    std::vector<double> normalised;

    double sum = 0;

    for (int i = 0; i < entries - 1; i++)
    {
        double value = (double(rand()) / entries) / RAND_MAX;
        sum += value;
        normalised.push_back(value);
    }

    normalised.push_back(1.0 - sum);

    return normalised;
}

int main()
{
    const int entries = 6;
    std::vector<double> zs = normalisedVector(entries);
    std::vector<double> expected = normalisedVector(entries);

    std::vector<double> rates;

    const double inc = 0.000001;

    for (int i = 0; i < zs.size(); i++)
    {
        double loss1 = calculateLoss(zs, expected);
        zs[i] += inc;
        double loss2 = calculateLoss(zs, expected);

        rates.push_back((loss2 - loss1) / inc);

        zs[i] -= inc;
    }

    // Check softmax derivative.
    const int x = 1;
    const int y = 2;

    auto a1 = calculateActivations(zs);
    zs[y] += inc;
    auto a2 = calculateActivations(zs);
    zs[y] -= inc;

    double rate = (a2[x] - a1[x]) / inc;
    std::cout << "activation " << x << " wrt " << y << ": " << rate << std::endl;
    std::cout << "calculated: " << -a1[x] * a1[y] << std::endl;

    for (auto loss : rates)
    {
        std::cout << loss << " ";
    }

    std::cout << std::endl;

    std::vector<double> calculated;

    auto as = calculateActivations(zs);

    for (int i = 0; i < zs.size(); i++)
    {
        calculated.push_back(as[i] - expected[i]);
    }

    for (auto c : calculated)
    {
        std::cout << c << " ";
    }

    std::cout << std::endl;

    return 0;
}

*/

int main()
{
    cop::Matrix m1{{1, 2}, {3, 4}};
    cop::Matrix m2{{1, 2}};

    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
    std::cout << m1 * m2 << std::endl;

    //return 0;
    //srand(time(nullptr));

    const int items = 1;
    const int inputSize = 2;
    const int outputSize = 2;

    std::vector<double> input;
    std::vector<double> expected(outputSize);

    for (int i = 0; i < items; i++)
    {
        double sum = 0.0;

        for (int k = 0; k < inputSize; k++)
        {
            rand();
            double value = double(rand()) / RAND_MAX;

            input.push_back(value);

            sum += value;
        }

        int category = outputSize * double(rand())/RAND_MAX;
        expected[category] = 1;
    }

    std::cout << "\nInput\n" << std::endl;
    for (auto s : input)
    {
        std::cout << s << " ";
    }

    std::cout << std::endl;

    std::cout << "Expected:\n";
    for (auto s : expected)
    {
        std::cout << s << " ";
    }

    std::cout << "\n" << std::endl;

    cop::NeuralNetwork network{inputSize, outputSize};

    network.writeLog();
    network.setBatchSize(1);
    network.setEpochs(1);
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
