#include <iostream>
#include <random>
#include <time.h>
#include <thread>
#include <sstream>
#include <string>

#include "neuralnetwork.h"
#include "imageloader.h"
#include "threadpool.h"
#include "matrix.h"

typedef std::vector<std::string> strings;

using namespace std::chrono_literals;

int main()
{
    cop::Matrix m1 = {
        {1, 2},
        {3, 4},
    };

    cop::Matrix m2 = {
        {1, 2},
        {3, 5},
    };

    std::cout << m1 + m2 << std::endl;

    //return 0;

    cop::ImageLoader imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";
    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    cop::NeuralNetwork network{imageData.getPixelsPerImage(), 256, 10};

    network.setBatchSize(128);
    network.setWorkers(std::thread::hardware_concurrency());
    network.setWorkers(6);
    network.fit(imageData.getImageData(), imageData.getNumberImages(), imageData.getLabelData());

    /*
    const int inputSize = 3;
    const int numberInputs = 20;

    float input[numberInputs * inputSize];
    float output[numberInputs];

    srand(time(nullptr));

    for(int i = 0; i < numberInputs; i++)
    {
        for(int j = 0; j < inputSize; j++)
        {
            input[i * inputSize + j] = float(rand())/RAND_MAX;
        }

        output[i] = float(rand())/RAND_MAX;
    }

    cop::NeuralNetwork network{inputSize, 4, 1};
    network.setBatchSize(3);
    network.fit(input, numberInputs, output);

    */

    return 0;
}