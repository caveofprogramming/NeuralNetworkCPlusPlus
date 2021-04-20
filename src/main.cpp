#include <iostream>
#include <random>
#include <time.h>

#include "neuralnetwork.h"
#include "imagedata.h"
#include "matrix.h"

int main()
{
    auto init = [](int, int) {
        return 0.0;
    };

    cop::ImageData imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";
    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    //cop::NeuralNetwork network{imageData.getPixelsPerImage(), 256, 10};

    //network.setBatchSize(128);
    //network.fit(imageData.getImageData(), imageData.getNumberImages(), imageData.getLabelData());

    const int inputSize = 3;
    const int numberInputs = 20;

    double input[numberInputs * inputSize];
    double output[numberInputs];

    srand(time(nullptr));

    for(int i = 0; i < numberInputs; i++)
    {
        for(int j = 0; j < inputSize; j++)
        {
            input[i * inputSize + j] = double(rand())/RAND_MAX;
        }

        output[i] = double(rand())/RAND_MAX;
    }

    cop::NeuralNetwork network{inputSize, 4, 1};
    network.setBatchSize(3);
    network.fit(input, numberInputs, output);

    return 0;
}