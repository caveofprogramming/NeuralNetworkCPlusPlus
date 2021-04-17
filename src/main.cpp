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

    cop::NeuralNetwork network{imageData.getPixelsPerImage(), 256, 10};

    network.fit(imageData.getImageData(), imageData.getNumberImages(), imageData.getLabelData());

    return 0;
}