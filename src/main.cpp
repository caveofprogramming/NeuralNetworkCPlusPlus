#include <iostream>
#include <random>
#include <time.h>

#include "imagedata.h"
#include "matrix.h"

int main()
{
    cop::ImageData imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";

    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    return 0;

    for (int epoch = 0; epoch < 20; epoch++)
    {
        for (int imageIndex = 0; imageIndex < imageData.getNumberImages(); imageIndex++)
        {
            
        }
    }

    return 0;
}