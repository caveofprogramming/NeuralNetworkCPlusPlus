#include <iostream>
#include <random>

#include "imagedata.h"
#include "matrix.h"

int main() 
{
    cop::ImageData imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";

    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    for(int i = 0; i < 10; i++)
    {
        int index = int(imageData.getNumberImages() * double(rand())/RAND_MAX);

        std::cout << index << std::endl;
        imageData.save(index);
    }

    return 0;
}