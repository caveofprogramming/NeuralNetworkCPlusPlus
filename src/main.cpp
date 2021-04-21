#include <iostream>
#include <random>
#include <time.h>
#include <thread>
#include <sstream>
#include <string>

#include "neuralnetwork.h"
#include "imagedata.h"
#include "threadpool.h"
#include "matrix.h"

typedef std::vector<std::string> strings;

using namespace std::chrono_literals;

class Test
{
public:
    std::vector<std::string> hello(int i, std::string name)
    {
        std::vector<std::string> values;

        std::stringstream ss;

        ss << i << ": " << name;
        values.push_back(ss.str());

        std::this_thread::sleep_for(5s);

        return values;
    };
};

int main()
{
    Test test;

    cop::ThreadPool<strings> threadPool(std::thread::hardware_concurrency());

    for(int i = 0; i < 10; i++)
    {
        auto task = std::bind(&Test::hello, &test, i, "Bob");
        threadPool.submit(task);
    }

    threadPool.start();

    for(int i = 0; i < 10; i++)
    {
        auto result = threadPool.get();

        for(auto s: result)
        {
            //std::cout << i << ": " << s << std::endl;
        }
    }

    threadPool.awaitComplete();
    return 0;


    auto init = [](int, int) {
        return 0.0;
    };

    cop::ImageData imageData;

    std::string directory = "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/";
    imageData.load(directory + "train-images-idx3-ubyte", directory + "train-labels-idx1-ubyte");

    cop::NeuralNetwork network{imageData.getPixelsPerImage(), 256, 10};

    network.setBatchSize(128);
    network.fit(imageData.getImageData(), imageData.getNumberImages(), imageData.getLabelData());

    /*
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

    */

    return 0;
}