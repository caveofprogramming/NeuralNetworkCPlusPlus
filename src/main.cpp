#include <sstream>
#include "network.h"
#include "matrix.h"
#include "imageloader.h"


typedef cop::Matrix<double> Matrix;
using namespace std;

int main()
{
     auto images = cop::ImageLoader::loadImages("../MNIST/train-images-idx3-ubyte");
     auto labels = cop::ImageLoader::loadLabels("../MNIST/train-labels-idx1-ubyte");

     for(int i = 0; i < 100; i++)
     {
          stringstream ss;
          ss << "test" << i << ".bmp";

          images[i].save(ss.str());

          std::cout << i << ": " << labels[i] << std::endl;
     }


     /*
     cop::Network network{2, 3, 2, 4, 2};

     cop::Matrix<double> input = ~cop::Matrix({1.5, 2.0});
     cop::Matrix<double> expected = ~cop::Matrix({0.7, -2.3});

     //network.rateOfCostChangeWrt(input, expected);
     auto output1 = network.calculateOutput(&input, &expected);
     auto cost1 = (output1 - expected).magnitude();

     network.setLearningRate(10);
     network.setLearn(true);
     network.calculateOutput(&input, &expected);
     network.setLearn(false);

     auto output2 = network.calculateOutput(&input, &expected);
     auto cost2 = (output2 - expected).magnitude();

     std::cout << std::fixed;
     std::cout.precision(10);

     std::cout << "Cost improvement: " << (cost1 - cost2) << std::endl;
     std::cout << "Cost1: " << cost1 << std::endl;
     std::cout << "Cost2: " << cost2 << std::endl;
     */

     return 0;
}