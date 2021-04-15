#include <sstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <time.h>

#include "neuralnetwork.h"
#include "matrix.h"
#include "imagedata.h"
typedef cop::Matrix<double> Matrix;
using namespace std;

/*
int main()
{
     Matrix m1{
          {2, 3},
          {4, 2}
     };

     Matrix m2{
          {5, 6},
          {6, 7}
     };

     time_t start = time(&start);

     for (int i = 0; i < 1E8; i++)
     {
          auto m3 = m1 * m2;
     }

     time_t end = time(&end);
     cout << end - start << " seconds" << endl;
}
*/

char matrixToChar(cop::Matrix<double> &m)
{
     char value = 0;

     for (auto i = 0; i < m.rows(); i++)
     {
          value <<= 1;

          if (abs(m[i][0] - 1.0) < 0.1)
          {
               value += 1;
          }
     }

     return value;
}

cop::Matrix<double> charToMatrix(char value)
{
     cop::Matrix<double> m(10, 1);

     for (auto i = 0; i < 8; i++)
     {
          m[7 - i][0] = value & 1;

          value >>= 1;
     }

     return m;
}

int main()
{
     int width;
     int height;

     std::cout << "Loading training data ..." << std::endl;

     std::vector<cop::Image> images = cop::ImageLoader::loadImages("/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/train-images-idx3-ubyte", width, height);
     cop::ImageLoader::loadLabels(images, "/Users/john/Projects/NeuralNetworkCPlusPlus/MNIST/train-labels-idx1-ubyte");

     unsigned long imageSize = width * height;

     cop::Network network{imageSize, 20, 10};

     network.setLearningRate(0.1);

     std::cout << "Training ...." << std::endl;

     int count = 0;

     double lossTotal = 0;

     for (int i = 0; i < 20; i++)
     {
          for (auto &image : images)
          {
               cop::Matrix<double> input(image.size(), 1, [&image](int row, int col) {
                    return image[row] / 255.0;
               });

               auto expected = charToMatrix(image.getLabel());

               auto output = network.run(&input, &expected);

               auto loss = (output - expected).magnitude() / (2 * output.rows());

               lossTotal += loss;

               if (count % 1000 == 0)
               {
                    std::cout << "\n" << i << ": "
                              << count << "; loss: " << lossTotal / 100.0 << std::endl;
                    lossTotal = 0;
               }

               count++;
          }
     }

     std::cout << "Loading test images ..." << std::endl;

     //images.clear();

     //images = cop::ImageLoader::loadImages("../MNIST/t10k-images-idx3-ubyte", width, height);
     //cop::ImageLoader::loadLabels(images, "../MNIST/t10k-labels-idx1-ubyte");

     std::cout << "Identifying characters ...." << std::endl;

     int correct = 0;
     int incorrect = 0;
     count = 0;

     for (auto image : images)
     {
          cop::Matrix<double> input(image.size(), 1, [&image](int row, int col) {
               return image[row] / 255.0;
          });

          char label = image.getLabel();

          auto output = network.run(&input);

          char result = matrixToChar(output);

          if (result == label)
          {
               correct++;
          }
          else
          {
               incorrect++;
          }

          if (count % 100 == 0)
          {
               std::cout << count << std::endl;
          }

          count++;
     }

     std::cout << "Correct: " << correct << "; incorrect: " << incorrect << std::endl;

     return 0;
}

