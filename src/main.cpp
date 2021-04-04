#include <sstream>
#include <vector>
#include <cmath>
#include "network.h"
#include "matrix.h"
#include "imageloader.h"

typedef cop::Matrix<double> Matrix;
using namespace std;

char matrixToChar(cop::Matrix<double> &m)
{
     char value = 0;

     for(auto i = 0; i < m.rows(); i++)
     {
          value <<= 1;
         
          if(abs(m[i][0] - 1.0) < 0.1)
          {
               value += 1;
          }
     }

     return value;
}

cop::Matrix<double> charToMatrix(char value)
{
     cop::Matrix<double> m(10, 1);

     for(auto i = 0; i < 8; i++)
     {
          m[7-i][0] = value & 1;

          value >>= 1;
     }

     return m;
}

int main()
{
     int width;
     int height;

     std::cout << "Loading training data ..." << std::endl;

     std::vector<cop::Image> images = cop::ImageLoader::loadImages("../MNIST/train-images-idx3-ubyte", width, height);
     cop::ImageLoader::loadLabels(images, "../MNIST/train-labels-idx1-ubyte");

     unsigned long imageSize = width * height;

     cop::Network network{imageSize, 10, 10};

     network.setLearningRate(0.1);

     std::cout << "Training ...." << std::endl;

     int count = 0;

     for (auto &image : images)
     {
          cop::Matrix<double> input(image.size(), 1, [&image](int row, int col) {
               return image[row];
          });

          auto expected = charToMatrix(image.getLabel());

          network.run(&input, &expected);

          if(count % 100 == 0)
          {
               std::cout << count << std::endl;
          }

          count++;
     }

     std::cout << "Loading test images ..." << std::endl;

     images = cop::ImageLoader::loadImages("../MNIST/t10k-images-idx3-ubyte", width, height);
     cop::ImageLoader::loadLabels(images, "../MNIST/t10k-labels-idx1-ubyte");

     std::cout << "Identifying characters ...." << std::endl;

     int correct = 0;
     int incorrect = 0;
     count = 0;

     for (auto &image : images)
     {
          cop::Matrix<double> input(image.size(), 1, [&image](int row, int col) {
               return image[row];
          });

          char label = image.getLabel();

          auto output = network.run(&input);

          if(matrixToChar(output) == label)
          {
               correct++;
          }
          else 
          {
               incorrect++;
          }

          if(count % 100 == 0)
          {
               std::cout << count << std::endl;
          }

          count++;
     }

     std::cout << "Correct: " << correct << "; incorrect: " << incorrect << std::endl;

     return 0;
}