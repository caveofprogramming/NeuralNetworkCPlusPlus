#include "network.h"
#include "matrix.h"

int main()
{
     cop::Network network = {2, 3, 2};

     std::cout << network << std::endl;

     cop::Matrix<double> input = ~cop::Matrix<double>{2, 3};
     cop::Matrix<double> expected = ~cop::Matrix<double>{3, 6};

     network.calculateLayerOutputs(&input, &expected);

     network.rateOfCostChangeWrt(input, expected);

     return 0;
}