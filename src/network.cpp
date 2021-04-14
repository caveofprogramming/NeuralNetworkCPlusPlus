#include <random>
#include <time.h>

#include "network.h"


cop::Network::Network(std::initializer_list<int> layerSizes)
{
    srand(time(NULL));

    size_t n = 0;

    for (auto m : layerSizes)
    {
        if (n != 0)
        {
            cop::Matrix<double> layer(m, n, cop::Matrix<double>::signedRandomUnit);
            cop::Matrix<double> biases(m, 1, cop::Matrix<double>::signedRandomUnit);

            w_.push_back(layer);
            b_.push_back(biases);
        }

        n = m;
    }
}
