#pragma once

#include <vector>
#include "matrix.h"

namespace cop
{
    class Network
    {
    private:
        std::vector<Matrix<double>> w_;
        std::vector<Matrix<double>> b_;
    };
}