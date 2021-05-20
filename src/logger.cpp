#include "logger.h"

cop::Logger logger;

void cop::Logger::log(int layer, cop::Matrix &weights, cop::Matrix &biases)
{
    std::unique_lock<std::mutex> lock(mtx_);

    out << "Layer " << layer << "\n";

    out << std::showpos << std::fixed << std::setprecision(5);

    for(int row = 0; row < weights.rows(); row++)
    {
        for(int col = 0; col < weights.cols(); col++)
        {
            out << std::setw(12) << weights[row][col];
        }

        out << " | " << biases[row][0] << "\n";
    }

    out << std::endl;
}