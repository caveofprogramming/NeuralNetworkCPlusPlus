#include "logger.h"

cop::Logger logger;

void cop::Logger::log(std::string label, std::vector<cop::Matrix> &weights, std::vector<cop::Matrix> &biases)
{
    for(int i = 0; i < weights.size(); i++)
    {
        auto &w = weights[i];
        auto &b = biases[i];

        log(label, i, w, b);
    }
}

void cop::Logger::log(std::string label, int layer, cop::Matrix &weights, cop::Matrix &biases)
{
    std::unique_lock<std::mutex> lock(mtx_);

    out << label << "\n";
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