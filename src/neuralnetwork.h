#pragma once

#include <vector>

#include "matrix.h"
#include "threadpool.h"

namespace cop
{
    class NeuralNetwork
    {
    private:
        std::vector<Matrix> w_;
        std::vector<Matrix> b_;

        //float learningRate_ = 0.01;
        int batchSize_ = 1;
        int epochs_ = 20;
        int logInterval_ = 1;

        std::ostream &log_  = std::cout;

    protected:
        void runEpoch(float *pInput, int numberInputVectors, float *pExpected);
        int runBatch(float *pInput, int numberInputVectors, float *pExpected);
        void computeOutputs(std::vector<cop::Matrix> &layerIo);

    public:
        NeuralNetwork(std::initializer_list<int> layerSizes);
        void fit(float *pInput, int numberInputVectors, float *pExpected);
        void setBatchSize(int batchSize) { batchSize_ = batchSize; }
    };
}