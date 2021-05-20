#pragma once

#include <vector>
#include <mutex>

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
        int workers_ = 1;

        std::mutex mtxLog_;

        std::ostream &log_  = std::cout;

    protected:
        void runEpoch(float *pInput, int numberInputVectors, float *pExpected);
        int runBatch(int sequence, float *pInput, int numberInputVectors, float *pExpected);
        void computeOutputs(std::vector<cop::Matrix> &layerIo);
        void computeDeltas(std::vector<cop::Matrix> &layerIo, const cop::Matrix &expected);
        void computeDeltasSlow(std::vector<cop::Matrix> &layerIo, const cop::Matrix &expected);

    public:

        NeuralNetwork(std::initializer_list<int> layerSizes);
        void fit(float *pInput, int numberInputVectors, float *pExpected);
        void setBatchSize(int batchSize) { batchSize_ = batchSize; }
        void setWorkers(int workers) { workers_ = workers; };
        double computeLoss(const Matrix &actual, const Matrix &expected);

        friend std::ostream &operator<<(std::ostream &out, const cop::NeuralNetwork &network);
    };
}