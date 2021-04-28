#include <random>
#include <time.h>
#include <thread>
#include <chrono>
#include <cmath>
#include "threadpool.h"
#include "activations.h"
#include "neuralnetwork.h"

using namespace std::chrono;

cop::NeuralNetwork::NeuralNetwork(std::initializer_list<int> layerSizes)
{
    srand(time(NULL));

    auto init = [](int, int) {
        return 0.0;
    };

    size_t n = 0;

    for (auto m : layerSizes)
    {
        if (n != 0)
        {
            w_.push_back(cop::Matrix(m, n, init));
            b_.push_back(cop::Matrix(m, 1, init));
        }

        n = m;
    }
}

void cop::NeuralNetwork::computeOutputs(std::vector<cop::Matrix> &layerIo, const cop::Matrix &expected)
{
    for (auto layer = 0; layer < int(w_.size()); layer++)
    {
        auto &weights = w_[layer];
        auto &biases = b_[layer];
        auto &input = layerIo[layer];
        auto &output = layerIo[layer + 1];

        output = (weights * input) + biases;
        cop::softmax(output.data(), output.rows());
    }

    //double loss = computeLoss(layerIo.back(), expected);
    //std::cout << loss << std::endl;
}

int cop::NeuralNetwork::runBatch(int sequence, float *pInput, int numberInputVectors, float *pExpected)
{
    int inputRows = w_[0].cols();
    int outputRows = w_.back().rows();

    float *pInputVector = pInput;
    float *pExpectedVector = pExpected;

    std::vector<cop::Matrix> layerIo;
    cop::Matrix expected(outputRows, 1);

    for (int i = 0; i < int(w_.size()); i++)
    {
        layerIo.push_back(cop::Matrix(w_[i].cols(), 1));
    }

    layerIo.push_back(cop::Matrix(w_.back().rows(), 1));

    for (int i = 0; i < numberInputVectors; i++)
    {
        layerIo[0].setData(pInput, inputRows * sizeof(float));
        expected.setData(pExpectedVector, outputRows);

        computeOutputs(layerIo, expected);

        pInputVector += inputRows;
        pExpectedVector += outputRows;
    }

    if (sequence % logInterval_ == 0)
    {
        log_ << "." << std::flush;
    }

    return 0;
}

float cop::NeuralNetwork::computeLoss(const Matrix &actual, const Matrix &expected)
{
    float sum = 0.0;

    for (int i = 0; i < actual.rows(); i++)
    {
        sum -= expected[i][0] * std::log(actual[i][0]);
    }

    return sum;
}

void cop::NeuralNetwork::runEpoch(float *pInput, int numberInputVectors, float *pExpected)
{
    steady_clock::time_point t1 = high_resolution_clock::now();

    const int inputVectorSize = w_[0].cols();

    const int numberBatches = numberInputVectors / batchSize_;
    const int lastBatchSize = numberInputVectors % batchSize_;

    float *pBatchInput = pInput;
    float *pBatchExpected = pExpected;
    const int outputSize = w_.back().rows();
    int batchSize = batchSize_;

    cop::ThreadPool<int> threadPool(workers_);

    for (int i = 0; i < numberBatches; ++i)
    {
        if (i == numberBatches - 1 && lastBatchSize != 0)
        {
            batchSize = lastBatchSize;
        }

        auto work = [&]() {
            return runBatch(i, pBatchInput, batchSize, pBatchExpected);
        };

        threadPool.submit(work);

        pBatchInput += (inputVectorSize * batchSize_);
        pBatchExpected += outputSize * batchSize_;
    }

    threadPool.start();

    for (int i = 0; i < numberBatches; ++i)
    {
        auto result = threadPool.get();

        // TODO unused variable warning disabled by useless code.
        result = 0;
    }

    threadPool.awaitComplete();

    steady_clock::time_point t2 = high_resolution_clock::now();
    duration<float, std::milli> duration = t2 - t1;

    log_ << std::endl
         << duration.count() << " ms" << std::endl;
}

void cop::NeuralNetwork::fit(float *pInput, int numberInputVectors, float *pExpected)
{
    log_ << "Batch size: " << batchSize_ << std::endl;
    log_ << "Number of inputs: " << numberInputVectors << std::endl;
    log_ << std::endl;

    steady_clock::time_point t1 = high_resolution_clock::now();

    logInterval_ = int(numberInputVectors / (80.0 * batchSize_)) + 1;

    for (int i = 0; i < epochs_; i++)
    {
        log_ << "Epoch: " << i + 1 << " " << std::flush;

        runEpoch(pInput, numberInputVectors, pExpected);

        log_ << std::endl;
    }

    steady_clock::time_point t2 = high_resolution_clock::now();

    duration<float, std::milli> duration = t2 - t1;

    std::cout << "\nCompleted in " << duration.count() / 1000 << " seconds" << std::endl;
}
