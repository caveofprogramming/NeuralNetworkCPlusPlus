#include <random>
#include <time.h>
#include <thread>
#include <chrono>
#include <cmath>
#include "threadpool.h"
#include "activations.h"
#include "neuralnetwork.h"
#include "logger.h"

using namespace std::chrono;

cop::NeuralNetwork::NeuralNetwork(std::initializer_list<int> layerSizes)
{
    //srand(time(NULL));

    auto init = [](int, int)
    {
        return (2.0 * rand()) / RAND_MAX - 1;
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

void cop::NeuralNetwork::writeLog()
{
    logger.log("Network", w_, b_);
}

void cop::NeuralNetwork::computeOutputs(std::vector<cop::Matrix> &layerIo)
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
}

void cop::NeuralNetwork::showGradients(std::vector<cop::Matrix> &layerIo, const cop::Matrix &expected)
{
    computeOutputs(layerIo);
    auto loss = computeLoss(layerIo.back(), expected);

    std::cout.precision(10);
    std::cout << "\nLOSS1  " << loss << std::endl;
    const double inc = 0.0000001;

/*
    b_[0][0][0] += inc;

    computeOutputs(layerIo);
    auto loss2 = computeLoss(layerIo.back(), expected);

    std::cout << "\nLOSS2  " << loss2 << std::endl;

    std::cout << "\nRATE: " << (loss2 - loss)/inc << std::endl;

    b_[0][0][0] -= inc;
*/

    for (int layer = 0; layer < w_.size(); layer++)
    {
        auto &weights = w_[layer];
        auto &biases = b_[layer];

        auto weightRates = Matrix(weights.rows(), weights.cols());
        auto biasRates = Matrix(biases.rows(), 1);

        for (int row = 0; row < weights.rows(); row++)
        {
            for (int col = 0; col < weights.cols(); col++)
            {
                double weightValue = weights[row][col];

                weights[row][col] = weightValue + inc;

                computeOutputs(layerIo);
                auto newLoss = computeLoss(layerIo.back(), expected);

                weightRates[row][col] = (newLoss - loss) / inc;

                weights[row][col] = weightValue;
            }

            double biasValue = biases[row][0];

            biases[row][0] = biasValue + inc;

            computeOutputs(layerIo);
            auto newLoss = computeLoss(layerIo.back(), expected);

            biasRates[row][0] = (newLoss - loss) / inc;

            biases[row][0] = biasValue;
        }

        logger.log("Computed gradients", layer, weightRates, biasRates);
    }

    // Recompute original layerIo
    computeOutputs(layerIo);
    auto loss2 = computeLoss(layerIo.back(), expected);

    std::cout << "Should be zero: " << (loss2 - loss) << std::endl;

}

void cop::NeuralNetwork::computeDeltas(std::vector<cop::Matrix> &deltas, std::vector<cop::Matrix> &layerIo, const cop::Matrix &expected)
{
    auto loss = computeLoss(layerIo.back(), expected);

    auto errors = layerIo.back() - expected;

    std::cout << "\nOutput:\n"
              << layerIo.back() << std::endl;

    double sum = 0;

    auto &output = layerIo.back();

    for (int i = 0; i < output.rows(); i++)
    {

        sum += output[i][0];
    }

    std::cout << "sum: " << sum << std::endl;

    std::cout << "Errors: \n"
              << errors << std::endl;
    
    deltas[0] = std::move(errors);
}

int cop::NeuralNetwork::runBatch(int sequence, double *pInput, int numberInputVectors, double *pExpected)
{
    int inputRows = w_[0].cols();
    int outputRows = w_.back().rows();

    double *pInputVector = pInput;
    double *pExpectedVector = pExpected;

    std::vector<cop::Matrix> layerIo;
    cop::Matrix expected(outputRows, 1);

    for (int i = 0; i < int(w_.size()); i++)
    {
        layerIo.push_back(cop::Matrix(w_[i].cols(), 1));
    }

    layerIo.push_back(cop::Matrix(w_.back().rows(), 1));

    // We will use this to sum up all the layer errors
    // from this batch, before applying them.
    std::vector<cop::Matrix> deltas;

    for (int i = 0; i < w_.size(); i++)
    {
        deltas.push_back(cop::Matrix(w_[i].rows(), 1));
    }

    for (int i = 0; i < numberInputVectors; i++)
    {
        std::unique_lock<std::mutex> mtxLog_;

        layerIo[0].setData(pInput, inputRows * sizeof(double));
        expected.setData(pExpectedVector, outputRows * sizeof(double));

        computeOutputs(layerIo);
        showGradients(layerIo, expected);
        computeDeltas(deltas, layerIo, expected);

        std::cout << "\nDeltas:\n" << deltas[0] << std::endl;
        std::cout << "\nLayer input:\n" << layerIo[0] << std::endl;
        std::cout << "\nCalculated gradients:\n" << deltas[0] * ~layerIo[0] << std::endl;

        pInputVector += inputRows;
        pExpectedVector += outputRows;
    }

    if (logInterval_ != 0 && sequence % logInterval_ == 0)
    {
        std::unique_lock<std::mutex> lock(mtxLog_);
        log_ << "." << std::flush;
    }

    return 0;
}

double cop::NeuralNetwork::computeLoss(const Matrix &actual, const Matrix &expected)
{
    // Categorical cross-entropy
    for (int i = 0; i < actual.rows(); i++)
    {
        if (expected[i][0] > 0.1)
        {
            return -std::log(actual[i][0]);
        }
    }

    return 0.0;
}

void cop::NeuralNetwork::runEpoch(double *pInput, int numberInputVectors, double *pExpected)
{

    steady_clock::time_point t1 = high_resolution_clock::now();

    const int inputVectorSize = w_[0].cols();

    const int numberBatches = numberInputVectors / batchSize_;
    const int lastBatchSize = numberInputVectors % batchSize_;

    double *pBatchInput = pInput;
    double *pBatchExpected = pExpected;
    const int outputSize = w_.back().rows();
    int batchSize = batchSize_;

    cop::ThreadPool<int> threadPool(workers_);

    for (int i = 0; i < numberBatches; ++i)
    {
        if (i == numberBatches - 1 && lastBatchSize != 0)
        {
            batchSize = lastBatchSize;
        }

        auto work = [this, i, pBatchInput, batchSize, pBatchExpected]()
        {
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
    duration<double, std::milli> duration = t2 - t1;

    log_ << " " << duration.count() << " ms" << std::endl;
}

void cop::NeuralNetwork::fit(double *pInput, int numberInputVectors, double *pExpected)
{

    log_ << "Batch size: " << batchSize_ << std::endl;
    log_ << "Number of inputs: " << numberInputVectors << std::endl;
    log_ << "Worker threads: " << workers_ << std::endl;
    log_ << "Epochs: " << epochs_ << std::endl;
    log_ << std::endl;

    steady_clock::time_point t1 = high_resolution_clock::now();

    logInterval_ = int((numberInputVectors / batchSize_) / 50.0);

    for (int i = 0; i < epochs_; i++)
    {
        log_ << "Epoch: " << i + 1 << " " << std::flush;

        runEpoch(pInput, numberInputVectors, pExpected);
    }

    steady_clock::time_point t2 = high_resolution_clock::now();

    duration<double, std::milli> duration = t2 - t1;

    std::cout << "\nCompleted in " << duration.count() / 1000 << " seconds" << std::endl;
}

// TODO change log_ to logger.log
