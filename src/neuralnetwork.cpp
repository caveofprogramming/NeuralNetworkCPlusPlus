#include <random>
#include <time.h>
#include <thread>
#include "threadpool.h"
#include "activations.h"
#include "neuralnetwork.h"

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

void cop::NeuralNetwork::computeOutputs(std::vector<cop::Matrix> &layerIo, double *pInput)
{
    for (int layer = 0; layer < w_.size(); layer++)
    {
        auto &weights = w_[layer];
        auto &biases = b_[layer];
        auto &input = layerIo[layer];
        auto &output = layerIo[layer + 1];

        weights.multiply(output, input);
        
        output.addTo(biases);
        cop::softmax(output.data(), output.rows());
        
    }
}

int cop::NeuralNetwork::runBatch(double *pInput, int numberInputVectors, double *pExpected)
{
    int inputRows = w_[0].cols();

    double *pInputVector = pInput;

    std::vector<cop::Matrix> layerIo;

    for (int i = 0; i < w_.size(); i++)
    {
        layerIo.push_back(cop::Matrix(w_[i].cols(), 1));
    }

    layerIo.push_back(cop::Matrix(w_.back().rows(), 1));

    for (int i = 0; i < numberInputVectors; i++)
    {
        layerIo[0].setData(pInput, inputRows * sizeof(double));

        computeOutputs(layerIo, pInputVector);

        pInputVector += inputRows;
    }

    return 0;
}

void cop::NeuralNetwork::runEpoch(double *pInput, int numberInputVectors, double *pExpected)
{
    time_t startTime = time(nullptr);

    const int inputVectorSize = w_[0].cols();

    const int numberBatches = numberInputVectors / batchSize_;
    const int lastBatchSize = numberInputVectors % batchSize_;

    double *pBatchInput = pInput;
    double *pBatchExpected = pExpected;
    const int outputSize = w_.back().rows();
    int batchSize = batchSize_;

    cop::ThreadPool<int> threadPool(std::thread::hardware_concurrency());

    for (int i = 0; i < numberBatches; ++i)
    {
        if (i == numberBatches - 1 && lastBatchSize != 0)
        {
            batchSize = lastBatchSize;
        }

        auto work = [&]() {
            return runBatch(pBatchInput, batchSize, pBatchExpected);
        };

        threadPool.submit(work);

        pBatchInput += (inputVectorSize * batchSize_);
        pBatchExpected += outputSize * batchSize_;

        if (i % logInterval_ == 0)
        {
            log_ << "." << std::flush;
        }
    }

    threadPool.start();

    for (int i = 0; i < numberBatches; ++i)
    {
        auto result = threadPool.get();
    }
    
    threadPool.awaitComplete();

    time_t endTime = time(nullptr);
    int duration = endTime - startTime;

    log_ << std::endl
         << duration << " seconds" << std::endl;
}

void cop::NeuralNetwork::fit(double *pInput, int numberInputVectors, double *pExpected)
{
    log_ << "Batch size: " << batchSize_ << std::endl;
    log_ << "Number of inputs: " << numberInputVectors << std::endl;
    log_ << std::endl;

    logInterval_ = int(numberInputVectors / (80.0 * batchSize_)) + 1;

    for (int i = 0; i < epochs_; i++)
    {
        log_ << "Epoch: " << i + 1 << " " << std::flush;

        runEpoch(pInput, numberInputVectors, pExpected);

        log_ << std::endl;
    }
}
