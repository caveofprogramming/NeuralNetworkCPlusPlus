#include <random>
#include <time.h>
#include <thread>
#include <chrono>
#include <cmath>
#include <fstream>
#include "threadpool.h"
#include "activations.h"
#include "neuralnetwork.h"
#include "logger.h"

using namespace std::chrono;

void cop::NeuralNetwork::save(std::string filename)
{
    std::ofstream outputFile;

    outputFile.open(filename, std::ios::binary);

    if (outputFile.is_open())
    {
        outputFile.write(reinterpret_cast<char *>(&batchSize_), sizeof(batchSize_));
        outputFile.write(reinterpret_cast<char *>(&epochs_), sizeof(epochs_));
        outputFile.write(reinterpret_cast<char *>(&logInterval_), sizeof(logInterval_));
        outputFile.write(reinterpret_cast<char *>(&workers_), sizeof(workers_));

        int layers = w_.size();
        outputFile.write(reinterpret_cast<char *>(&layers), sizeof(layers));

        for (int layer = 0; layer < w_.size(); layer++)
        {
            auto &weights = w_[layer];
            auto &biases = b_[layer];

            weights.serialize(outputFile);
            biases.serialize(outputFile);
        }

        outputFile.close();
    }
    else
    {
        std::cout << "Could not create file " + filename;
    }
}

void cop::NeuralNetwork::load(std::string filename)
{
    std::ifstream inputFile;

    inputFile.open(filename, std::ios::binary);

    if (inputFile.is_open())
    {
        inputFile.read(reinterpret_cast<char *>(&batchSize_), sizeof(batchSize_));
        inputFile.read(reinterpret_cast<char *>(&epochs_), sizeof(epochs_));
        inputFile.read(reinterpret_cast<char *>(&logInterval_), sizeof(logInterval_));
        inputFile.read(reinterpret_cast<char *>(&workers_), sizeof(workers_));

        int layers = 0;
        inputFile.read(reinterpret_cast<char *>(&layers), sizeof(layers));

        w_.clear();
        b_.clear();

        for (int layer = 0; layer < layers; layer++)
        {
            Matrix weights;
            Matrix biases;

            weights.deserialize(inputFile);
            biases.deserialize(inputFile);

            w_.push_back(std::move(weights));
            b_.push_back(std::move(biases));
        }

        inputFile.close();
    }
    else
    {
        std::cout << "Could not create file " + filename;
    }
}

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
    cop::logger << "Network:\n";

    for (int i = 0; i < w_.size(); i++)
    {
        cop::logger << w_[i].augment(b_[i]).toString() << "\n";
    }
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

void cop::NeuralNetwork::computeDeltas(std::vector<cop::Matrix> &deltas, std::vector<cop::Matrix> &layerIo, const cop::Matrix &expected)
{
    auto loss = computeLoss(layerIo.back(), expected);

    auto errors = layerIo.back() - expected;

    deltas.push_back(std::move(errors));
    std::cout << "\ndeltas\n"
              << deltas.back() << std::endl;
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
        computeDeltas(deltas, layerIo, expected);

        auto loss1 = computeLoss(layerIo.back(), expected);

        /******* TEST *****************************/

        double inc = 0.0000001;
        b_[1][1][0] += inc;
        computeOutputs(layerIo);

        auto loss2 = computeLoss(layerIo.back(), expected);

        std::cout << "Rate: " << (loss2-loss1)/inc << std::endl;


        /******************************************/

        pInputVector += inputRows;
        pExpectedVector += outputRows;
    }

    if (logInterval_ != 0 && sequence % logInterval_ == 0)
    {
        cop::logger << ".";
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

    cop::logger << duration.count() << " ms";
}

void cop::NeuralNetwork::fit(double *pInput, int numberInputVectors, double *pExpected)
{
    cop::logger << "Batch size:" << batchSize_ << "\n";
    cop::logger << "Number of inputs:" << numberInputVectors << "\n";
    cop::logger << "Worker threads: " << workers_ << "\n";
    cop::logger << "Epochs: " << epochs_ << "\n";

    steady_clock::time_point t1 = high_resolution_clock::now();

    logInterval_ = int((numberInputVectors / batchSize_) / 50.0);

    for (int i = 0; i < epochs_; i++)
    {
        cop::logger << "Epoch: " << i + 1 << "\n";

        runEpoch(pInput, numberInputVectors, pExpected);
    }

    steady_clock::time_point t2 = high_resolution_clock::now();

    duration<double, std::milli> duration = t2 - t1;

    cop::logger << "Completed in " << duration.count() / 1000 << " seconds";
}
