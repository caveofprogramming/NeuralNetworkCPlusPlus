#include <iostream>
#include <string>
#include <mutex>
#include "matrix.h"

namespace cop
{
    class Logger
    {
    private:
        std::ostream &out = std::cout;
        std::mutex mtx_;

    
    public:
        void log(std::string text)
        {
            std::unique_lock<std::mutex> lock(mtx_);
            out << text << std::endl;
        }

        void log(std::string text, double value)
        {
            std::unique_lock<std::mutex> lock(mtx_);
            out << text << " " << value << std::endl;
        }

        void log(std::string label, std::vector<cop::Matrix> &weights, std::vector<cop::Matrix> &biases);
        void log(std::string label, int layer, cop::Matrix &weights, cop::Matrix &biases);
    };

    static Logger logger;
}