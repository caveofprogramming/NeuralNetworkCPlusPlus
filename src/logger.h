#pragma once

#include <mutex>
#include <iostream>

namespace cop
{
    struct Endl
    {

    };

    struct Lock
    {

    };

    struct Unlock
    {

    };

    class Logger
    {
    private:
        std::ostream &out_ = std::cout;
        std::mutex mtx_;

    public:
        cop::Logger &operator<<(std::string text);
        cop::Logger &operator<<(double value);
        cop::Logger &operator<<(int value);
        cop::Logger &operator<<(cop::Lock &lock);
        cop::Logger &operator<<(cop::Unlock &lock);
        cop::Logger &operator<<(cop::Endl &lock);
    };

    inline Lock lock;
    inline Unlock unlock;
    inline Endl endl;
    inline Logger logger;
}