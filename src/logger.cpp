#include "logger.h"

cop::Logger &cop::Logger::operator<<(std::string text)
{
    out_ << text;
    return *this;
}

cop::Logger &cop::Logger::operator<<(double value)
{
    out_ << value;
    return *this;
}

cop::Logger &cop::Logger::operator<<(int value)
{
    out_ << value;
    return *this;
}

cop::Logger &cop::Logger::operator<<(cop::Lock &lock)
{
    mtx_.lock();
    return *this;
}

cop::Logger &cop::Logger::operator<<(cop::Unlock &lock)
{
    mtx_.unlock();
    return *this;
}

cop::Logger &cop::Logger::operator<<(cop::Endl &lock)
{
    out_ << std::endl;
    return *this;
}