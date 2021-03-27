#pragma once

#include <vector>
#include <string>

namespace cop
{
    class Image
    {
    private:
        int width;
        int height;
        std::vector<char> pixels;

    public:
        Image(int width, int height);

        char * get();

        bool save(std::string filename);
    };
}