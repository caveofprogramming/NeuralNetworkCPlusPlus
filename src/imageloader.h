#pragma once

#include <string>
#include <fstream>
#include <cstdint>

#include "image.h"

namespace cop
{
    class ImageLoader
    {
    protected:
        static void reverseBytes(char *pStart, int size);
        static uint32_t readInt32(std::ifstream &file);
    public:
        static std::vector<cop::Image> loadImages(std::string filename);
        static std::vector<int> loadLabels(std::string filename);
    };

}