#include "imageloader.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>

void cop::ImageLoader::reverseBytes(char *pStart, int size)
{
    char *pEnd = pStart + size;
    std::reverse(pStart, pEnd);
}

uint32_t cop::ImageLoader::readInt32(std::ifstream &inputFile)
{
    uint32_t value;

    inputFile.read(reinterpret_cast<char *>(&value), sizeof(value));
    reverseBytes(reinterpret_cast<char *>(&value), sizeof(value));

    return value;
}

void cop::ImageLoader::loadLabels(std::vector<Image> &images, std::string filename)
{
    std::ifstream inputFile;
    inputFile.open(filename, std::ios::binary);

    if (inputFile.is_open())
    {
        uint32_t magicNumber = readInt32(inputFile);

        if (magicNumber != 2049)
        {
            std::cerr << "Invalid file format" << std::endl;
            return;
        }

        uint32_t numberLabels = readInt32(inputFile);

        char label;

        for (int i = 0; i < numberLabels; i++)
        {
            inputFile.read(reinterpret_cast<char *>(&label), 1);
            images[i].setLabel((int)label);
        }
    }
}

std::vector<cop::Image> cop::ImageLoader::loadImages(std::string filename, int &width, int &height)
{
    std::vector<cop::Image> images;
    std::ifstream inputFile;

    inputFile.open(filename, std::ios::binary);

    if (inputFile.is_open())
    {
        uint32_t magicNumber = readInt32(inputFile);

        if (magicNumber != 2051)
        {
            std::cerr << "Invalid file format" << std::endl;
            return images;
        }

        uint32_t numberImages = readInt32(inputFile);
        height = readInt32(inputFile);
        width = readInt32(inputFile);

        int imageSize = width * height;

        for (int i = 0; i < numberImages; i++)
        {
            Image image(width, height);
            inputFile.read(image.get(), imageSize);
            images.push_back(image);
        }

        inputFile.close();
    }
    else
    {
        std::cerr << "Could not read file " + filename << std::endl;
    }

    return images;
}