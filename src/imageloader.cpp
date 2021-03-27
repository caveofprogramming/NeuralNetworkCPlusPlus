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

std::vector<int> cop::ImageLoader::loadLabels(std::string filename)
{
    std::vector<int> labels;

    std::ifstream inputFile;
    inputFile.open(filename, std::ios::binary);

    if (inputFile.is_open())
    {
        uint32_t magicNumber = readInt32(inputFile);

        if (magicNumber != 2049)
        {
            std::cerr << "Invalid file format" << std::endl;
            return labels;
        }

        uint32_t numberLabels = readInt32(inputFile);

        char label;

        for (int i = 0; i < numberLabels; i++)
        {
            inputFile.read(reinterpret_cast<char *>(&label), 1);
            labels.push_back((int)label);
        }
    }

    return labels;
}

std::vector<cop::Image> cop::ImageLoader::loadImages(std::string filename)
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
        uint32_t numberRows = readInt32(inputFile);
        uint32_t numberCols = readInt32(inputFile);

        int imageSize = numberRows * numberCols;

        for (int i = 0; i < numberImages; i++)
        {
            Image image(numberCols, numberRows);
            inputFile.read(image.get(), imageSize);
            images.push_back(image);
        }

        inputFile.close();
    }
    else
    {
        std::cout << "Could not read file " + filename << std::endl;
    }

    return images;
}