#include "imagedata.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <iostream>
#include "bitmapfileheader.h"
#include "bitmapinfoheader.h"

cop::ImageData::~ImageData()
{
    delete[] pixels_;
    delete[] imageData_;
}

void cop::ImageData::save(int index)
{
    int label = getLabel(index);

    load(index);

    auto pData = getBuffer();

    std::stringstream filename;

    filename << "image" << index << "_" << label << ".bmp";

    cop::BitmapFileHeader bmfh;
    cop::BitmapInfoHeader bmih;

    bmfh.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);
    bmfh.fileSize = sizeof(bmfh) + sizeof(bmih) + (imageWidth_ * imageHeight_ * 3);

    bmih.width = imageWidth_;
    bmih.height = imageHeight_;

    std::ofstream file;
    file.open(filename.str(), ios::out | ios::binary);

    if (!file)
    {
        stringstream message;
        message << "Unable to write file " << filename.str();
        throw runtime_error(message.str());
    }

    file.write((char *)&bmfh, sizeof(bmfh));
    file.write((char *)&bmih, sizeof(bmih));

    for(int i = 0; i < pixelsPerImage_; i++)
    {
        int value = int(0x100 * pData[i]);
        file.write((char *)&value, 1);
        file.write((char *)&value, 1);
        file.write((char *)&value, 1);
    }

    std::cout << "Written " << filename.str() << std::endl;
}

uint32_t cop::ImageData::readInt32(std::ifstream &inputFile)
{
    uint32_t value;

    char *pStart = reinterpret_cast<char *>(&value);

    inputFile.read(pStart, sizeof(value));

    char *pEnd = pStart + sizeof(value);

    std::reverse(pStart, pEnd);

    return value;
}

int cop::ImageData::getNumberImages()
{
    return numberImages_;
}

int cop::ImageData::getWidth()
{
    return imageWidth_;
}

int cop::ImageData::getHeight()
{
    return imageHeight_;
}

int cop::ImageData::getLabel(int index)
{
    return int(labels_[index]);
}

void cop::ImageData::load(int index)
{
    uint8_t *pData = pixels_ + (index * pixelsPerImage_);

    for (int i = 0; i < pixelsPerImage_; i++)
    {
        imageData_[i] = double(*pData++ / 255.0);
    }
}

double *cop::ImageData::getBuffer()
{
    return imageData_;
}

void cop::ImageData::load(std::string imageFileName, std::string labelFileName)
{
    std::ifstream imageFile;
    std::ifstream labelFile;
    std::stringstream message;

    imageFile.open(imageFileName, std::ios::binary);
    labelFile.open(labelFileName, std::ios::binary);

    if (!imageFile.is_open())
    {
        message << "Unable to open image file " << imageFileName;
        throw std::runtime_error(message.str());
    }

    if (!labelFile.is_open())
    {
        message << "Unable to open label file " << labelFileName;
        throw std::runtime_error(message.str());
    }

    uint32_t magicNumber1 = readInt32(imageFile);

    if (magicNumber1 != 2051)
    {
        message << "Invalid image file format " << imageFileName;
        throw std::runtime_error(message.str());
    }

    uint32_t magicNumber2 = readInt32(labelFile);

    if (magicNumber2 != 2049)
    {
        message << "Invalid label file format " << labelFileName;
        throw std::runtime_error(message.str());
    }

    numberImages_ = readInt32(imageFile);
    uint32_t numberLabels = readInt32(labelFile);

    if (numberImages_ != numberLabels)
    {
        message << "Number of images and number of labels found do not match.";
        throw std::runtime_error(message.str());
    }

    imageHeight_ = readInt32(imageFile);
    imageWidth_ = readInt32(imageFile);
    pixelsPerImage_ = imageWidth_ * imageHeight_;

    int imageDataSize = pixelsPerImage_ * numberImages_;

    pixels_ = new uint8_t[imageDataSize];
    labels_ = new uint8_t[numberLabels];

    imageFile.read(reinterpret_cast<char *>(pixels_), imageDataSize);
    labelFile.read(reinterpret_cast<char *>(labels_), numberLabels);

    imageFile.close();
    labelFile.close();

    imageData_ = new double[imageDataSize];
}
