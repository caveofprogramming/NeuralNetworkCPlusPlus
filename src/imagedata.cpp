#include "imagedata.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <iostream>
#include <cmath>
#include "bitmapfileheader.h"
#include "bitmapinfoheader.h"

cop::ImageData::~ImageData()
{
    delete[] imageData_;
    delete[] labelData_;
}

int cop::ImageData::getLabel(int index)
{
    double *pData = labelData_ + (index * 10);

    int result = 0;

    for (int i = 0; i < 10; i++)
    {
        result <<= 1;

        double value = *pData++;

        if (value > 0.1)
        {
            result = i;
            break;
        }
    }

    return result;
}

double *cop::ImageData::getImageData()
{
    return imageData_;
}

double *cop::ImageData::getLabelData()
{
    return nullptr;
}

int cop::ImageData::getPixelsPerImage()
{
    return pixelsPerImage_;
}

void cop::ImageData::save(int index)
{
    int label = getLabel(index);

    std::stringstream filename;

    filename << "image" << index << "_" << label << ".bmp";

    cop::BitmapFileHeader bmfh;
    cop::BitmapInfoHeader bmih;

    bmfh.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + (256 * 4);
    bmfh.fileSize = bmfh.dataOffset + (imageWidth_ * imageHeight_ * 3);

    bmih.width = imageWidth_;
    bmih.height = imageHeight_;
    bmih.bitsPerPixel = 8;
    bmih.colors = 256;

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

    // Palette
    for (int i = 0xFF; i >= 0; --i)
    {
        uint32_t color = ((i << 16) + (i << 8) + i);
        file.write((char *)&color, 4);
    }

    double *pBuffer = getImage(index);

    for (int row = imageHeight_ - 1; row >= 0; --row)
    {
        for (int col = 0; col < imageWidth_; col++)
        {
            uint8_t pixel = uint8_t(pBuffer[row * imageWidth_ + col] * 0xFF);
            file.write((char *)&pixel, 1);
        }
    }
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

double *cop::ImageData::getImage(int index)
{
    return imageData_ + pixelsPerImage_ * index;
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

    int totalPixels = pixelsPerImage_ * numberImages_;

    uint8_t *byteImageData = new uint8_t[totalPixels];
    imageData_ = new double[totalPixels];

    uint8_t *byteLabelData = new uint8_t[numberLabels];
    labelData_ = new double[numberImages_ * 10]{0};

    imageFile.read(reinterpret_cast<char *>(byteImageData), totalPixels);
    labelFile.read(reinterpret_cast<char *>(byteLabelData), numberLabels);

    imageFile.close();
    labelFile.close();


    for (int i = 0; i < totalPixels; ++i)
    {
        imageData_[i] = double(byteImageData[i]) / 255.0;
    }

    double *pLabel = labelData_;

    for (int i = 0; i < numberImages_; i++)
    {
        uint8_t label = byteLabelData[i];
        pLabel[label] = 1.0;
        pLabel  += 10;
    }

    delete[] byteImageData;
    delete[] byteLabelData;
}
