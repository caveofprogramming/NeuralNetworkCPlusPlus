#include "imageloader.h"

#include <algorithm>
#include <exception>
#include <sstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include "bitmapfileheader.h"
#include "bitmapinfoheader.h"

cop::ImageLoader::~ImageLoader()
{
}

/*
 * This translates the one-hot vector label back into a number from 0-9
 */
int cop::ImageLoader::getLabel(int index)
{
    float *pData = labels_.data() + (index * 10);

    int result = 0;

    for (int i = 0; i < 10; i++)
    {
        result <<= 1;

        float value = *pData++;

        if (value > 0.1)
        {
            result = i;
            break;
        }
    }

    return result;
}

float *cop::ImageLoader::getImageData()
{
    return images_.data();
}

float *cop::ImageLoader::getLabelData()
{
    return labels_.data();
}

int cop::ImageLoader::getPixelsPerImage()
{
    return pixelsPerImage_;
}

void cop::ImageLoader::save(int index)
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

    int imageIndex = index * imageWidth_ * imageHeight_;

    for (int row = imageHeight_ - 1; row >= 0; --row)
    {
        for (uint32_t col = 0; col < imageWidth_; col++)
        {
            uint8_t pixel = uint8_t(images_[imageIndex + (row * imageWidth_) + col] * 0xFF);
            file.write((char *)&pixel, 1);
        }
    }
}

uint32_t cop::ImageLoader::readInt32(std::ifstream &inputFile)
{
    uint32_t value;

    char *pStart = reinterpret_cast<char *>(&value);

    inputFile.read(pStart, sizeof(value));

    char *pEnd = pStart + sizeof(value);

    std::reverse(pStart, pEnd);

    return value;
}

int cop::ImageLoader::getNumberImages()
{
    return numberImages_;
}

int cop::ImageLoader::getWidth()
{
    return imageWidth_;
}

int cop::ImageLoader::getHeight()
{
    return imageHeight_;
}

float *cop::ImageLoader::getImage(int index)
{
    return images_.data() + pixelsPerImage_ * index;
}

void cop::ImageLoader::load(std::string imageFileName, std::string labelFileName)
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

    // We've checked the files contain the right magic numbers.
    // Read the data.

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

    // Each label will be in one-hot vector form.
    // Eg. "3" is 0001000000
    std::vector<uint8_t> byteImageData(totalPixels);
    std::vector<uint8_t> byteLabelData(numberLabels * 10);

    imageFile.read(reinterpret_cast<char *>(byteImageData.data()), totalPixels);
    labelFile.read(reinterpret_cast<char *>(byteLabelData.data()), numberLabels);

    imageFile.close();
    labelFile.close();

    // Got all the data. Insert image and label data into the vectors.

    images_.resize(totalPixels);
    labels_.resize(numberLabels * 10);

    for (int i = 0; i < totalPixels; ++i)
    {
        images_[i] = float(byteImageData[i]) / 255.0;
    }

    for (uint32_t i = 0; i < numberLabels; i++)
    {
        int label = byteLabelData[i];
        labels_[(i * 10) + label] = 1;
    }
}
