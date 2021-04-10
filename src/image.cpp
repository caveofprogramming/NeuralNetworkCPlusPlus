#include <fstream>

#include "image.h"
#include "bitmapfileheader.h"
#include "bitmapinfoheader.h"

cop::Image::Image(int width, int height) : width(width), height(height)
{
    pixels.resize(width * height);
}

char *cop::Image::get()
{
    return pixels.data();
}

double cop::Image::operator[](int index)
{
    return pixels[index];
}

bool cop::Image::save(std::string filename)
{
    cop::BitmapFileHeader bmfh;
    cop::BitmapInfoHeader bmih;

    bmfh.dataOffset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader);
    bmfh.fileSize = sizeof(bmfh) + sizeof(bmih) + (width * height * 3);

    bmih.width = width;
    bmih.height = height;

    std::ofstream file;
    file.open(filename, ios::out | ios::binary);

    if (!file)
    {
        return false;
    }

    file.write((char *)&bmfh, sizeof(bmfh));
    file.write((char *)&bmih, sizeof(bmih));

    for (int y = height - 1; y >= 0; y--)
    {

        for (int x = 0; x < width; x++)
        {
            char value = *(pixels.data() + (y * width) + x) ^ 0xFF;
            file.write(&value, 1);
            file.write(&value, 1);
            file.write(&value, 1);
        }
    }

    file.close();

    if (!file)
    {
        return false;
    }

    return true;
}