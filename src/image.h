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
        int label;
        std::vector<char> pixels;

    public:
        Image(int width, int height);

        char * get();
        int size() { return width * height; };
        void setLabel(int label) { this->label = label; };
        char getLabel() { return label; };

        bool save(std::string filename);
        float operator[](int index);

        
    };
}