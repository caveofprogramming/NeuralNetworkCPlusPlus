#include <string>
#include <fstream>
#include <vector>

namespace cop
{
    class ImageLoader
    {
    private:
        uint32_t numberImages_ = 0;
        uint32_t imageWidth_ = 0;
        uint32_t imageHeight_ = 0;
        uint32_t imageSize = 0;
        uint32_t pixelsPerImage_ = 0;

        std::vector<float> images_;
        std::vector<float> labels_;

    protected:
        static uint32_t readInt32(std::ifstream &file);

    public:
        void load(std::string imageFileName, std::string labelFileName);
        void save(int index);
        void load(int index);

        int getNumberImages();
        int getWidth();
        int getHeight();
        int getPixelsPerImage();
        float *getImageData();
        float *getLabelData();
        float *getImage(int index);
        int getLabel(int index);

        ~ImageLoader();
    };
}