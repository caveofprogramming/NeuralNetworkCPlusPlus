#include <string>
#include <fstream>

namespace cop
{
    class ImageData
    {
    private:
        uint32_t numberImages_ = 0;
        uint32_t imageWidth_ = 0;
        uint32_t imageHeight_ = 0;
        uint32_t pixelsPerImage_ = 0;
        uint8_t *pixels_ = nullptr;
        uint8_t *labels_ = nullptr;

        double *imageData_ = nullptr;

    protected:
        static uint32_t readInt32(std::ifstream &file);

    public:
        void load(std::string imageFileName, std::string labelFileName);
        void save(int index);
        void load(int index);

        int getNumberImages();
        int getWidth();
        int getHeight();
        int getLabel(int index);
        double *getBuffer();


        ~ImageData();
    };
}