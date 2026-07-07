#include "sketchguesser/dataset.hpp"
#include <iostream>

int main()
{
    try
    {
        Dataset dataset("data/processed/dataset.bin");

        std::cout << "numImages: " << dataset.numImages() << std::endl;
        std::cout << "numClasses: " << dataset.numClasses() << std::endl;

        std::cout << "First label: " << static_cast<int>(dataset.getLabel(0)) << std::endl;
        std::cout << "Last label: " << static_cast<int>(dataset.getLabel(dataset.numImages() - 1)) << std::endl;

        const uint8_t* img = dataset.getImage(0);
        std::cout << "First image, first 10 pixel values: ";
        for (int i = 0; i < 10; i++)
            std::cout << static_cast<int>(img[i]) << " ";
        std::cout << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}