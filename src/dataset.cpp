#include "sketchguesser/dataset.hpp"
#include <fstream>
#include <stdexcept>
#include<cstdint>

Dataset::Dataset(const std::string& path) {
    load(path);
}

void Dataset::load(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("The file couldn't be opened " + path);
    }
    int32_t num_classes,num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));
    if (file.fail())
    {
        throw std::runtime_error("File failed to load");
    }
     numImages_=num_images;
     numClasses_=num_classes;
     images_.resize(numImages_ * imageSize );
     labels_.resize(numImages_);

     for(size_t i=0;i<numImages_;i++)
     {
        file.read(reinterpret_cast<char*>(images_.data()+i * imageSize), imageSize);
        file.read(reinterpret_cast<char*>(labels_.data()+i ), 1);
     }
     if (file.fail())
    {
        throw std::runtime_error("File failed to load");
    }
    }
    size_t Dataset::numImages() const
    {
        return numImages_;
    }
    int Dataset::numClasses() const
    {
        return numClasses_;
    }

    const uint8_t* Dataset:: getImage(size_t index) const
    {
        return (images_.data()+index * imageSize);
    }
    uint8_t Dataset::getLabel(size_t index) const
    {
        return (*(labels_.data()+index));
    }
