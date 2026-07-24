#include "include\sketchguesser/dataset.hpp"
#include <fstream>
#include <stdexcept>
#include<cstdint>
#include<iostream>
#include<numeric>
#include<random>
#include<algorithm>

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

DatasetSplit splitDataset(size_t totalImages, float trainRatio)
{
    std::vector<size_t> indices(totalImages);
    std::iota(indices.begin(), indices.end(),0);

    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t trainSize = static_cast<size_t>(totalImages * trainRatio);

    DatasetSplit split;
    split.trainIndices.assign(indices.begin(), indices.begin() + trainSize);
    split.valIndices.assign(indices.begin() + trainSize, indices.end());
    
    std::cout<<"[Split Complete] Train :"<<split.trainIndices.size()<<" |Val: "<<split.valIndices.size()<<std::endl;
    return split;
}