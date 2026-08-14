#include "sketchguesser/dataset.hpp"
#include<fstream>
#include<stdexcept> //std::runtime_error, std::out_of_range

Dataset::Dataset(const std::string& path) 
{
    load(path);
}

void Dataset::load(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("The file couldn't be opened " + path);
    }
    int32_t num_classes,num_images; //int32_t is used to ensure that it is 32 bit signed int

    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));

    if (file.fail())
    {
        throw std::runtime_error("File failed to open");
    }

     numImages_=num_images;
     numClasses_=num_classes;

     //Allocates memory
     images_.resize(numImages_*imageSize );
     labels_.resize(numImages_);

     for(size_t i=0;i<numImages_;i++)
     {
        file.read(reinterpret_cast<char*>(images_.data()+i*imageSize),imageSize);//images_ is uint8_t
        file.read(reinterpret_cast<char*>(labels_.data()+i),1);
     }

    if (file.fail() || file.bad()) 
    {
    throw std::runtime_error("Error reading data from dataset.bin (File may be corrupt or truncated).");
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
      // check to prevent acessing illegal memory
      if (index >= numImages_)
      {
        throw std::out_of_range("Image index out of bounds!");
      }
    
      // Return pointer to the first pixel of the requested image
       return images_.data()+(index*imageSize);
    }

    uint8_t Dataset::getLabel(size_t index) const
    {   
        if (index >= numImages_) 
        {
         throw std::out_of_range("Label index out of bounds!");
        }
        //returns label to the requested image
        return (*(labels_.data()+index));
    }

