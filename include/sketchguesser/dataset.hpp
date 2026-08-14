#pragma once

#include <cstdint>// uint8_t and uint32_t
#include <string>
#include <vector>

class Dataset {
public:
   //constexpr makes sure the value is known at compile time
    static constexpr int imageWidth  = 28;
    static constexpr int imageHeight = 28;
    static constexpr int imageSize   = imageWidth * imageHeight; // 784
    
    //prevents compiler from converting string path to full dataset object
    explicit Dataset(const std::string& path);
    
    //getter methods
    size_t numImages() const;
    int numClasses() const;

    const uint8_t* getImage(size_t index) const;
    uint8_t getLabel(size_t index) const;

private:
    //Reads and parses the custom packed binary format
    void load(const std::string& path);

    std::vector<uint8_t> images_;
    std::vector<uint8_t> labels_;

    size_t numImages_  = 0;
    int    numClasses_ = 0;
};

