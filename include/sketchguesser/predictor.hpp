#pragma once
#include <string>
#include <cstdint>
#include "network.hpp"
#include "tensor.hpp"

class Predictor
{
private:

    Network network;
    std::vector<std::string> classes =
    {
        "cup",
        "donut",
        "envelope",
        "guitar",
        "house",
        "pants"
    };

public:

    bool loadModel(const std::string& path);

    std::string predict(const uint8_t* buffer,int width,int height);
};