#pragma once
#include "tensor.hpp"
#include <cstdint>

namespace DoodleGuesser
{
struct BoundingBox
{
    int minX, minY, maxX, maxY;
};

BoundingBox findBoundingBox(const uint8_t* buffer, int width, int height);

Tensor preprocess(const uint8_t* buffer, int width, int height);
}