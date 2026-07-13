#pragma once

#include "layer.hpp"
#include "../tensor.hpp"

#include <vector>

class Convolution : public Layer
{
private:
    int numFilters;
    int kernelSize;

    // Stores each 3×3 filter as a Tensor
    std::vector<Tensor> filters;

public:
    // Constructor
    Convolution(int numFilters = 4, int kernelSize = 3);

    // Forward propagation
    Tensor forward(const Tensor& input) override;

    // Backward propagation (implemented later)
    Tensor backward(const Tensor& gradient) override;
};