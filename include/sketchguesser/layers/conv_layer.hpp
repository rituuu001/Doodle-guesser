#pragma once

#include "layer.hpp"
#include "../tensor.hpp"

#include <vector>

class Convolution : public Layer
{
private:
    int numFilters;
    int kernelSize;
    
    Tensor cachedInput;

    // Stores each 3x3 filter as a Tensor
    std::vector<Tensor> filters;
    //biases as vector
    std::vector<float> biases;

  std::vector<float> dBiases;
  std::vector<Tensor> dFilters;

public:
    // Constructor
    Convolution(int numFilters = 4, int kernelSize = 3);

    // Forward propagation
    Tensor forward(const Tensor& input) override;

    // Backward propagation 
    Tensor backward(const Tensor& gradient) override;

    void update(float learning_rate) override;

    //getter functions
    std::vector<Tensor>& getFilters() { return filters; }
    std::vector<float>& getBiases() { return biases; }
    std::vector<Tensor>& getDFilters() { return dFilters; }
    std::vector<float>& getDBiases() { return dBiases; }
};