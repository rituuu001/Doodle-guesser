#pragma once

#include "../tensor.hpp"

class Layer
{
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradient)=0;
    
    // updates weights and biases
    virtual void update(float learning_rate) {}

    virtual ~Layer() = default;
};
