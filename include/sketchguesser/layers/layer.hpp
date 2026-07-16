#pragma once

#include "../tensor.hpp"

class Layer
{
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradient) = 0;

   virtual ~Layer() = default;
};
