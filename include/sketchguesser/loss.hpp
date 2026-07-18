#pragma once

#include "tensor.hpp"

class CrossEntropyLoss
{
public:
    float forward(const Tensor& prediction,
                  const Tensor& target);

    Tensor backward(const Tensor& prediction,
                    const Tensor& target);
};