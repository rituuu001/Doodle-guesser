#pragma once
#include "layer.hpp"
#include "../tensor.hpp"

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer() = default;

    Tensor forward(const Tensor& input) override;
    //Tensor backward(const Tensor& gradient) override;
};