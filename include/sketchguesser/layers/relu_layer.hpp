#pragma once
#include"layer.hpp"
#include"../tensor.hpp"

class ReLULayer :public Layer {
private:
    Tensor input_cache;
public:
    Tensor forward(
        const Tensor& input) override;
    
    Tensor backward(
        const Tensor& grad) override;

};