#pragma once
#include "layer.hpp"
#include "../tensor.hpp"
#include<Eigen/Dense>

class SoftmaxLayer : public Layer {
private:
    Eigen::VectorXf output_cache_;
public:
    SoftmaxLayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient) override;

    void update(double learning_rate) override {}
};