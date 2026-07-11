#pragma once
#include "layer.hpp"
#include "../tensor.hpp"
#include <Eigen/Dense>

class FCLayer : public Layer {
private:
    int input_size;
    int output_size;
    Eigen::MatrixXf weights_;  // (output_size x input_size)
    Eigen::VectorXf bias_;     // (output_size)

public:
    FCLayer(int input_size, int output_size);

    Tensor forward(const Tensor& input) override;
    //Tensor backward(const Tensor& gradient) override;
};