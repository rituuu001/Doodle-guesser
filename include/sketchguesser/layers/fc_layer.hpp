#pragma once
#include "layer.hpp"
#include "../tensor.hpp"
#include <Eigen/Dense>

class FCLayer : public Layer {
private:
    Tensor input_cache;
    float learning_rate = 0.01f;
    int input_size;
    int output_size;
    Eigen::MatrixXf weights_;  // (output_size x input_size)
    Eigen::VectorXf bias_;     // (output_size)

    Eigen::MatrixXf last_dW;
    Eigen::VectorXf last_dB;

public:
    FCLayer(int input_size, int output_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient) override;

    Eigen::MatrixXf getLastDW() const { return last_dW; }
    Eigen::VectorXf getLastDB() const { return last_dB; }
    Eigen::MatrixXf getWeights() const { return weights_; }
    Eigen::VectorXf getBias() const { return bias_; }

    void setWeight(int i, int j, float value) { weights_(i, j) = value; }
    void setBias(int i, float value) { bias_(i) = value; }
};