#pragma once

#include "layer.hpp"
#include "../tensor.hpp"
#include <Eigen/Dense>

class FCLayer : public Layer
{
private:
    Tensor input_cache;

    int input_size;
    int output_size;

    Eigen::MatrixXf weights_;   // (output_size x input_size)
    Eigen::VectorXf bias_;      // (output_size)

    Eigen::MatrixXf dW_;
    Eigen::VectorXf dB_;

public:
    FCLayer(int input_size, int output_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradient) override;

    void update(double learning_rate) override;

    // Serialization getters
    Eigen::MatrixXf& getWeights()
    {
        return weights_;
    }

    Eigen::VectorXf& getBias()
    {
        return bias_;
    }

    const Eigen::MatrixXf& getWeights() const
    {
        return weights_;
    }

    const Eigen::VectorXf& getBias() const
    {
        return bias_;
    }

    // Gradient getters
    const Eigen::MatrixXf& getDw() const
    {
        return dW_;
    }

    const Eigen::VectorXf& getDb() const
    {
        return dB_;
    }
};