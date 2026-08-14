#include "sketchguesser/layers/softmax_layer.hpp"
#include <Eigen/Dense>

Tensor SoftmaxLayer::forward(const Tensor& tensor_input) {
    int size = tensor_input.size();
    Eigen::VectorXf eig_input(size);
    for (int i = 0; i < size; i++) {
        eig_input[i] = tensor_input[i];
    }

    float max = eig_input.maxCoeff();
    Eigen::VectorXf exponent = (eig_input.array() - max).exp();
    float total = exponent.sum();
    Eigen::VectorXf eig_output = exponent / total;

    // Save predictions in cache for backward pass
    output_cache_ = eig_output;

    Tensor tensor_output(1, 1, size);
    for (int i = 0; i < size; i++) {
        tensor_output[i] = eig_output[i];
    }
    return tensor_output;
}

Tensor SoftmaxLayer::backward(const Tensor& grad_output) {
    int size = grad_output.size();
    Eigen::VectorXf g(size);
    for (int i = 0; i < size; i++) {
        g[i] = grad_output[i];
    }

    // Jacobian computation: dx_i = p_i * (g_i - sum(g * p))
    float dot = g.dot(output_cache_);
    Eigen::VectorXf grad_input_eig = output_cache_.array() * (g.array() - dot);

    Tensor grad_input(1, 1, size);
    for (int i = 0; i < size; i++) {
        grad_input[i] = grad_input_eig[i];
    }
    return grad_input;
}