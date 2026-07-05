#include "sketchguesser/layers/relu_layer.hpp"

std::vector<Eigen::MatrixXf> ReLULayer::forward(const std::vector<Eigen::MatrixXf>& input)
{
    input_cache = input;
    std::vector<Eigen::MatrixXf> output;
    for(const auto& feature_map : input)
    {
        output.push_back(feature_map.cwiseMax(0.0f));
    }

    return output;
}

std::vector<Eigen::MatrixXf> ReLULayer::backward(
    const std::vector<Eigen::MatrixXf>& grad)
    {
    //forweek3
    return grad;
    }