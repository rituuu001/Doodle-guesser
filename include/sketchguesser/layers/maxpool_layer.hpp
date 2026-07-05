#pragma once
#include <vector>
#include <Eigen/Dense>

struct MaxIndex
{
    int row,col;
};

class MaxPoolLayer 
{
private:
    std::vector<Eigen::MatrixXf> input_cache;
    std::vector<std::vector<std::vector<MaxIndex>>> max_indices;

public:
    std::vector<Eigen::MatrixXf> forward(
        const std::vector<Eigen::MatrixXf>& input);

    std::vector<Eigen::MatrixXf> backward(
        const std::vector<Eigen::MatrixXf>& grad);
};