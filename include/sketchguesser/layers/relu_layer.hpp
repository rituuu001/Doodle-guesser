#pragma once
#include <vector>
#include <Eigen/Dense>

class ReLULayer {
private:
    std::vector<Eigen::MatrixXf> input_cache;
public:
    std::vector<Eigen::MatrixXf> forward(
        const::std::vector<Eigen::MatrixXf>& input);
    
    std::vector<Eigen::MatrixXf> backward(
        const::std::vector<Eigen::MatrixXf>& grad);

};