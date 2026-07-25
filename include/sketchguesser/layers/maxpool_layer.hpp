#pragma once
#include <vector>
#include"layer.hpp"
#include"../tensor.hpp"
#include<algorithm>

struct MaxIndex
{
    int row,col;
};

class MaxPoolLayer : public Layer
{
private:
    Tensor input_cache;
    std::vector<std::vector<std::vector<MaxIndex>>> max_indices;

public:
    Tensor forward(
        const Tensor& input) override;

    Tensor backward(
        const Tensor& grad) override;
};