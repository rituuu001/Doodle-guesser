#pragma once

#include"layers/layer.hpp"
#include<memory>
#include<vector>

class Optimizer
{
    private:
    double learningRate;

    public:
    Optimizer(double lr):learningRate(lr){}
    void step(std::vector<std::unique_ptr<Layer>>& layers)
    {
        for (auto& layer : layers)
            layer->update(learningRate);
    }


};