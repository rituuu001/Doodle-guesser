#pragma once

#include"layers/layer.hpp"
#include<memory>//unique_ptr
#include<vector>

class Optimizer//SGD for now
{
    private:
    float learningRate;

    public:
    Optimizer(float lr):learningRate(lr){}
    void step(std::vector<std::unique_ptr<Layer>>& layers)
    {
        for (auto& layer : layers)
            layer->update(learningRate);
    }


};