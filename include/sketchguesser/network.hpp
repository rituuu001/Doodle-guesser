#pragma once

#include"layers/layer.hpp"
#include"layers/conv_layer.hpp"
#include"layers/fc_layer.hpp"
#include"layers/maxpool_layer.hpp"
#include"layers/relu_layer.hpp"
#include"layers/softmax_layer.hpp"

#include"tensor.hpp"

#include<vector>
#include<iostream>
#include<memory>

class Network
{
    private:
    std::vector<std::unique_ptr<Layer>> layers;

    public:
    Network(){
        layers.push_back(std::make_unique<Convolution>(4,3));
        layers.push_back(std::make_unique<ReLULayer>());
        layers.push_back(std::make_unique<MaxPoolLayer>());
        layers.push_back(std::make_unique<FCLayer>(676,32));
        layers.push_back(std::make_unique<ReLULayer>());
        layers.push_back(std::make_unique<FCLayer>(32,6));
        layers.push_back(std::make_unique<SoftmaxLayer>());
    }

    void addLayer(std::unique_ptr<Layer> layer)
    {
        layers.push_back(std::move(layer));
    }
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradient);

    const std::vector<std::unique_ptr<Layer>>& getLayers()const { return layers; }
    std::vector<std::unique_ptr<Layer>>& getLayers(){ return layers; }


};