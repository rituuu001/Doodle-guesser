#include "sketchguesser/layers/relu_layer.hpp"
#include<algorithm>

Tensor ReLULayer::forward(const Tensor& input)
{
    input_cache = input;
    Tensor output(input.getChannels(), input.getHeight(), input.getWidth());
    for(int c=0; c < input.getChannels(); c++)
    {
        for(int h=0; h < input.getHeight(); h++)
        {
            for(int w=0; w < input.getWidth(); w++)
            {
                float value = input(c, h, w);
                output(c, h, w) = std::max(0.0f, value);
            }
        }
    }
    return output;
}

Tensor ReLULayer::backward(
    const Tensor& grad)
    {
    //forweek3
    return grad;
    }