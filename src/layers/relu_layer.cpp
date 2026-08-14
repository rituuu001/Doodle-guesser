#include "sketchguesser/layers/relu_layer.hpp"
#include<algorithm> //max

Tensor ReLULayer::forward(const Tensor& input)
{
    input_cache = input;
    Tensor output(input.getChannels(), input.getHeight(), input.getWidth());
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = std::max(0.0f, input[i]);
    }
    return output;
}

Tensor ReLULayer::backward(const Tensor& grad)
{
  Tensor output(input_cache.getChannels(), input_cache.getHeight(), input_cache.getWidth());
  for (size_t i = 0; i < grad.size(); i++)
    {
        // If the forward input was positive, pass the gradient through. Otherwise, zero it out.
        if (input_cache[i] > 0.0f)
        {
            output[i] = grad[i];
        }
        else
        {
            output[i] = 0.0f;
        }
    }

    return output;
}