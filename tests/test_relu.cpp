#include <iostream>
#include "sketchguesser/layers/relu_layer.hpp"
#include "sketchguesser/tensor.hpp"

int main()
{
    // 1 feature map, 3x3 input
    Tensor input(1, 3, 3);

    input(0,0,0) = -2.0f;
    input(0,0,1) =  1.5f;
    input(0,0,2) =  0.0f;

    input(0,1,0) =  4.0f;
    input(0,1,1) = -1.0f;
    input(0,1,2) = -3.5f;

    input(0,2,0) =  2.2f;
    input(0,2,1) =  0.0f;
    input(0,2,2) = -7.0f;

    std::cout << "Input:\n";
    for (int h = 0; h < input.getHeight(); h++)
    {
        for (int w = 0; w < input.getWidth(); w++)
        {
            std::cout << input(0, h, w) << " ";
        }
        std::cout << "\n";
    }

    ReLULayer relu;
    Tensor output = relu.forward(input);

    std::cout << "\nOutput after ReLU:\n";
    for (int h = 0; h < output.getHeight(); h++)
    {
        for (int w = 0; w < output.getWidth(); w++)
        {
            std::cout << output(0, h, w) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nExpected:\n";
    std::cout << "0 1.5 0\n";
    std::cout << "4 0 0\n";
    std::cout << "2.2 0 0\n";

    return 0;
}