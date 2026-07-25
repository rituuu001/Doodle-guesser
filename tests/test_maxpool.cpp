#include <iostream>
#include "sketchguesser/layers/maxpool_layer.hpp"
#include "sketchguesser/tensor.hpp"

int main()
{
    // 1 feature map, 4x4 input — easy to trace by hand
    Tensor input(1, 4, 4);

    input(0, 0, 0) = 1.0f; input(0, 0, 1) = 3.0f; input(0, 0, 2) = 2.0f; input(0, 0, 3) = 1.0f;
    input(0, 1, 0) = 4.0f; input(0, 1, 1) = 2.0f; input(0, 1, 2) = 0.0f; input(0, 1, 3) = 5.0f;
    input(0, 2, 0) = 1.0f; input(0, 2, 1) = 0.0f; input(0, 2, 2) = 3.0f; input(0, 2, 3) = 2.0f;
    input(0, 3, 0) = 2.0f; input(0, 3, 1) = 1.0f; input(0, 3, 2) = 4.0f; input(0, 3, 3) = 1.0f;

    std::cout << "Input:\n";
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            std::cout << input(0, h, w) << " ";
        }
        std::cout << "\n";
    }

    MaxPoolLayer pool;
    Tensor output = pool.forward(input);

    std::cout << "\nOutput after MaxPool:\n";
    for (int h = 0; h < output.getHeight(); h++) {
        for (int w = 0; w < output.getWidth(); w++) {
            std::cout << output(0, h, w) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nExpected:\n4 5\n2 4\n";

    return 0;
}