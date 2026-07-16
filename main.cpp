#include <iostream>
#include "sketchguesser/layers/conv_layer.hpp"
#include "sketchguesser/layers/relu_layer.hpp"
#include "sketchguesser/layers/maxpool_layer.hpp"
#include "sketchguesser/tensor.hpp"

int main()
{
    std::cout << "Sketch-Guesser build OK\n";

    // Quick sanity check
    Convolution conv(4, 3);
    Tensor input(1, 28, 28);
    for (int y = 0; y < 28; y++)
        for (int x = 0; x < 28; x++)
            input(0, y, x) = static_cast<float>((y * 28 + x) % 7) * 0.1f;

    Tensor convOut = conv.forward(input);
    std::cout << "Conv output shape: " << convOut.getChannels() << "x"
               << convOut.getHeight() << "x" << convOut.getWidth() << "\n";

    return 0;
}
