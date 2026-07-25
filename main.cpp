
#include "sketchguesser/network.hpp"
#include <iostream>

#include "sketchguesser/optimizer.hpp"
#include "sketchguesser/layers/fc_layer.hpp"


int main()
{
    Network net;
    Optimizer opt(0.01);

    Tensor input(1, 28, 28);
    for (int h = 0; h < 28; h++)
        for (int w = 0; w < 28; w++)
            input(0, h, w) = 0.5f;

    Tensor output = net.forward(input);

    Tensor fakeGradient(1, 1, 6);
    for (int i = 0; i < 6; i++)
        fakeGradient(i) = 0.1f;

    net.backward(fakeGradient);

    FCLayer* fc2 = dynamic_cast<FCLayer*>(net.getLayers()[5].get());

    if (!fc2)
    {
        std::cout << "Cast failed - not an FCLayer at index 5" << std::endl;
        return 1;
    }

    std::cout << "dW_ norm before step: " << fc2->getDw().norm() << std::endl;

    float beforeNorm = fc2->getWeights().norm();
    std::cout << "Weight matrix norm before step: " << beforeNorm << std::endl;

    opt.step(net.getLayers());

    float afterNorm = fc2->getWeights().norm();
    std::cout << "Weight matrix norm after step: " << afterNorm << std::endl;

    if (beforeNorm != afterNorm)
        std::cout << "PASS: weights changed" << std::endl;
    else
        std::cout << "FAIL: weights did not change" << std::endl;

    return 0;
}
  