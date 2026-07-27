#include <iostream>
#include <cmath>

#include "sketchguesser/loss.hpp"
#include "sketchguesser/layers/softmax_layer.hpp"
#include "sketchguesser/tensor.hpp"

int main()
{
    const float eps = 1e-5f;

    SoftmaxLayer softmax;
    CrossEntropyLoss loss;

    // Hardcoded logits (NOT probabilities)
    Tensor logits(1,1,6);

    logits(0)=1.2f;
    logits(1)=0.5f;
    logits(2)=3.1f;
    logits(3)=-0.8f;
    logits(4)=0.3f;
    logits(5)=1.0f;

    Tensor target(1,1,6);

    target(0)=0;
    target(1)=0;
    target(2)=1;
    target(3)=0;
    target(4)=0;
    target(5)=0;

    // Forward
    Tensor prediction = softmax.forward(logits);

    float L = loss.forward(prediction,target);

    std::cout << "Loss = " << L << "\n\n";

    // Analytical gradient
    Tensor analytical = loss.backward(prediction,target);

    std::cout << "Gradient Check\n\n";

    for(int i=0;i<logits.size();i++)
    {
        Tensor plus = logits;
        Tensor minus = logits;

        plus(i) += eps;
        minus(i) -= eps;

        Tensor pred_plus = softmax.forward(plus);
        Tensor pred_minus = softmax.forward(minus);

        float loss_plus = loss.forward(pred_plus,target);
        float loss_minus = loss.forward(pred_minus,target);

        float numerical =
            (loss_plus-loss_minus)/(2.0f*eps);

        float analytical_grad = analytical(i);

        float rel_error =
            std::fabs(analytical_grad-numerical)/
            std::max(
                1e-8f,
                std::fabs(analytical_grad)+std::fabs(numerical));

        std::cout << "Index " << i << "\n";
        std::cout << "Analytical : " << analytical_grad << "\n";
        std::cout << "Numerical  : " << numerical << "\n";
        std::cout << "Rel Error  : " << rel_error << "\n\n";
    }

    return 0;
}