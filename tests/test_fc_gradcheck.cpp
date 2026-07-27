#include "sketchguesser/layers/fc_layer.hpp"
#include "sketchguesser/tensor.hpp"
#include <iostream>
#include <cmath>

float computeLoss(FCLayer& fc, const Tensor& input)
{
    Tensor output = fc.forward(input);
    float sum = 0.0f;
    for (size_t i = 0; i < output.getData().size(); i++)
        sum += output.getData()[i];
    return sum;
}

int main()
{
    std::cout << "=== FC Layer Gradient Check (dW, dB, dX) ===\n";

    FCLayer fc(2, 1);

    Tensor input(1, 1, 2);
    input.getData()[0] = 1.0f;
    input.getData()[1] = 2.0f;

    // Direct access to mutable weight and bias references
    Eigen::MatrixXf& weights = fc.getWeightsMutable();
    Eigen::VectorXf& bias = fc.getBiasMutable();

    Eigen::MatrixXf original_weights = weights;
    Eigen::VectorXf original_bias = bias;

    Tensor output = fc.forward(input);
    Tensor grad(1, 1, 1);
    grad.getData()[0] = 1.0f;
    Tensor dX_returned = fc.backward(grad); // Computes dW_ and dB_ internally

    Eigen::MatrixXf analytical_dW = fc.getWeightsGradient();
    Eigen::VectorXf analytical_dB = fc.getBiasGradient();

    float epsilon = 1e-4f;

    // --- Check dW ---c
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            weights(i, j) = original_weights(i, j) + epsilon;
            float loss_plus = computeLoss(fc, input);

            weights(i, j) = original_weights(i, j) - epsilon;
            float loss_minus = computeLoss(fc, input);

            weights(i, j) = original_weights(i, j); // restore

            float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
            std::cout << "W[" << i << "][" << j << "] analytical=" << analytical_dW(i, j)
                      << " numerical=" << numerical
                      << " diff=" << std::abs(analytical_dW(i, j) - numerical) << "\n";
        }
    }

    // --- Check dB ---
    for (int i = 0; i < 1; i++) {
        bias(i) = original_bias(i) + epsilon;
        float loss_plus = computeLoss(fc, input);

        bias(i) = original_bias(i) - epsilon;
        float loss_minus = computeLoss(fc, input);

        bias(i) = original_bias(i); // restore

        float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        std::cout << "B[" << i << "] analytical=" << analytical_dB(i)
                  << " numerical=" << numerical
                  << " diff=" << std::abs(analytical_dB(i) - numerical) << "\n";
    }

    // Restore original weights/bias fully
    weights = original_weights;
    bias = original_bias;

    // --- Check dX ---
    for (int j = 0; j < 2; j++) {
        Tensor input_plus = input;
        input_plus.getData()[j] = input.getData()[j] + epsilon;
        float loss_plus = computeLoss(fc, input_plus);

        Tensor input_minus = input;
        input_minus.getData()[j] = input.getData()[j] - epsilon;
        float loss_minus = computeLoss(fc, input_minus);

        float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
        float ana_dx = dX_returned.getData()[j];
        std::cout << "X[" << j << "] analytical=" << ana_dx
                  << " numerical=" << numerical
                  << " diff=" << std::abs(ana_dx - numerical) << "\n";
    }

    return 0;
}