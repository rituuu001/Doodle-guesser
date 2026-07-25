#include "sketchguesser/layers/fc_layer.hpp"
#include "sketchguesser/tensor.hpp"
#include <iostream>
#include <cmath>

float computeLoss(FCLayer& fc, const Tensor& input)
{
    Tensor output = fc.forward(input);
    float sum = 0.0f;
    for (int i = 0; i < output.size(); i++)
        sum += output(i);
    return sum;
}

int main()
{
    FCLayer fc(2, 1);

    Tensor input(1, 1, 2);
    input(0) = 1.0f;
    input(1) = 2.0f;

    // Save ORIGINAL weights/bias before backward() modifies them
    Eigen::MatrixXf original_weights = fc.getWeights();
    Eigen::VectorXf original_bias = fc.getBias();

    Tensor output = fc.forward(input);
    Tensor grad(1, 1, 1);
    grad(0) = 1.0f;
    Tensor dX_returned = fc.backward(grad);   // this updates weights_/bias_ internally

    Eigen::MatrixXf analytical_dW = fc.getLastDW();
    Eigen::VectorXf analytical_dB = fc.getLastDB();

    float epsilon = 1e-4f;

    // --- Check dW ---
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            fc.setWeight(i, j, original_weights(i, j) + epsilon);
            float loss_plus = computeLoss(fc, input);

            fc.setWeight(i, j, original_weights(i, j) - epsilon);
            float loss_minus = computeLoss(fc, input);

            fc.setWeight(i, j, original_weights(i, j));  // restore

            float numerical = (loss_plus - loss_minus) / (2 * epsilon);
            std::cout << "W[" << i << "][" << j << "] analytical=" << analytical_dW(i,j)
                      << " numerical=" << numerical
                      << " diff=" << std::abs(analytical_dW(i,j) - numerical) << "\n";
        }
    }

    // --- Check dB ---
    for (int i = 0; i < 1; i++) {
        fc.setBias(i, original_bias(i) + epsilon);
        float loss_plus = computeLoss(fc, input);

        fc.setBias(i, original_bias(i) - epsilon);
        float loss_minus = computeLoss(fc, input);

        fc.setBias(i, original_bias(i));  // restore

        float numerical = (loss_plus - loss_minus) / (2 * epsilon);
        std::cout << "B[" << i << "] analytical=" << analytical_dB(i)
                  << " numerical=" << numerical
                  << " diff=" << std::abs(analytical_dB(i) - numerical) << "\n";
    }

    // --- Restore weights/bias fully to ORIGINAL before checking dX ---
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < 2; j++)
            fc.setWeight(i, j, original_weights(i, j));
    for (int i = 0; i < 1; i++)
        fc.setBias(i, original_bias(i));

    // --- Check dX (now using the SAME weights dX_returned was computed with) ---
    for (int j = 0; j < 2; j++) {
        Tensor input_plus = input;
        input_plus(j) = input(j) + epsilon;
        float loss_plus = computeLoss(fc, input_plus);

        Tensor input_minus = input;
        input_minus(j) = input(j) - epsilon;
        float loss_minus = computeLoss(fc, input_minus);

        float numerical = (loss_plus - loss_minus) / (2 * epsilon);
        std::cout << "X[" << j << "] analytical=" << dX_returned(j)
                  << " numerical=" << numerical
                  << " diff=" << std::abs(dX_returned(j) - numerical) << "\n";
    }

    return 0;
}