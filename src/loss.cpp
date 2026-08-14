#include "sketchguesser/loss.hpp"
#include <cmath>

float CrossEntropyLoss::forward(const Tensor& prediction, const Tensor& target) {
    float loss = 0.0f;
    const float epsilon = 1e-7f;//to prevent undefined mathematical conditions

    for (int i = 0; i < prediction.size(); i++) {
        loss -= target[i] * std::log(prediction[i] + epsilon);
    }

    return loss;
}

Tensor CrossEntropyLoss::backward(const Tensor& prediction, const Tensor& target) {
    Tensor gradient(
        prediction.getChannels(),
        prediction.getHeight(),
        prediction.getWidth());

    const float epsilon = 1e-7f;

    for (int i = 0; i < prediction.size(); i++) {
        // Raw derivative of -y * log(p):  dL/dp = -y / (p + eps)
        gradient[i] = -target[i] / (prediction[i] + epsilon);
    }

    return gradient;
}