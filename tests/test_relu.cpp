#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "sketchguesser/layers/maxpool_layer.hpp"

int main()
{
    // 1 feature map, 4x4 input — easy to trace by hand
    Eigen::MatrixXf m(4, 4);
    m << 1.0, 3.0, 2.0, 1.0,
         4.0, 2.0, 0.0, 5.0,
         1.0, 0.0, 3.0, 2.0,
         2.0, 1.0, 4.0, 1.0;

    std::cout << "Input:\n" << m << "\n\n";

    MaxPoolLayer pool;
    auto output = pool.forward({m});

    std::cout << "Output after MaxPool:\n" << output[0] << "\n\n";
    std::cout << "Expected:\n4 5\n2 4\n";

    return 0;
}