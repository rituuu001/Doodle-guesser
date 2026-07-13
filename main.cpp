<<<<<<< HEAD
#include "sketchguesser/dataset.hpp"
#include <iostream>

int main()
{
    try
    {
        Dataset dataset("data/processed/dataset.bin");

        std::cout << "numImages: " << dataset.numImages() << std::endl;
        std::cout << "numClasses: " << dataset.numClasses() << std::endl;

        std::cout << "First label: " << static_cast<int>(dataset.getLabel(0)) << std::endl;
        std::cout << "Last label: " << static_cast<int>(dataset.getLabel(dataset.numImages() - 1)) << std::endl;

        const uint8_t* img = dataset.getImage(0);
        std::cout << "First image, first 10 pixel values: ";
        for (int i = 0; i < 10; i++)
            std::cout << static_cast<int>(img[i]) << " ";
        std::cout << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

=======
#include "sketchguesser/layers/fc_layer.hpp"
#include "sketchguesser/layers/softmax_layer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    // --- FCLayer Test 1: zero input -> zero output (bias is zero-initialized) ---
    {
        FCLayer layer(4, 3);
        Tensor zero_input(1, 1, 4);

        Tensor output = layer.forward(zero_input);

        assert(output.size() == 3);
        for (int i = 0; i < output.size(); ++i) {
            assert(output(i) == 0.0f);
        }
        std::cout << "[PASS] FCLayer: zero input -> zero output\n";
    }

    // --- FCLayer Test 2: mismatched input size throws ---
    {
        FCLayer layer(4, 3);
        Tensor bad_input(1, 1, 5);

        bool threw = false;
        try {
            Tensor output = layer.forward(bad_input);
        } catch (const std::invalid_argument& e) {
            threw = true;
        }
        assert(threw);
        std::cout << "[PASS] FCLayer: mismatched size throws\n";
    }

    // --- FCLayer Test 3: FC1-scale (676 -> 32) runs without crashing ---
    {
        FCLayer layer(676, 32);
        Tensor input(1, 1, 676);
        for (int i = 0; i < input.size(); ++i) input(i) = 0.01f * i;

        Tensor output = layer.forward(input);
        assert(output.size() == 32);
        std::cout << "[PASS] FCLayer: FC1-sized layer runs, output size = " << output.size() << "\n";
    }

    // --- SoftmaxLayer Test 1: output sums to 1 ---
    {
        SoftmaxLayer softmax;
        Tensor input(1, 1, 6);
        float raw[6] = {2.0f, 1.0f, 0.1f, -1.0f, 3.0f, 0.5f};
        for (int i = 0; i < 6; ++i) input(i) = raw[i];

        Tensor output = softmax.forward(input);

        assert(output.size() == 6);
        float sum = 0.0f;
        std::cout << "Softmax output: ";
        for (int i = 0; i < output.size(); ++i) {
            std::cout << output(i) << " ";
            sum += output(i);
        }
        std::cout << "\n";

        std::cout << "Sum of outputs: " << sum << "\n";
        assert(std::fabs(sum - 1.0f) < 1e-5f);
        std::cout << "[PASS] SoftmaxLayer: outputs sum to 1\n";
    }

    // --- SoftmaxLayer Test 2: numerical stability with large values ---
    {
        SoftmaxLayer softmax;
        Tensor input(1, 1, 3);
        input(0) = 1000.0f;
        input(1) = 1000.0f;
        input(2) = 1000.0f;

        Tensor output = softmax.forward(input);

        // all inputs equal -> should be a uniform distribution, no NaN/inf
        std::cout << "Large-value softmax output: ";
        for (int i = 0; i < output.size(); ++i) {
            std::cout << output(i) << " ";
            assert(std::isfinite(output(i)));
        }
        std::cout << "\n";
        std::cout << "[PASS] SoftmaxLayer: no overflow with large inputs\n";
    }

    // --- Chained test: FC2 -> Softmax, mimicking real architecture (32 -> 6 -> softmax) ---
    {
        FCLayer fc2(32, 6);
        SoftmaxLayer softmax;

        Tensor input(1, 1, 32);
        for (int i = 0; i < input.size(); ++i) input(i) = 0.05f * (i - 16);

        Tensor logits = fc2.forward(input);
        Tensor probs = softmax.forward(logits);

        float sum = 0.0f;
        std::cout << "Chained FC2->Softmax output: ";
        for (int i = 0; i < probs.size(); ++i) {
            std::cout << probs(i) << " ";
            sum += probs(i);
        }
        std::cout << "\n";
        assert(std::fabs(sum - 1.0f) < 1e-5f);
        std::cout << "[PASS] Chained FC2 -> Softmax works, sums to 1\n";
    }

    std::cout << "\nAll tests passed.\n";
>>>>>>> 5ba6ed9 (feat: Implemented fc and softmax layers forward pass)
    return 0;
}