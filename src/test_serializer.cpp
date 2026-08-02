#include "sketchguesser/network.hpp"
#include "sketchguesser/serializer.hpp"
#include "sketchguesser/tensor.hpp"
#include <iostream>
#include <cmath>

int main() {
    std::cout << "Starting Serialization Round-Trip Test\n\n";

    // 1. Instantiate Network A
    Network netA;

    // 2. Create a dummy input (1 channel, 28x28 image grid)
    Tensor input(1, 28, 28);
    for (int i = 0; i < input.size(); ++i) {
        input(i) = static_cast<float>(i) / 1000.0f; // Fill with test pattern
    }

    // 3. Compute initial output from Network A
    std::cout << "Step 1: Computing prediction from Network A...\n";
    Tensor outputA = netA.forward(input);

    // 4. Save Network A's weights to a binary file
    std::string filepath = "test_model.bin";
    std::cout << "Step 2: Saving Network A weights to '" << filepath << "'...\n";
    if (!Serializer::save(netA, filepath)) {
        std::cerr << "Failed to save model weights!\n";
        return 1;
    }

    // 5. Instantiate Network B (starts with fresh initialized weights)
    std::cout << "Step 3: Initializing fresh Network B...\n";
    Network netB;

    // 6. Load saved weights into Network B
    std::cout << "Step 4: Loading saved weights into Network B...\n";
    if (!Serializer::load(netB, filepath)) {
        std::cerr << "Failed to load model weights!\n";
        return 1;
    }

    // 7. Compute output from Network B using the SAME dummy input
    std::cout << "Step 5: Computing prediction from Network B...\n";
    Tensor outputB = netB.forward(input);

    // 8. Compare Network A and Network B predictions
    std::cout << "\nComparing Output Vectors:\n";
    bool is_identical = true;
    for (int i = 0; i < outputA.size(); ++i) {
        float valA = outputA(i);
        float valB = outputB(i);
        float diff = std::abs(valA - valB);

        std::cout << "Class " << i << " | Net A: " << valA << " | Net B: " << valB << " | Diff: " << diff << "\n";

        if (diff > 1e-5f) {
            is_identical = false;
        }
    }

    std::cout << "\nResult:\n";
    if (is_identical) {
        std::cout << "SUCCESS: Network A and Network B outputs are identical.\n";
        std::cout << "Weight save and load functionality verified.\n";
    } else {
        std::cout << "FAILURE: Network B output does not match Network A.\n";
    }

    return 0;
}