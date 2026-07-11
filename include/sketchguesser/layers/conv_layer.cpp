#include "conv_layer.hpp"

#include <cstdlib>   // rand(), srand()
#include <ctime>     // time()

// Constructor
Convolution::Convolution(int numFilters, int kernelSize)
    : numFilters(numFilters), kernelSize(kernelSize)
{
    // Seed the random number generator only once
    static bool seeded = false;

    if (!seeded)
    {
        std::srand(std::time(nullptr));
        seeded = true;
    }

    // Create each filter
    for (int f = 0; f < numFilters; f++)
    {
        Tensor filter(1, kernelSize, kernelSize);

        // Fill filter with random values between -1 and 1
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                filter(0, i, j) =
                    (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
            }
        }

        // Store the filter
        filters.push_back(filter);
    }
}

// Forward propagation
Tensor Convolution::forward(const Tensor& input)
{
    // Get input dimensions
    int inputHeight = input.getHeight();
    int inputWidth = input.getWidth();

    // Calculate output dimensions
    int outputHeight = inputHeight - kernelSize + 1;
    int outputWidth = inputWidth - kernelSize + 1;

    // Output tensor
    Tensor output(numFilters, outputHeight, outputWidth);

    // Apply every filter
    for (int f = 0; f < numFilters; f++)
    {
        // Slide the filter across the input
        for (int row = 0; row < outputHeight; row++)
        {
            for (int col = 0; col < outputWidth; col++)
            {
                float sum = 0.0f;

                // Perform cross-correlation
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        sum += input(0, row + i, col + j)
                             * filters[f](0, i, j);
                    }
                }

                // Store the computed value
                output(f, row, col) = sum;
            }
        }
    }

    return output;
}

// Backward propagation (implemented later)
Tensor Convolution::backward(const Tensor& gradient)
{
    // Placeholder for backpropagation
    return gradient;
}