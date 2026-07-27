#include "sketchguesser/layers/conv_layer.hpp"
#include<Eigen/Dense>
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
        //initialize biases and dbiases to 0
        biases.push_back(0.0f);
        dBiases.push_back(0.0f);
    }
}

// Forward propagation
Tensor Convolution::forward(const Tensor& input)
{
    // Get input dimensions
    int inChannels = input.getChannels();
    
    int inputHeight = input.getHeight();
    int inputWidth = input.getWidth();
    cachedInput = input;
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
                for(int ch =0; ch < inChannels; ch++)
                {
                    for (int i = 0; i < kernelSize; i++)
                    {
                        for (int j = 0; j < kernelSize; j++)
                        {
                            sum += input(ch, row + i, col + j)
                                 * filters[f](ch, i, j);
                        }
                    }
                }
                sum += biases[f];
                output(f, row, col) = sum;
            }
        }
    }

    return output;
}


Tensor Convolution::backward(const Tensor& gradient)
{
    int inChannels = cachedInput.getChannels();
    int inH = cachedInput.getHeight();
    int inw = cachedInput.getWidth();

    dBiases = std::vector<float>(numFilters, 0.0f);
    dFilters.clear();

    Tensor dInput(cachedInput.getChannels(),cachedInput.getHeight(),cachedInput.getWidth());
    for (int i = 0; i< dInput.size(); i++)
    {
        dInput(i) = 0.0f;
    }
    for (int f = 0; f < numFilters; f++)
        {
            Tensor df(inChannels, kernelSize, kernelSize);
            for (int i = 0; i < df.size(); i++) {
                df(i) = 0.0f;
            }
            dFilters.push_back(df);
        }

        // 3. Accumulate gradients
        for (int f = 0; f < numFilters; f++)
        {
            for (int y = 0; y < gradient.getHeight(); y++)
            {
                for (int x = 0; x < gradient.getWidth(); x++)
                {
                    float g = gradient(f, y, x);
                    dBiases[f] += g;

                    for (int ch = 0; ch < inChannels; ch++)
                    {
                        for (int row = 0; row < kernelSize; row++)
                        {
                            for (int col = 0; col < kernelSize; col++)
                            {
                                dFilters[f](ch, row, col) += g * cachedInput(ch, y + row, x + col);
                                dInput(ch, y + row, x + col) += g * filters[f](ch, row, col);
                            }
                        }
                    }
                }
            }
        }

        return dInput;
    }

void Convolution::update(double learning_rate)
{
    float lr = static_cast<float>(learning_rate);
    for (int f = 0; f < numFilters; f++)
    {
        for (int i = 0; i < filters[f].size(); i++)
            filters[f](i) -= lr * dFilters[f](i);

        biases[f] -= lr * dBiases[f];
    }
}