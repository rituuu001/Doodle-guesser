#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include<iomanip>

#include "../include/sketchguesser/dataset.hpp"
#include "../include/sketchguesser/loss.hpp"
#include "../include/sketchguesser/tensor.hpp"
#include "../include/sketchguesser/optimizer.hpp"
#include "../include/sketchguesser/network.hpp"

Tensor prepareInputTensor(const uint8_t* raw_pixels)
{
    Tensor input(1, Dataset::imageHeight, Dataset::imageWidth);
    for(size_t i =0; i < Dataset::imageSize; ++i)
    {
        input(i) = static_cast<float>(raw_pixels[i]) / 255.0f;
    }
    return input;
}

Tensor prepareTargetTensor(uint8_t label, int num_classes)
{
    Tensor target(1, 1, num_classes);
    for(int i =0; i <num_classes; ++i)
    {
        target(i) = (i == label) ? 1.0f : 0.0f;
    }
    return target;
}

int main(int argc, char** argv)
{
    std::cout<<"\n Task 4 : Small subset training run \n";

    std::string dataset_path = "data/dataset.bin";
    if (argc > 1)
    {
        dataset_path = argv[1];
    }

    std::cout<<"Loading dataset from: "<<dataset_path<<"...\n"<<std::endl;
    Dataset dataset(dataset_path);

    size_t total_available = dataset.numImages();
    int num_classes = dataset.numClasses();

    std::cout << "total dataset images : "<<total_available<<"\n";
    std::cout<<" number of classes: "<<num_classes<<"\n\n";

    size_t subset_size = std::min<size_t> (100, total_available);
    int epochs = 10;
    float learning_rate = 0.01f;

    std::cout << "--- Training Setup ---\n";
    std::cout << "Subset size: " << subset_size << " samples\n";
    std::cout << "Epochs:      " << epochs << "\n";
    std::cout << "Learning Rate: " << learning_rate << "\n";
    std::cout << "----------------------\n\n";

    Network net;
    CrossEntropyLoss loss_func;
    Optimizer optimizer(learning_rate);

    std::vector<size_t> indices(subset_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::default_random_engine rng(42);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        int correct_predictions = 0;

        for (size_t idx : indices) {
            const uint8_t* raw_img = dataset.getImage(idx);
            uint8_t true_label = dataset.getLabel(idx);

            Tensor input = prepareInputTensor(raw_img);
            Tensor target = prepareTargetTensor(true_label, num_classes);

            // 1. Forward Pass
            Tensor output = net.forward(input);
            float loss_val = loss_func.forward(output, target);
            epoch_loss += loss_val;

            // Argmax calculation for accuracy
            int predicted_class = 0;
            float max_prob = output(0);
            for (int c = 1; c < num_classes; ++c) {
                if (output(c) > max_prob) {
                    max_prob = output(c);
                    predicted_class = c;
                }
            }
            if (predicted_class == true_label) {
                correct_predictions++;
            }

            // 2. Backward Pass
            Tensor grad_loss = loss_func.backward(output, target);
            net.backward(grad_loss);

            // 3. Optimizer Step
            optimizer.step(net.getLayers());
        }

        float avg_loss = epoch_loss / subset_size;
        float accuracy = (static_cast<float>(correct_predictions) / subset_size) * 100.0f;

        std::cout << "Epoch [" << std::setw(2) << epoch << "/" << epochs << "] "
                  << "| Avg Loss: " << std::fixed << std::setprecision(4) << avg_loss << " "
                  << "| Accuracy: " << std::setprecision(2) << accuracy << "%\n";
    }

    std::cout << "\n=========================================\n";
    std::cout << "   Small-Subset Run Complete!            \n";
    std::cout << "=========================================\n";

    return 0;
}