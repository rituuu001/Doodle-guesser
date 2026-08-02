#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>
#include<iomanip>
#include <fstream>

#include "sketchguesser/dataset.hpp"
#include "sketchguesser/loss.hpp"
#include "sketchguesser/tensor.hpp"
#include "sketchguesser/optimizer.hpp"
#include "sketchguesser/network.hpp"
#include "sketchguesser/serializer.hpp"


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

struct EpochMetrics
{
    float loss = 0.0f;
    float accuracy = 0.0f;
};

enum class Mode
{
    Train,
    Validation
};

EpochMetrics runEpoch(
    Network& net,
    Dataset& dataset,
    CrossEntropyLoss& loss_func,
    Optimizer* optimizer,
    const std::vector<size_t>& indices,
    int num_classes,
    Mode mode)
{
    EpochMetrics metrics;

    float epoch_loss = 0.0f;
    int correct_predictions = 0;

    for(size_t idx : indices)
    {
        const uint8_t* raw_img = dataset.getImage(idx);
        uint8_t true_label = dataset.getLabel(idx);

        Tensor input = prepareInputTensor(raw_img);
        Tensor target = prepareTargetTensor(true_label, num_classes);

        // Forward Pass
        Tensor output = net.forward(input);

        float loss = loss_func.forward(output, target);
        epoch_loss += loss;

        // Accuracy
        int predicted_class = 0;
        float max_prob = output(0);

        for(int c = 1; c < num_classes; c++)
        {
            if(output(c) > max_prob)
            {
                max_prob = output(c);
                predicted_class = c;
            }
        }

        if(predicted_class == true_label)
            correct_predictions++;

        // Training only
        if(mode == Mode::Train)
        {
            Tensor grad_loss = loss_func.backward(output, target);
            net.backward(grad_loss);
            optimizer->step(net.getLayers());
        }
    }

    metrics.loss =epoch_loss/indices.size();
    metrics.accuracy =100.0f * correct_predictions /indices.size();
    return metrics;
}

int main()
{
    std::ofstream logFile("../docs/training_log.csv");

    if(!logFile.is_open())
   {
    std::cerr << "Could not create training log.\n";
    return 1;
   }

    logFile << "Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n";
    std::string dataset_path = "../data/processed/dataset.bin";
    

    std::cout<<"Loading dataset from: "<<dataset_path<<"...\n"<<std::endl;
    Dataset dataset(dataset_path);

    size_t total = dataset.numImages();
    int num_classes = dataset.numClasses();

    std::cout << "total dataset images : "<<total<<"\n";
    std::cout<<" number of classes: "<<num_classes<<"\n\n";

    int epochs = 30;
    float learning_rate = 0.01f;
    float bestValidationAccuracy = 0.0f;

    std::cout << "Training Setup"<<std::endl;
    std::cout << "Epochs: " <<epochs<<std::endl;
    std::cout << "Learning Rate: " <<learning_rate<<std::endl;
   

    Network net;
    CrossEntropyLoss loss_func;
    Optimizer optimizer(learning_rate);

    std::default_random_engine rng(42);

    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), rng);

    size_t trainSize = static_cast<size_t>(0.8 * total);

    std::vector<size_t> trainIndices(indices.begin(),indices.begin() + trainSize);
    std::vector<size_t> valIndices(indices.begin() + trainSize,indices.end());

for(int epoch = 1; epoch <= epochs; epoch++)
{  
    //training loop
    EpochMetrics trainMetrics = runEpoch(net,dataset,loss_func, &optimizer,
                                         trainIndices,num_classes,Mode::Train);

    //validation loop
    EpochMetrics validationMetrics =runEpoch(net,dataset,loss_func,nullptr,
                                         valIndices,num_classes,Mode::Validation);
    
    //displays the loss and accuracy of training and validation loop every epoch                                    
    std::cout << "Epoch "<<epoch<<"/"<<epochs<<std::endl;
    std::cout <<std::fixed<<std::setprecision(4);

    std::cout <<"Train Loss : "<<trainMetrics.loss<< std::endl;
    std::cout << "Train Acc  : "<<trainMetrics.accuracy<<"%"<<std::endl;

    std::cout << "Val Loss   : "<<validationMetrics.loss<<std::endl;
    std::cout << "Val Acc    : "<<validationMetrics.accuracy<< "%"<<std::endl;
    
    //writes the loss and accuracy in training_log.csv
    logFile
    << epoch << ","
    << trainMetrics.loss << ","
    << trainMetrics.accuracy << ","
    << validationMetrics.loss << ","
    << validationMetrics.accuracy
    << "\n";


    //saves the model with highest accuracy
    if(validationMetrics.accuracy > bestValidationAccuracy)
{
    bestValidationAccuracy = validationMetrics.accuracy;

    Serializer::save(net, "../models/doodle_model.bin");

    std::cout << "New best model saved!\n";
}
        
}   
logFile.close();
return 0;
}