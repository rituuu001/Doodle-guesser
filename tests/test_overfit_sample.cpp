#include<iostream>
#include<cmath>
#include<memory>
#include<cstdlib>

#include "../include/sketchguesser/network.hpp"
#include "../include/sketchguesser/loss.hpp"
#include "../include/sketchguesser/tensor.hpp"
#include "../include/sketchguesser/optimizer.hpp"

int main()
{
    std::cout<<"Running overfit one sample test"<<std::endl;

    std::srand(42);

    Tensor input(1, 28, 28);
    for(size_t i =0; i< input.getData().size(); i++)
    {
        input.getData()[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    int target_class = 2;
    Tensor target(1, 1, 6);
    target.getData() = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};

    Network net;
    CrossEntropyLoss loss_func;

    float learning_rate = 0.01f;
    Optimizer optimizer(learning_rate);

    int iterations = 200;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    std::cout<<"Training on 1 sample for "<<iterations<<" iterations...\n"<<std::endl;

    for(int iter = 1; iter <= iterations; ++iter)
    {
        Tensor output = net.forward(input);
        float loss_val = loss_func.forward(output, target);

        if(iter == 1)
        {
            initial_loss = loss_val;
        }
        final_loss = loss_val;
        
        Tensor grad_loss = loss_func.backward(output, target);
        net.backward(grad_loss);

        optimizer.step(net.getLayers());

        if(iter % 20 == 0 || iter == 1)
        {
            float target_prob = output.getData()[target_class];
            std::cout<<"Iter ["<<iter<<"/"<<iterations<<"] | Loss: "<<loss_val<<" |Prob(Class "<<target_class<<"): "<<target_prob<<std::endl;
        }

    }

    std::cout<<"\n Initial loss: "<<initial_loss<<std::endl;
    std::cout<<"\n Final loss : "<<final_loss<<std::endl;
    std::cout<<"\n"<<std::endl;

    if (final_loss < initial_loss && final_loss < 0.1f)
    {
        std::cout<<"SUCCESS : model successfully overfitted the sample"<<std::endl;
        return 0;
    }
    else
    {
        std::cout<<"FAILURE: Loss failed to drop significanlty"<<std::endl;
        return 1;
    }
    }