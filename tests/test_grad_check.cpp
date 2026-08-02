#include<iostream>
#include<cmath>
#include<memory>
#include<algorithm>

#include "../include/sketchguesser/network.hpp"
#include "../include/sketchguesser/loss.hpp"
#include "../include/sketchguesser/tensor.hpp"
#include "../include/sketchguesser/layers/fc_layer.hpp"

int main()
{
    std::cout<< "=== Running End-to-End Gradient check ==="<<std::endl;

    std::srand(42);
    //creating dummy input and taret matching network shape
    Tensor input(1, 28, 28);
    for (size_t i =0 ; i< input.getData().size(); ++i)
    {
        input.getData()[i] = (static_cast<float>(rand())/ RAND_MAX)*0.5f + 0.1f;
    }

    Tensor target(1, 1, 6);
    target.getData() = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    Network net;
    auto* fc1 = dynamic_cast<FCLayer*>(net.getLayers()[5].get());

    // Optional safety check to verify the cast succeeded
    if (!fc1) {
        std::cerr << "Error: Layer at index 5 is not an FCLayer!" << std::endl;
        return 1;
    }
    
    CrossEntropyLoss loss_func;

    Tensor output = net.forward(input);
    float loss_val = loss_func.forward(output, target);
    Tensor grad_loss = loss_func.backward(output, target);
    net.backward(grad_loss);

    Eigen::MatrixXf analytical_grad = fc1->getWeightsGradient();

    float epsilon = 1e-4f;
    Eigen::MatrixXf& weights = fc1->getWeightsMutable();
    Eigen::MatrixXf num_grad = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
    float max_rel_error = 0.0f;

    //calc numerical grad using centered diff method
    for (int r = 0; r< weights.rows(); ++r)
    {
        for(int c = 0; c < weights.cols(); ++c)
        {
            float original_w = weights(r,c);

            //loss at w + epsilon
            weights(r,c) = original_w + epsilon;
            Tensor out_plus = net.forward(input);
            float loss_plus = loss_func.forward(out_plus, target);

            //loss at w - epsilon
            weights(r,c) = original_w - epsilon;
            Tensor out_minus = net.forward(input);
            float loss_minus = loss_func.forward(out_minus, target);

            //restore the org weight
            weights(r,c) = original_w;

            //centered diff formula : (f(x+eps) - f(x-eps))/ (2*eps)
            float d_w = (loss_plus - loss_minus) / (2.0f * epsilon);
            num_grad(r,c) = d_w;

            //compare relative error between analytical and numerical gradients
            float ana_w = analytical_grad(r,c);
            float diff = std::abs(ana_w - d_w);
            float denom = std::max(1e-5f, std::abs(ana_w) + std::abs(d_w));
            float rel_error = diff / denom;

            if ( rel_error > max_rel_error )
            {
                max_rel_error = rel_error;
            }
        }

    }

    std::cout << "Max Relative Error: "<<max_rel_error<< std::endl;

    if (max_rel_error < 0.08f)
    {
        std::cout<<"Success: Gradient Check Passed!!"<< std::endl;
    }
    else{
        std::cout<< "Warning: Gradient Check Failed or Error is High"<<std::endl;
    }

    return 0;
}