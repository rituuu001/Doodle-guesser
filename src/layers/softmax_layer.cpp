#include"sketchguesser/layers/softmax_layer.hpp"
#include <Eigen/Dense>

Tensor SoftmaxLayer::forward(const Tensor& tensor_input)
{
    Eigen::VectorXf eig_input(tensor_input.size());
    for (int i=0;i<tensor_input.size();i++)
    {
        eig_input[i]=tensor_input(i);
    }
    float  max= eig_input.maxCoeff();
    Eigen::VectorXf exponent = (eig_input.array()-max).exp();
    float total =exponent.sum();
    Eigen :: VectorXf eig_output=exponent/total;
    Tensor tensor_output(1,1,eig_output.size());
    for(int i=0;i<eig_output.size();i++)
    {
        tensor_output(i)=eig_output[i];
    }
    return tensor_output;
}