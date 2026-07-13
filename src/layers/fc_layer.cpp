#include"sketchguesser/layers/fc_layer.hpp"
#include<cmath>

FCLayer::FCLayer(int input_size,int output_size)
{
    this->input_size=input_size;
    this->output_size=output_size;
    weights_=Eigen::MatrixXf::Random(output_size,input_size);
    weights_ *= 1.0f / std::sqrt(static_cast<float>(input_size));
    bias_ = Eigen::VectorXf::Zero(output_size);

}

Tensor FCLayer::forward(const Tensor& tensor_input) 
{ 
    if (tensor_input.size()!= input_size)
     throw std::invalid_argument("Input size mismatch " );

    Eigen::VectorXf eig_input(input_size);
    for (int i=0;i<input_size;i++)
    {
        eig_input[i]=tensor_input(i);
    }
    Eigen::VectorXf eig_output = weights_* eig_input + bias_;
    Tensor tensor_output(1,1,output_size);
    for(int i=0;i<output_size;i++)
    {
        tensor_output(i)=eig_output[i];
    }
    return tensor_output;

}