#include"sketchguesser/network.hpp"


Tensor Network::forward(const Tensor& input)
{
      Tensor in=input;
      for (auto& layer:layers)
      {
        Tensor output=layer->forward(in);
        in=output;
      }
      return in;
}

Tensor Network::backward(const Tensor& gradient)
{
      Tensor grad=gradient;
      for (int i=static_cast<int>(layers.size())-1;i>=0;i--)
      {
        Tensor output=layers[i]->backward(grad);
        if (i==3)
        {
          Tensor reshaped(4,13,13);
          for (int c=0;c<4;c++)
           {
             for (int h=0;h<13;h++)
              {
                for (int w=0;w<13;w++)
                 {
                   reshaped(c,h,w)=output(c*reshaped.getHeight()*reshaped.getWidth()+
                                        h*reshaped.getWidth()+w);
                 }
             }
           }
           output=reshaped;
        }
        grad=output;
      }
      return grad;
}

