#include "sketchguesser/layers/maxpool_layer.hpp"

Tensor MaxPoolLayer::forward(
    const Tensor& input)
    {
        input_cache = input;
        int num_filters = input.getChannels();
        int in_rows = input.getHeight();
        int in_cols = input.getWidth();
        int out_rows = in_rows / 2;
        int out_cols = in_cols / 2;
        Tensor output(
            num_filters, out_rows, out_cols);
        max_indices.resize(num_filters, std::vector<std::vector<MaxIndex>>(out_rows, std::vector<MaxIndex>(out_cols)));
        for(int d=0; d<num_filters; d++)
        {
            for(int r =0; r<out_rows; r++)
            {
                for(int c=0; c < out_cols; c++)
                {
                    int sr = 2 * r;
                    int sc = 2 * c;

                    float max_val = input(d, sr, sc);
                    int max_r = sr;
                    int max_c = sc;

                    if (input(d,sr, sc+1) > max_val)
                    {
                        max_val = input(d,sr, sc+1);
                        max_r = sr;
                        max_c = sc + 1;

                    }
                    if (input(d,sr+1, sc) > max_val)
                    {
                        max_val = input(d,sr+1, sc);
                        max_r = sr + 1;
                        max_c = sc;
                    }
                    if (input(d,sr+1, sc+1) > max_val)
                    {
                        max_val = input(d,sr+1, sc+1);
                        max_r = sr+1;
                        max_c = sc+1;
                    }

                    output(d,r, c) = max_val;
                    max_indices[d][r][c] = {max_r, max_c};
                }
            }
        }
        return output;
    }


Tensor MaxPoolLayer::backward(
    const Tensor& grad)
{
    int num_filters = input_cache.getChannels();
    int in_rows = input_cache.getHeight();
    int in_cols = input_cache.getWidth();
    int grad_rows = grad.getHeight();
    int grad_cols = grad.getWidth();
    int grad_filters = grad.getChannels();
    Tensor output(num_filters, in_rows, in_cols);
    for(int d=0; d<grad_filters; d++)
    {
        for(int r =0 ;r<grad_rows; r++)
        {
            for(int c =0; c<grad_cols; c++)
            {
                MaxIndex idx = max_indices[d][r][c];
                output(d, idx.row, idx.col) = grad(d, r, c);
            }
        }
    }
    
    return output;
}