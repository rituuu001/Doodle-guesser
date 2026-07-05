#include "sketchguesser/layers/maxpool_layer.hpp"

std::vector<Eigen::MatrixXf> MaxPoolLayer::forward(
    const std::vector<Eigen::MatrixXf>& input)
    {
        input_cache = input;
        int num_filters = input.size();
        int in_rows = input[0].rows();
        int in_cols = input[0].cols();
        int out_rows = in_rows / 2;
        int out_cols = in_cols / 2;
        std::vector<Eigen::MatrixXf> output(
            num_filters, Eigen::MatrixXf(out_rows, out_cols));
        max_indices.resize(num_filters, std::vector<std::vector<MaxIndex>>(out_rows, std::vector<MaxIndex>(out_cols)));
        for(int d=0; d<num_filters; d++)
        {
            for(int r =0; r<out_rows; r++)
            {
                for(int c=0; c < out_cols; c++)
                {
                    int sr = 2 * r;
                    int sc = 2 * c;

                    float max_val = input[d](sr, sc);
                    int max_r = sr;
                    int max_c = sc;

                    if (input[d](sr, sc+1) > max_val)
                    {
                        max_val = input[d](sr, sc+1);
                        max_r = sr;
                        max_c = sc + 1;

                    }
                    if (input[d](sr+1, sc) > max_val)
                    {
                        max_val = input[d](sr+1, sc);
                        max_r = sr + 1;
                        max_c = sc;
                    }
                    if (input[d](sr+1, sc+1) > max_val)
                    {
                        max_val = input[d](sr+1, sc+1);
                        max_r = sr+1;
                        max_c = sc+1;
                    }

                    output[d](r, c) = max_val;
                    max_indices[d][r][c] = {max_r, max_c};
                }
            }
        }
        return output;
    }


std::vector<Eigen::MatrixXf> MaxPoolLayer::backward(
    const std::vector<Eigen::MatrixXf>& grad)
{
    // Week 3 
    return grad;
}