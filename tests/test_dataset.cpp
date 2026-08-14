#include "include/sketchguesser/dataset.hpp"
#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<random>

int main()
{
    try{
        Dataset dataset("../data/processed/dataset.bin");
        std::cout<<"---Dataset Load Successful---"<<std::endl;
        std::cout<<"Total Images: "<<dataset.numImages()<<std::endl;

        if (dataset.numImages() > 0)
        {
            std::cout<<"First Image Label: "<<static_cast<int>(dataset.getLabel(0))<<std::endl;
            std::cout<<"First Pixel Value :"<<static_cast<int>(*dataset.getImage(0))<<std::endl;

            size_t total = dataset.numImages();
            std::vector<size_t> indices(total);
            std::iota(indices.begin(), indices.end(), 0);

            std::mt19937 g(42);
            std::shuffle(indices.begin(), indices.end(),g);

            size_t trainSize = static_cast<size_t>(total * 0.8f);

            std::vector<size_t> trainIndices(indices.begin(), indices.begin() + trainSize);
            std::vector<size_t> valIndices(indices.begin() + trainSize, indices.end());

            std::cout<< "\n -- Dataset Split --"<<std::endl;
            std::cout<<"Train Samples (80%): "<<trainIndices.size()<<std::endl;
            std::cout<<"Val Samples (20%): "<<valIndices.size()<<std::endl;
        }
    } catch(const std::exception& e)
    {
        std::cerr << "Error: "<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}
