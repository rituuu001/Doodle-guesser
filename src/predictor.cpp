#include "sketchguesser/predictor.hpp"
#include "sketchguesser/preprocessing.hpp"
#include "sketchguesser/serializer.hpp"


bool Predictor::loadModel(const std::string& path)
{
    return Serializer::load(network,path);
}


std::string Predictor::predict(const uint8_t* buffer,int width,int height)
{
   Tensor input = DoodleGuesser::preprocess(buffer, width, height);

    Tensor output = network.forward(input);

    int predicted = 0;
    float maxProb = output(0);

    for(int i = 1; i < 6; i++)
    {
        if(output(i) > maxProb)
        {
            maxProb = output(i);
            predicted = i;
        }
    }


    const std::vector<std::string> labels = {
        "cup",
        "donut",
        "envelope",
        "guitar",
        "house",
        "pants"
    };


    return labels[predicted];
}