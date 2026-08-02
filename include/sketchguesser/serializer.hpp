#pragma once

#include <string>
#include "sketchguesser/network.hpp"

class Serializer {
public:
    // Saves all network trainable weights/biases to a binary file
    static bool save(const Network& net, const std::string& filepath);

    // Loads trainable weights/biases from a binary file into the network
    static bool load(Network& net, const std::string& filepath);
};