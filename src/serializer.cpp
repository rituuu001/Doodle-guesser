#include "sketchguesser/serializer.hpp"
#include "sketchguesser/layers/conv_layer.hpp"
#include "sketchguesser/layers/fc_layer.hpp"
#include <fstream>
#include <iostream>

template <typename T>
static void save_bytes(std::ofstream& out, const T* data, size_t count = 1) {
    out.write(reinterpret_cast<const char*>(data), count * sizeof(T));
}

template <typename T>
static bool load_bytes(std::ifstream& in, T* data, size_t count = 1) {
    return in.read(reinterpret_cast<char*>(data), count * sizeof(T)).good();
}

bool Serializer::save(const Network& net, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;

    uint32_t magic = 0x534E5731;
    save_bytes(out, &magic);

    for (const auto& layer : net.getLayers()) {
        if (auto conv = dynamic_cast<Convolution*>(layer.get())) {
            int num_filters = conv->getFilters().size();
            save_bytes(out, &num_filters);
            
            for (const auto& filter : conv->getFilters()) {
                int c = filter.getChannels(), h = filter.getHeight(), w = filter.getWidth();
                save_bytes(out, &c); save_bytes(out, &h); save_bytes(out, &w);
                save_bytes(out, filter.getData().data(), filter.size());
            }

            int num_biases = conv->getBiases().size();
            save_bytes(out, &num_biases);
            save_bytes(out, conv->getBiases().data(), num_biases);
        } 
        else if (auto fc = dynamic_cast<FCLayer*>(layer.get())) {
            int rows = fc->getWeights().rows(), cols = fc->getWeights().cols();
            save_bytes(out, &rows); save_bytes(out, &cols);
            save_bytes(out, fc->getWeights().data(), rows * cols);

            int b_size = fc->getBias().size();
            save_bytes(out, &b_size);
            save_bytes(out, fc->getBias().data(), b_size);
        }
    }
    return true;
}

bool Serializer::load(Network& net, const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;

    uint32_t magic = 0;
    load_bytes(in, &magic);
    if (magic != 0x534E5731) return false;

    for (auto& layer : net.getLayers()) {
        if (auto conv = dynamic_cast<Convolution*>(layer.get())) {
            int num_filters;
            load_bytes(in, &num_filters);

            for (auto& filter : conv->getFilters()) {
                int c, h, w;
                load_bytes(in, &c); load_bytes(in, &h); load_bytes(in, &w);
                if (!load_bytes(in, filter.getData().data(), filter.size())) return false;
            }

            int num_biases;
            load_bytes(in, &num_biases);
            load_bytes(in, conv->getBiases().data(), num_biases);
        } 
        else if (auto fc = dynamic_cast<FCLayer*>(layer.get())) {
            int rows, cols;
            load_bytes(in, &rows); load_bytes(in, &cols);
            load_bytes(in, fc->getWeightsMutable().data(), rows * cols);

            int b_size;
            load_bytes(in, &b_size);
            load_bytes(in, fc->getBiasMutable().data(), b_size);
        }
    }
    return true;
}