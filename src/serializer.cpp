#include "sketchguesser/serializer.hpp"
#include "sketchguesser/layers/conv_layer.hpp"
#include "sketchguesser/layers/fc_layer.hpp"
#include <fstream>
#include <iostream>


// Helper: Save/Load standard Tensor
static void writeTensor(std::ofstream& out, const Tensor& tensor) {
    int c = tensor.getChannels();
    int h = tensor.getHeight();
    int w = tensor.getWidth();

    out.write(reinterpret_cast<const char*>(&c), sizeof(int));
    out.write(reinterpret_cast<const char*>(&h), sizeof(int));
    out.write(reinterpret_cast<const char*>(&w), sizeof(int));

    for (int i = 0; i < tensor.size(); ++i) {
        float val = tensor[i];
        out.write(reinterpret_cast<const char*>(&val), sizeof(float));
    }
}

static bool readTensor(std::ifstream& in, Tensor& tensor) {
    int c, h, w;
    in.read(reinterpret_cast<char*>(&c), sizeof(int));
    in.read(reinterpret_cast<char*>(&h), sizeof(int));
    in.read(reinterpret_cast<char*>(&w), sizeof(int));

    if (c != tensor.getChannels() || h != tensor.getHeight() || w != tensor.getWidth()) {
        std::cerr << "[Serializer Error] Tensor shape mismatch!\n";
        return false;
    }

    for (int i = 0; i < tensor.size(); ++i) {
        float val;
        in.read(reinterpret_cast<char*>(&val), sizeof(float));
        tensor[i] = val;
    }
    return in.good();
}

// Helper: Save/Load Eigen Matrix (for FCLayer weights)
static void writeEigenMatrix(std::ofstream& out, const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(mat.data()), rows * cols * sizeof(float));
}

static bool readEigenMatrix(std::ifstream& in, Eigen::MatrixXf& mat) {
    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));

    if (rows != mat.rows() || cols != mat.cols()) {
        std::cerr << "[Serializer Error] Eigen Matrix shape mismatch!\n";
        return false;
    }

    in.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(float));
    return in.good();
}

// Helper: Save/Load Eigen Vector (for FCLayer biases)
static void writeEigenVector(std::ofstream& out, const Eigen::VectorXf& vec) {
    int size = vec.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(int));
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
}

static bool readEigenVector(std::ifstream& in, Eigen::VectorXf& vec) {
    int size;
    in.read(reinterpret_cast<char*>(&size), sizeof(int));

    if (size != vec.size()) {
        std::cerr << "[Serializer Error] Eigen Vector size mismatch!\n";
        return false;
    }

    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    return in.good();
}

bool Serializer::save(const Network& net, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;

    uint32_t magic = 0x534E5731; // Unique binary header
    save_bytes(out, &magic);

    for (const auto& layer : net.getLayers()) {
        if (auto conv = dynamic_cast<Convolution*>(layer.get())) {
            int num_filters = conv->getFilters().size();
            save_bytes(out, &num_filters);
            
            for (const auto& filter : conv->getFilters()) {
                int c = filter.getChannels(), h = filter.getHeight(), w = filter.getWidth();
                save_bytes(out, &c); save_bytes(out, &h); save_bytes(out, &w);
                for (int i = 0; i < filter.size(); ++i) {
                    float val = filter(i);
                    save_bytes(out, &val);
                }
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
                for (int i = 0; i < filter.size(); ++i) {
                    load_bytes(in, &filter(i));
                }
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