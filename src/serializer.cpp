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
    if (!out.is_open()) {
        std::cerr << "[Serializer Error] Could not open file for writing: " << filepath << std::endl;
        return false;
    }

    uint32_t magic = 0x534E5731; 
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    for (const auto& layer : net.getLayers()) {
        // Handle Convolution Layer
        auto conv = dynamic_cast<Convolution*>(layer.get());
        if (conv) {
            auto& filters = conv->getFilters();
            int num_filters = filters.size();
            out.write(reinterpret_cast<const char*>(&num_filters), sizeof(int));
            for (const auto& filter : filters) {
                writeTensor(out, filter);
            }

            auto& biases = conv->getBiases();
            int num_biases = biases.size();
            out.write(reinterpret_cast<const char*>(&num_biases), sizeof(int));
            out.write(reinterpret_cast<const char*>(biases.data()), num_biases * sizeof(float));
            continue;
        }

        // Handle Fully Connected Layer
        auto fc = dynamic_cast<FCLayer*>(layer.get());
        if (fc) {
            writeEigenMatrix(out, fc->getWeights());
            writeEigenVector(out, fc->getBias());
            continue;
        }
    }

    out.close();
    std::cout << "[Serializer] Successfully saved weights to " << filepath << std::endl;
    return true;
}

bool Serializer::load(Network& net, const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[Serializer Error] Could not open file for reading: " << filepath << std::endl;
        return false;
    }

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x534E5731) {
        std::cerr << "[Serializer Error] Invalid binary format!" << std::endl;
        return false;
    }

    for (auto& layer : net.getLayers()) {
        // Handle Convolution Layer
        auto conv = dynamic_cast<Convolution*>(layer.get());
        if (conv) {
            int num_filters;
            in.read(reinterpret_cast<char*>(&num_filters), sizeof(int));
            auto& filters = conv->getFilters();
            for (auto& filter : filters) {
                if (!readTensor(in, filter)) return false;
            }

            int num_biases;
            in.read(reinterpret_cast<char*>(&num_biases), sizeof(int));
            auto& biases = conv->getBiases();
            in.read(reinterpret_cast<char*>(biases.data()), num_biases * sizeof(float));
            continue;
        }

        // Handle Fully Connected Layer
        auto fc = dynamic_cast<FCLayer*>(layer.get());
        if (fc) {
            if (!readEigenMatrix(in, fc->getWeightsMutable())) return false;
            if (!readEigenVector(in, fc->getBiasMutable())) return false;
            continue;
        }
    }

    in.close();
    std::cout << "[Serializer] Successfully loaded weights from " << filepath << std::endl;
    return true;
}