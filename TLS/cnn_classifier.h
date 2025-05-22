#pragma once
#ifndef CNN_CLASSIFIER_H
#define CNN_CLASSIFIER_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

// 激活函数及其导数
Matrix sigmoid(const Matrix& x);
Matrix sigmoid_derivative(const Matrix& x);
Matrix relu(const Matrix& x);
Matrix relu_derivative(const Matrix& x);

// 卷积层
class ConvolutionalLayer {
public:
    ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride = 1, int padding = 0);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output, float learning_rate);

private:
    int input_channels_, output_channels_, kernel_size_, stride_, padding_;
    std::vector<Matrix> kernels_;
    Vector biases_;
    Matrix input_;
};

// 池化层
class PoolingLayer {
public:
    PoolingLayer(int pool_size, int stride);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);

private:
    int pool_size_, stride_;
    Matrix input_;
    std::vector<std::pair<int, int>> max_indices_;
};

// 全连接层
class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output, float learning_rate);

private:
    int input_size_, output_size_;
    Matrix weights_;
    Vector biases_;
    Matrix input_;
};

// 扁平化层
class FlattenLayer {
public:
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);

private:
    int rows_, cols_;
};

// CNN模型
class CNN {
public:
    void addLayer(std::unique_ptr<ConvolutionalLayer> layer);
    void addLayer(std::unique_ptr<PoolingLayer> layer);
    void addLayer(std::unique_ptr<FullyConnectedLayer> layer);
    void addLayer(std::unique_ptr<FlattenLayer> layer);

    Matrix forward(const Matrix& input);
    void backward(const Matrix& grad_output, float learning_rate);

    void train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets,
        int epochs, float learning_rate);
    int predict(const Matrix& input);

private:
    std::vector<std::unique_ptr<ConvolutionalLayer>> conv_layers_;
    std::vector<std::unique_ptr<PoolingLayer>> pool_layers_;
    std::vector<std::unique_ptr<FullyConnectedLayer>> fc_layers_;
    std::unique_ptr<FlattenLayer> flatten_layer_;

    std::vector<Matrix> conv_outputs_;
    std::vector<Matrix> pool_outputs_;
    Matrix flatten_output_;
    std::vector<Matrix> fc_outputs_;
};

// 特征提取器
class TLSPacketFeatureExtractor {
public:
    static std::vector<Matrix> loadFeaturesFromDirectory(const std::string& dir_path);
    static std::vector<int> loadLabelsFromDirectory(const std::string& dir_path);
};

#endif    