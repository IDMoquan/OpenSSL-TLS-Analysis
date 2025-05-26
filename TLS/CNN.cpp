﻿#pragma once
#include <vector>
#include "Head.h"
using namespace std;
using namespace Eigen;

void LoadData(vector<Feature>& f ,vector<MatrixXf>& features, vector<int>& labels, const int& label, int len) {
    int size = static_cast<int>(f.size());
    MatrixXf feature(len, 2);
    for (int i = 0; i < size; i++) {
        feature(i, 0) = static_cast<float>(f[i].GetSize());
        feature(i, 1) = f[i].GetDirection();
    }
    for (int i = size; i < len; i++) {
        feature(i, 0) = 0.0f;
        feature(i, 1) = 0.0f;
    }
    features.push_back(feature);
    labels.push_back(label);
}

CNN::CNN(int input_rows, int input_cols, int filter_size, int num_filters, int pool_size, int num_classes)
    : input_rows(input_rows), input_cols(input_cols), filter_size(filter_size), num_filters(num_filters), pool_size(pool_size), num_classes(num_classes) {
    // 初始化卷积核和偏置
    weights.resize(num_filters);
    biases.resize(num_filters);
    for (int i = 0; i < num_filters; ++i) {
        weights[i] = MatrixXf::Random(filter_size, input_cols);
        biases[i] = MatrixXf::Random(1, 1); // 初始化偏置项为随机值
    }

    // 初始化全连接层权重和偏置
    int fc_input_size = ((input_rows - filter_size + 1) / pool_size) * input_cols;
    fc_weights = MatrixXf::Random(fc_input_size, num_classes);
    fc_bias = MatrixXf::Random(1, num_classes);
}

// 前向传播
MatrixXf CNN::forward(const MatrixXf& input) {
    // 卷积层
    for (int i = 0; i < num_filters; ++i) {
        MatrixXf temp_output(input_rows - filter_size + 1, input_cols); // 临时变量存储卷积输出
        conv2d(input, weights[i], temp_output);
        float bias_value = biases[i](0, 0); // 获取偏置项的标量值
        temp_output = temp_output.array() + bias_value; // 将偏置项加到卷积输出
        temp_output = relu(temp_output); // ReLU激活
        if (i == 0) {
            conv_output = temp_output; // 第一次卷积后初始化conv_output
        }
        else {
            conv_output += temp_output; // 累加卷积输出
        }
    }

    // 池化层
    max_pooling(conv_output, pool_size, pooled_output);

    // 将池化层的输出展平为一维向量
    pooled_output_flattened = Eigen::Map<MatrixXf>(pooled_output.data(), pooled_output.size(), 1).transpose();

    // 全连接层
    MatrixXf fc_output =  fully_connected(pooled_output_flattened, fc_weights, fc_bias);

    return softmax(fc_output);

}

// 卷积操作
void CNN::conv2d(const MatrixXf& input, const MatrixXf& kernel, MatrixXf& output) {
    int kernel_size = static_cast<int>(kernel.rows());
    int input_rows = static_cast<int>(input.rows());
    int input_cols = static_cast<int>(input.cols());
    int output_rows = input_rows - kernel_size + 1;
    output.resize(output_rows, input_cols);
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < input_cols; ++j) {
            float sum = 0.0;
            for (int ki = 0; ki < kernel_size; ++ki) {
                sum += input(i + ki, j) * kernel(ki, j);
            }
            output(i, j) = sum;
        }
    }
}

// ReLU激活函数
MatrixXf CNN::relu(const MatrixXf& input) {
    return input.unaryExpr([](float x) { return max(0.0f, x); });
}

//池化层
void CNN::max_pooling(const MatrixXf& input, int pool_size, MatrixXf& output) {
    int input_rows = static_cast<int>(input.rows());
    int input_cols = static_cast<int>(input.cols());
    int output_rows = input_rows / pool_size;
    output.resize(output_rows, input_cols);
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < input_cols; ++j) {
            float max_val = -INFINITY;
            for (int pi = 0; pi < pool_size; ++pi) {
                max_val = max(max_val, input(i * pool_size + pi, j));
            }
            output(i, j) = max_val;
        }
    }
}

// 全连接层
MatrixXf CNN::fully_connected(const MatrixXf& input, const MatrixXf& weights, const MatrixXf& bias) {
    //cout << input.cols() << " " << weights.rows() << endl;
    return (input * weights).rowwise() + bias.row(0);
}

MatrixXf softmax(const MatrixXf& input) {
    float max_vals = input.maxCoeff();
    MatrixXf shifted_input = input.array() - max_vals;
    // 计算每个元素的指数
    MatrixXf exp_input = shifted_input.array().exp();
    float sum = exp_input.sum();

    // 归一化
    return exp_input / sum;
}
pair<MatrixXf, float> CNN::cross_entropy(const MatrixXf& prob, const vector<float>& labels) {
    int num_classes = static_cast<int>(prob.cols());
    float loss_value = 0.0f;
    MatrixXf delta(1, num_classes);
    for (int i = 0; i < num_classes; i++) {
        delta(0, i) = prob(0, i) - labels[i];
        loss_value += (labels[i] * log(prob(0, i)));
    }
    loss_value = -loss_value / num_classes;
    return make_pair(delta, loss_value);
}

void CNN::backward(const float& loss, const MatrixXf& delta, const float& learning_rate) {
    float lambda = 0.001;
    // 计算全连接层的梯度
    MatrixXf d_fc_weights = pooled_output_flattened.transpose() * delta + lambda * fc_weights;;
    MatrixXf d_fc_bias = delta.colwise().sum();

    // 对梯度进行裁剪
    d_fc_weights = d_fc_weights.cwiseMax(-1.0).cwiseMin(1.0);
    d_fc_bias = d_fc_bias.cwiseMax(-1.0).cwiseMin(1.0);

    // 更新全连接层的权重和偏置
    fc_weights -= learning_rate * d_fc_weights;
    fc_bias -= learning_rate * d_fc_bias;

    // 反向传播到池化层
    MatrixXf d_pooled_output_flattened = delta * fc_weights.transpose();
    MatrixXf d_pooled_output = Eigen::Map<MatrixXf>(d_pooled_output_flattened.data(), pooled_output.rows(), pooled_output.cols());

    // 反向传播到卷积层
    MatrixXf d_conv_output;
    max_pooling_backward(d_pooled_output, pool_size, d_conv_output);

    for (int i = 0; i < num_filters; ++i) {
        MatrixXf d_input;
        MatrixXf d_kernel;
        conv2d_backward(conv_output, weights[i], d_conv_output, d_input, d_kernel);
        weights[i] -= learning_rate * d_kernel;
        biases[i](0,0) -= learning_rate * d_input.sum();
    }
}

void CNN::conv2d_backward(const MatrixXf& input, const MatrixXf& kernel,
    const MatrixXf& d_output, MatrixXf& d_input, MatrixXf& d_kernel) {
    int kernel_size = static_cast<int>(kernel.rows());
    int input_rows = static_cast<int>(input.rows());
    int input_cols = static_cast<int>(input.cols());
    int output_rows = static_cast<int>(d_output.rows());
    int output_cols = static_cast<int>(d_output.cols());

    d_input.resizeLike(input);
    d_input.setZero();
    d_kernel.resizeLike(kernel);
    d_kernel.setZero();

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    int row = i + ki;
                    int col = j + kj;
                    if (row < input_rows && col < input_cols) {
                        d_input(row, col) += d_output(i, j) * kernel(ki, kj);
                        d_kernel(ki, kj) += d_output(i, j) * input(row, col);
                    }
                }
            }
        }
    }
}

void CNN::max_pooling_backward(const MatrixXf& d_output, int pool_size, MatrixXf& d_input) {
    int output_rows = static_cast<int>(d_output.rows());
    int output_cols = static_cast<int>(d_output.cols());
    int input_rows = output_rows * pool_size;
    int input_cols = output_cols;
    d_input.resize(input_rows, input_cols);
    d_input.setZero();

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            float max_val = -INFINITY;
            int max_row = 0, max_col = 0;
            for (int pi = 0; pi < pool_size; ++pi) {
                for (int pj = 0; pj < pool_size; ++pj) {
                    int row = i * pool_size + pi;
                    int col = j * pool_size + pj;
                    if (row < input_rows && col < input_cols) {
                        if (conv_output(row, col) > max_val) {
                            max_val = conv_output(row, col);
                            max_row = row;
                            max_col = col;
                        }
                    }
                }
            }
            d_input(max_row, max_col) += d_output(i, j);
        }
    }
}

//通过文件夹名称返回对应标签
int Label_Number(string label) {
    if (label == "bd")  return 1;
    if (label == "bz")  return 2;
    if (label == "csdn")  return 3;
    if (label == "gh")  return 4;
    if (label == "iqy")  return 5;
    if (label == "my")  return 6;
    if (label == "qd")  return 7;
    if (label == "tb")  return 8;
    if (label == "wb")  return 9;
    if (label == "zh")  return 10;
    else return -1;
}

string Website_Name(int label) {
    if (label == 1)  return "www.baidu.com";
    if (label == 2)  return "www.bilibili.com";
    if (label == 3)  return "www.csdn.com";
    if (label == 4)  return "www.github.com";
    if (label == 5)  return "www.iqiyi.com";
    if (label == 6)  return "www.maoyan.com";
    if (label == 7)  return "www.qidian.com";
    if (label == 8)  return "www.taobao.com";
    if (label == 9)  return "www.weibo.com";
    if (label == 10)  return "www.zhihu.com";
    else return "null";
}
