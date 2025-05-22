#include "cnn_classifier.h"

// �����ʵ��
Matrix sigmoid(const Matrix& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

Matrix sigmoid_derivative(const Matrix& x) {
    Matrix sig = sigmoid(x);
    return sig.array() * (1.0 - sig.array());
}

Matrix relu(const Matrix& x) {
    return x.array().max(0.0);
}

Matrix relu_derivative(const Matrix& x) {
    return (x.array() > 0).cast<float>();
}

// �����ʵ��
ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride, int padding)
    : input_channels_(input_channels), output_channels_(output_channels),
    kernel_size_(kernel_size), stride_(stride), padding_(padding) {

    // �����ʼ������˺�ƫ��
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 0.1);

    kernels_.resize(output_channels_);
    for (int i = 0; i < output_channels_; ++i) {
        kernels_[i] = Matrix::Zero(kernel_size_, kernel_size_);
        for (int j = 0; j < kernel_size_; ++j) {
            for (int k = 0; k < kernel_size_; ++k) {
                kernels_[i](j, k) = dist(gen);
            }
        }
    }

    biases_ = Vector::Zero(output_channels_);
}

Matrix ConvolutionalLayer::forward(const Matrix& input) {
    input_ = input;
    int input_height = input.rows();
    int input_width = input.cols();

    // ��������ߴ�
    int output_height = (input_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int output_width = (input_width + 2 * padding_ - kernel_size_) / stride_ + 1;

    Matrix output = Matrix::Zero(output_height, output_width);

    // Ӧ�þ��
    for (int oc = 0; oc < output_channels_; ++oc) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                float sum = biases_(oc);
                for (int ic = 0; ic < input_channels_; ++ic) {
                    for (int ki = 0; ki < kernel_size_; ++ki) {
                        for (int kj = 0; kj < kernel_size_; ++kj) {
                            int in_i = i * stride_ + ki - padding_;
                            int in_j = j * stride_ + kj - padding_;

                            if (in_i >= 0 && in_i < input_height && in_j >= 0 && in_j < input_width) {
                                sum += input(in_i, in_j) * kernels_[oc](ki, kj);
                            }
                        }
                    }
                }
                output(i, j) = sum;
            }
        }
    }

    // Ӧ��ReLU�����
    return relu(output);
}

Matrix ConvolutionalLayer::backward(const Matrix& grad_output, float learning_rate) {
    int input_height = input_.rows();
    int input_width = input_.cols();
    int output_height = grad_output.rows();
    int output_width = grad_output.cols();

    // ����ReLU����
    Matrix grad_relu = grad_output.array() * relu_derivative(forward(input_)).array();

    // ��ʼ�������ݶ�
    Matrix grad_input = Matrix::Zero(input_height, input_width);

    // ���������ݶȺ������ݶ�
    for (int oc = 0; oc < output_channels_; ++oc) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                float gradient = grad_relu(i, j);

                // ����ƫ��
                biases_(oc) -= learning_rate * gradient;

                // ���¾����
                for (int ic = 0; ic < input_channels_; ++ic) {
                    for (int ki = 0; ki < kernel_size_; ++ki) {
                        for (int kj = 0; kj < kernel_size_; ++kj) {
                            int in_i = i * stride_ + ki - padding_;
                            int in_j = j * stride_ + kj - padding_;

                            if (in_i >= 0 && in_i < input_height && in_j >= 0 && in_j < input_width) {
                                kernels_[oc](ki, kj) -= learning_rate * gradient * input_(in_i, in_j);
                                grad_input(in_i, in_j) += gradient * kernels_[oc](ki, kj);
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

// �ػ���ʵ��
PoolingLayer::PoolingLayer(int pool_size, int stride)
    : pool_size_(pool_size), stride_(stride) {
}

Matrix PoolingLayer::forward(const Matrix& input) {
    input_ = input;
    int input_height = input.rows();
    int input_width = input.cols();

    // ��������ߴ�
    int output_height = (input_height - pool_size_) / stride_ + 1;
    int output_width = (input_width - pool_size_) / stride_ + 1;

    Matrix output = Matrix::Zero(output_height, output_width);
    max_indices_.clear();

    // Ӧ�����ػ�
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float max_val = -INFINITY;
            int max_i = 0, max_j = 0;

            for (int ki = 0; ki < pool_size_; ++ki) {
                for (int kj = 0; kj < pool_size_; ++kj) {
                    int in_i = i * stride_ + ki;
                    int in_j = j * stride_ + kj;

                    if (input(in_i, in_j) > max_val) {
                        max_val = input(in_i, in_j);
                        max_i = in_i;
                        max_j = in_j;
                    }
                }
            }

            output(i, j) = max_val;
            max_indices_.push_back({ max_i, max_j });
        }
    }

    return output;
}

Matrix PoolingLayer::backward(const Matrix& grad_output) {
    int input_height = input_.rows();
    int input_width = input_.cols();
    int output_height = grad_output.rows();
    int output_width = grad_output.cols();

    Matrix grad_input = Matrix::Zero(input_height, input_width);

    // �����ݶ�
    int index = 0;
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            auto [max_i, max_j] = max_indices_[index++];
            grad_input(max_i, max_j) = grad_output(i, j);
        }
    }

    return grad_input;
}

// ȫ���Ӳ�ʵ��
FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size)
    : input_size_(input_size), output_size_(output_size) {

    // �����ʼ��Ȩ�غ�ƫ��
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 0.1);

    weights_ = Matrix::Zero(output_size_, input_size_);
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            weights_(i, j) = dist(gen);
        }
    }

    biases_ = Vector::Zero(output_size_);
}

Matrix FullyConnectedLayer::forward(const Matrix& input) {
    input_ = input;
    std::cout << weights_.cols() << " " << input.rows();
    return (weights_ * input).array() + biases_.array();
}

Matrix FullyConnectedLayer::backward(const Matrix& grad_output, float learning_rate) {
    Matrix grad_input = weights_.transpose() * grad_output;
    Matrix grad_weights = grad_output * input_.transpose();

    // ����Ȩ�غ�ƫ��
    weights_ -= learning_rate * grad_weights;
    biases_ -= learning_rate * grad_output.col(0);

    return grad_input;
}

// ��ƽ����ʵ��
Matrix FlattenLayer::forward(const Matrix& input) {
    rows_ = input.rows();
    cols_ = input.cols();
    return input.reshaped<Eigen::RowMajor>(rows_ * cols_, 1);
}

Matrix FlattenLayer::backward(const Matrix& grad_output) {
    return grad_output.reshaped<Eigen::RowMajor>(rows_, cols_);
}

// CNNģ��ʵ��
void CNN::addLayer(std::unique_ptr<ConvolutionalLayer> layer) {
    conv_layers_.push_back(std::move(layer));
}

void CNN::addLayer(std::unique_ptr<PoolingLayer> layer) {
    pool_layers_.push_back(std::move(layer));
}

void CNN::addLayer(std::unique_ptr<FullyConnectedLayer> layer) {
    fc_layers_.push_back(std::move(layer));
}

void CNN::addLayer(std::unique_ptr<FlattenLayer> layer) {
    flatten_layer_ = std::move(layer);
}

Matrix CNN::forward(const Matrix& input) {
    Matrix current_output = input;

    // ͨ�������ͳػ���
    for (size_t i = 0; i < conv_layers_.size(); ++i) {
        current_output = conv_layers_[i]->forward(current_output);
        conv_outputs_.push_back(current_output);

        if (i < pool_layers_.size()) {
            current_output = pool_layers_[i]->forward(current_output);
            pool_outputs_.push_back(current_output);
        }
    }

    // ͨ����ƽ����
    current_output = flatten_layer_->forward(current_output);
    flatten_output_ = current_output;

    // ͨ��ȫ���Ӳ�
    for (size_t i = 0; i < fc_layers_.size(); ++i) {
        current_output = fc_layers_[i]->forward(current_output);
        fc_outputs_.push_back(current_output);
    }

    // Ӧ��softmax�����
    Matrix exp = current_output.array().exp();
    return exp / exp.sum();
}

void CNN::backward(const Matrix& grad_output, float learning_rate) {
    // ��ʼ���ݶ�
    Matrix current_grad = grad_output;

    // ȫ���Ӳ㷴�򴫲�
    for (int i = fc_layers_.size() - 1; i >= 0; --i) {
        current_grad = fc_layers_[i]->backward(current_grad, learning_rate);
    }

    // ��ƽ���㷴�򴫲�
    current_grad = flatten_layer_->backward(current_grad);

    // �ػ���;���㷴�򴫲�
    for (int i = pool_layers_.size() - 1; i >= 0; --i) {
        current_grad = pool_layers_[i]->backward(current_grad);
        current_grad = conv_layers_[i]->backward(current_grad, learning_rate);
    }

    // ��ջ���
    conv_outputs_.clear();
    pool_outputs_.clear();
    fc_outputs_.clear();
}

void CNN::train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets,
    int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        int correct = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // ǰ�򴫲�
            Matrix output = forward(inputs[i]);

            // ������ʧ (������)
            float loss = -targets[i].array().cwiseProduct(output.array().log()).sum();
            total_loss += loss;

            // ����Ԥ�����
            int predicted = output.col(0).maxCoeff(&predicted);
            int target = targets[i].col(0).maxCoeff(&target);

            if (predicted == target) {
                correct++;
            }

            // �����ݶ�
            Matrix grad_output = output - targets[i];

            // ���򴫲�
            backward(grad_output, learning_rate);
        }

        // ��ӡѵ������
        float accuracy = static_cast<float>(correct) / inputs.size();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
            << " | Loss: " << total_loss / inputs.size()
            << " | Accuracy: " << accuracy << std::endl;
    }
}

int CNN::predict(const Matrix& input) {
    Matrix output = forward(input);
    int predicted;
    output.col(0).maxCoeff(&predicted);
    return predicted;
}

// ������ȡ��ʵ��
std::vector<Matrix> TLSPacketFeatureExtractor::loadFeaturesFromDirectory(const std::string& dir_path) {
    std::vector<Matrix> features;
    // ʵ��ʵ���У�������Ҫ��ȡpcap�ļ�����ȡ����
    // �������ʾ���������������
    for (int i = 0; i < 10; ++i) {
        Matrix feature = Matrix::Random(100, 2); // ������������Ϊ100x2
        features.push_back(feature);
    }
    return features;
}

std::vector<int> TLSPacketFeatureExtractor::loadLabelsFromDirectory(const std::string& dir_path) {
    std::vector<int> labels;
    // ʵ��ʵ���У�������Ҫ���ļ���������Ԫ��������ȡ��ǩ
    // �������ʾ�������������ǩ
    for (int i = 0; i < 10; ++i) {
        labels.push_back(i % 5); // ������5�����
    }
    return labels;
}