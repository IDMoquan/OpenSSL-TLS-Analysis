#pragma once
#include <vector>
#include "Head.h"
using namespace std;
using namespace Eigen;

void printPcapFileHeader(Pcap_Header* pfh) {
    if (pfh == NULL) {
        return;
    }
    printf("=====================\n"
        "magic:0x%0x\n"
        "version_major:%u\n"
        "version_minor:%u\n"
        "thiszone:%d\n"
        "sigfigs:%u\n"
        "snaplen:%u\n"
        "linktype:%u\n"
        "=====================\n",
        pfh->magic,
        pfh->major,
        pfh->minor,
        pfh->thiszone,
        pfh->sigfigs,
        pfh->snaplen,
        pfh->linktype);
}

void printPcapHeader(Pcap_Packet_Header* ph) {
    if (ph == NULL) {
        return;
    }
    printf("=====================\n"
        "ts.timestamp_s:%u\n"
        "ts.timestamp_ms:%u\n"
        "capture_len:%u\n"
        "len:%d\n"
        "=====================\n",
        ph->timestamp_sec,
        ph->timestamp_msec,
        ph->caplen,
        ph->len);
}

void printPcap(void* data, size_t size) {
    unsigned  short iPos = 0;
    //int * p = (int *)data;
    //unsigned short* p = (unsigned short *)data;
    if (data == NULL) {
        return;
    }
    printf("\n==data:0x%x,len:%lu=========", data, size);

    for (iPos = 0; iPos < size / sizeof(unsigned short); iPos++) {
        //printf(" %x ",(int)( * (p+iPos) ));
        //unsigned short a = ntohs(p[iPos]);

        unsigned short a = ntohs(*((unsigned short*)data + iPos));
        if (iPos % 8 == 0) printf("\n");
        if (iPos % 4 == 0) printf(" ");

        printf("%04x", a);


    }
    /*
     for (iPos=0; iPos <= size/sizeof(int); iPos++) {
        //printf(" %x ",(int)( * (p+iPos) ));
        int a = ntohl(p[iPos]);

        //int a = ntohl( *((int *)data + iPos ) );
        if (iPos %4==0) printf("\n");

        printf("%08x ",a);


    }
     */
    printf("\n============\n");
}

void LoadData(vector<Feature>& f ,vector<MatrixXf>& features, vector<int>& labels, const int& label, int len) {
    int size = f.size();
    MatrixXf feature(len, 2);
    for (int i = 0; i < size; i++) {
        feature(i, 0) = f[i].GetSize();
        feature(i, 1) = f[i].GetDirection();
    }
    for (int i = size; i < len; i++) {
        feature(i, 0) = 0.0f;
        feature(i, 1) = 0.0f;
    }
    //cout << feature <<  " " << label <<endl << "---------------------" << endl;
    features.push_back(feature);
    labels.push_back(label);
}

CNN::CNN(int input_rows, int input_cols, int filter_size, int num_filters, int pool_size, int num_classes)
    : input_rows(input_rows), input_cols(input_cols), filter_size(filter_size), num_filters(num_filters), pool_size(pool_size), num_classes(num_classes) {
    // ��ʼ������˺�ƫ��
    weights.resize(num_filters);
    biases.resize(num_filters);
    for (int i = 0; i < num_filters; ++i) {
        weights[i] = MatrixXf::Random(filter_size, input_cols);
        biases[i] = MatrixXf::Random(1, 1); // ��ʼ��ƫ����Ϊ���ֵ
    }

    // ��ʼ��ȫ���Ӳ�Ȩ�غ�ƫ��
    int fc_input_size = ((input_rows - filter_size + 1) / pool_size) * input_cols;
    fc_weights = MatrixXf::Random(fc_input_size, num_classes);
    fc_bias = MatrixXf::Random(1, num_classes);
}

// ǰ�򴫲�
MatrixXf CNN::forward(const MatrixXf& input) {
    MatrixXf conv_output;
    MatrixXf pooled_output;

    // �����
    for (int i = 0; i < num_filters; ++i) {
        MatrixXf temp_output(input_rows - filter_size + 1, input_cols); // ��ʱ�����洢������
        conv2d(input, weights[i], temp_output);
        float bias_value = biases[i](0, 0); // ��ȡƫ����ı���ֵ
        temp_output.array() += bias_value; // ��ƫ����ӵ�������
        temp_output = relu(temp_output); // ReLU����
        if (i == 0) {
            conv_output = temp_output; // ��һ�ξ�����ʼ��conv_output
        }
        else {
            conv_output += temp_output; // �ۼӾ�����
        }
    }

    // �ػ���
    max_pooling(conv_output, pool_size, pooled_output);

    // ���ػ�������չƽΪһά����
    MatrixXf pooled_output_flattened = Eigen::Map<MatrixXf>(pooled_output.data(), pooled_output.size(), 1).transpose();

    // ȫ���Ӳ�
    return fully_connected(pooled_output_flattened, fc_weights, fc_bias);
}

// �������
void CNN::conv2d(const MatrixXf& input, const MatrixXf& kernel, MatrixXf& output) {
    int kernel_size = kernel.rows();
    int input_rows = input.rows();
    int input_cols = input.cols();
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

// ReLU�����
MatrixXf CNN::relu(const MatrixXf& input) {
    return input.unaryExpr([](float x) { return max(0.0f, x); });
}


// ���ػ�����
void CNN::max_pooling(const MatrixXf& input, int pool_size, MatrixXf& output) {
    int input_rows = input.rows();
    int input_cols = input.cols();
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

// ȫ���Ӳ�
MatrixXf CNN::fully_connected(const MatrixXf& input, const MatrixXf& weights, const MatrixXf& bias) {
    //cout << input.cols() << " " << weights.rows() << endl;
    return (input * weights).rowwise() + bias.row(0);
}

MatrixXf softmax(const MatrixXf& input) {
    // ����ÿ��Ԫ�ص�ָ��
    MatrixXf exp_input = input.array().exp();
    //cout << exp_input.rows();

    // ����ÿ�еĺͣ���ת��Ϊ������
    int sum = exp_input.sum();

    // ���й�һ��
    return exp_input / sum;
}

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
}
