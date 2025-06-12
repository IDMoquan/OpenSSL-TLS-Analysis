#pragma once
#include <WinSock2.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <filesystem>
#pragma comment(lib, "ws2_32.lib")

using namespace Eigen;
using namespace std;
using namespace filesystem;

// 网络字节序结构体定义
#pragma pack(push, 1)
struct Pcap_Header {
    unsigned int magic;              //0xA1 B2 C3 D4:用来标示文件的开始
    unsigned short major;          //0×02 00:当前文件主要的版本号
    unsigned short minor;          //0×04 00当前文件次要的版本号
    int thiszone;                          //当地的标准时间；全零
    unsigned int sigfigs;             //时间戳的精度；全零
    unsigned int snaplen;           //最大的存储长度
    unsigned int linktype;          //链路类型*
};

struct Pcap_Packet_Header {
    unsigned int timestamp_sec;         //时间戳高位(second)
    unsigned int timestamp_msec;      //时间戳低位(microsecond)
    unsigned int caplen;
    unsigned int len;
};

struct Byte6 {
    char v1, v2, v3, v4, v5, v6;
};

struct Address {
    char a1, a2, a3, a4;
};

struct Ethernet2 {
    Byte6 destination;
    Byte6 source;
    short type;
};

struct Protocol {
    char version;
    char diff_svcs_field;
    short tot_len;
    short identification;
    short flags;
    char time_to_live;
    char protocol;
    short header_checksum;
    Address source_address;
    Address destination_address;
};

struct TC_Protocol {
    short source_port;
    short destination_port;
    int sequence_number;
    int acknowledge_number;
    short flags;
    short window;
    short checksum;
    short urgent_pointer;
};
#pragma pack(pop)

struct Direction {
    Address source;
    Address destination;
};

class Feature {
private:
    unsigned int size;
    bool direction_send;
public:
    Feature(unsigned int& s, const short& port);

    unsigned int GetSize() const;
    bool GetDirection() const;
};

//激活函数
double sigmoid(double x);
double sigmoid_derivative(double x);

class CNN {
private:
    int rows;                // 矩阵行数
    int cols;                 // 矩阵列数 (固定为5)
    int hidden_size;     // 隐藏层神经元数量
    int output_size;     // 输出类别数量(网站数量)

    // 权重矩阵和偏置向量
    MatrixXd weights_input_hidden;    // 输入到隐藏层的权重
    VectorXd bias_hidden;                    // 隐藏层偏置
    MatrixXd weights_hidden_output;  // 隐藏层到输出层的权重
    VectorXd bias_output;                    // 输出层偏置

    // 学习率
    double learning_rate;

public:
    // 构造函数
    CNN(int matrix_rows, int matrix_cols, int hidden_dim, int output_dim, double lr = 0.01, bool reproducible = true);

    // 前向传播
    VectorXd forward(const MatrixXd& input);

    // 反向传播和参数更新
    void backward(const MatrixXd& input, const VectorXd& target);

    // 训练模型
    void train(const vector<MatrixXd>& featureMatrices, const vector<VectorXd>& labels, int epochs);

    // 预测
    int predict(const MatrixXd& featureMatrix);

    // 保存模型
    void saveModel(const string& filename);

    // 加载模型
    void loadModel(const string& filename);
};

//处理数据至特征值矩阵
void LoadData(const vector<Feature>& f, vector<MatrixXd>& features, vector<VectorXd>& labels, const int& label, int len, int max_size, int min_size, int avr_size);

//通过文件夹名称返回对应标签
int Label_Number(string label);

string Website_Name(int label);

enum Website_label {
    bd = 1, bz, csdn, gh, iqy, my, qd, tb, wb, zh
};

pair<vector<MatrixXd>, pair<vector<VectorXd>, int>> ReadTrain(path folderPath);

double MainPredict(path folderPath, CNN& cnn, int max_count);