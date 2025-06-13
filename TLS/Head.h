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
#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#pragma comment(lib, "ws2_32.lib")

using namespace Eigen;
using namespace std;
using namespace filesystem;

// 网络字节序结构体定义
#pragma pack(push, 1)

//Pcap文件头
struct Pcap_Header {
    unsigned int magic;                     //0xA1 B2 C3 D4:用来标示文件的开始
    unsigned short major;                   //0×02 00:当前文件主要的版本号
    unsigned short minor;                   //0×04 00当前文件次要的版本号
    int thiszone;                           //当地的标准时间；全零
    unsigned int sigfigs;                   //时间戳的精度；全零
    unsigned int snaplen;                   //最大的存储长度
    unsigned int linktype;                  //链路类型*
};



//Pcap包头
struct Pcap_Packet_Header {
    unsigned int timestamp_sec;             //时间戳高位(second)
    unsigned int timestamp_msec;            //时间戳低位(microsecond)
    unsigned int caplen;                    //数据帧长度
    unsigned int len;                       //离线数据长度
};



//6字节数据
struct Byte6 {
    char v1, v2, v3, v4, v5, v6;        
};

//IP地址
struct Address {
    char a1, a2, a3, a4;                  
};

//Pcap包Ethernet部分
struct Ethernet2 {
    Byte6 destination;                      //目标MAC地址
    Byte6 source;                           //源MAC地址
    short type;                             //上一层协议
};



//Pcap包Protocol部分
struct Protocol {
    char version;                           //版本、首部长度
    char diff_svcs_field;                   //区分服务
    short tot_len;                          //首部长度
    short identification;                   //标识
    short flags;                            //标志、片偏移
    char time_to_live;                      //生存时间
    char protocol;                          //协议
    short header_checksum;                  //首部校验和
    Address source_address;                 //源IP地址
    Address destination_address;            //目标IP地址
};

//Pcap包TC_Protol部分
struct TC_Protocol {
    short source_port;                      //源端口
    short destination_port;                 //目标端口
    int sequence_number;                    //序号
    int acknowledge_number;                 //确认号
    short flags;                            //偏移字段、保留字段、紧急比特URG、确认比特ACK、推送比特PSH、复位比特RST、同步比特SYN、终止比特FIN
    short window;                           //窗口字段
    short checksum;                         //首部校验和
    short urgent_pointer;                   //紧急指针字段
};
#pragma pack(pop)




//方向
struct Direction {
    Address source;                         //源IP
    Address destination;                    //目标IP
};

//特征
class Feature {
private:
    unsigned int size;                              //大小
    bool direction_send;                            //方向
public:
    Feature(unsigned int& s, const short& port);    //构造函数
        
    unsigned int GetSize() const;                   //获取大小
    bool GetDirection() const;                      //获取方向
};

//激活函数
double sigmoid(double x);
double sigmoid_derivative(double x);

class CNN {
private:
    int rows;                                       //矩阵行数
    int cols;                                       //矩阵列数 (固定为5)
    int hidden_size;                                //隐藏层神经元数量
    int output_size;                                //输出类别数量(网站数量)

    //权重矩阵和偏置向量
    MatrixXd weights_input_hidden;                  //输入到隐藏层的权重
    VectorXd bias_hidden;                           //隐藏层偏置
    MatrixXd weights_hidden_output;                 //隐藏层到输出层的权重
    VectorXd bias_output;                           //输出层偏置

    //学习率
    double learning_rate;

public:
    //构造函数
    CNN(int matrix_rows, int matrix_cols, int hidden_dim, int output_dim, double lr = 0.01, bool reproducible = true);

    //前向传播
    VectorXd forward(const MatrixXd& input);

    //反向传播和参数更新
    void backward(const MatrixXd& input, const VectorXd& target);

    //训练模型
    void train(const vector<MatrixXd>& featureMatrices, const vector<VectorXd>& labels, int epochs);

    //预测
    int predict(const MatrixXd& featureMatrix);

    //保存模型
    void saveModel(const string& filename);

    //加载模型
    void loadModel(const string& filename);
};

//处理数据至特征值矩阵
void LoadData(const vector<Feature>& f, vector<MatrixXd>& features, vector<VectorXd>& labels, const int& label, int len, int max_size, int min_size, int avr_size);

//通过文件夹名称返回对应标签
int Label_Number(string label);

//根据标签获取网站
string Website_Name(int label);

//网站标签枚举
enum Website_label {
    bd = 1, bz, csdn, gh, iqy, my, qd, tb, wb, zh
};

//读取训练数据主函数
pair<vector<MatrixXd>, pair<vector<VectorXd>, int>> ReadTrain(path folderPath);

//测试函数
double MainPredict(path folderPath, CNN& cnn, int max_count);