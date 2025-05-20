#pragma once
#ifndef _HEADER_H_
#define _HEADER_H_
#include <WinSock2.h>
#include <Eigen/Eigen>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#pragma comment(lib, "ws2_32.lib")
using namespace std;
using namespace Eigen;

struct Pcap_Header {
    unsigned int magic = 0;             //0xA1 B2 C3 D4:用来标示文件的开始
    unsigned short major = 0 ;           //0×02 00:当前文件主要的版本号
    unsigned short minor = 0;           //0×04 00当前文件次要的版本号
    int thiszone = 0;                   //当地的标准时间；全零
    unsigned int sigfigs = 0;           //时间戳的精度；全零
    unsigned int snaplen = 0;           //最大的存储长度
    unsigned int linktype = 0;          //链路类型*
};

/*链路常用类型：
0           BSD loopback devices, except for later OpenBSD
1           Ethernet, and Linux loopback devices
6           802.5 Token Ring
7           ARCnet
8           SLIP
9           PPP
10          FDDI
100         LLC/SNAP-encapsulated ATM
101         “raw IP”, with no link
102         BSD/OS SLIP
103         BSD/OS PPP
104         Cisco HDLC
105         802.11
108         later OpenBSD loopback devices (with the AF_value in network byte order)
113         special Linux “cooked” capture
114         LocalTalk
*/

struct Pcap_Packet_Header {
    unsigned int timestamp_sec;     //时间戳高位(second)
    unsigned int timestamp_msec;    //时间戳低位(microsecond)
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


struct Direction {
	Address source;
	Address destination;
};

class Feature{
private:
	unsigned int size;
	bool direction_send;
public:
	Feature(unsigned int& s, const short& port){
		size = s;
		//通过端口判断传输方向
		if (port == 443) {
			direction_send = 1;
		}
		else {
			direction_send = 0;
		}
	}
	unsigned int GetSize() {
		return size;
	}
	bool GetDirection() {
		return direction_send;
	}
};

class CNN {
public:
	CNN(int input_rows, int input_cols, int filter_size, int num_filters, int pool_size, int num_classes);
    // 前向传播
	MatrixXf forward(const MatrixXf& input);
	pair<MatrixXf, float> cross_entropy(const MatrixXf& prob, const vector<float>&labels);

private:
	int input_rows;
	int input_cols;
	int filter_size;
	int num_filters;
	int pool_size;
	int num_classes;
	vector<MatrixXf> weights;
	vector<MatrixXf> biases;
	MatrixXf fc_weights;
	MatrixXf fc_bias;


    // 卷积操作
    void conv2d(const MatrixXf& input, const MatrixXf& kernel, MatrixXf& output);
    // ReLU激活函数
    MatrixXf relu(const MatrixXf& input);
    // 最大池化操作
    void max_pooling(const MatrixXf& input, int pool_size, MatrixXf& output);
    // 全连接层
    MatrixXf fully_connected(const MatrixXf& input, const MatrixXf& weights, const MatrixXf& bias);
};

void printPcapFileHeader(Pcap_Header* pfh);
void printPcapHeader(Pcap_Packet_Header* ph);
void printPcap(void* data, size_t size);
void LoadData(vector<Feature>& f, vector<MatrixXf>& features, vector<int>& labels, const int& label, int len);
int Label_Number(string label);
MatrixXf softmax(const MatrixXf& input);

enum Website_label {
	bd = 1, bz, tb, wb, 
};
#endif