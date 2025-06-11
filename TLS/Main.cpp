#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    vector<MatrixXd> features_matrix;
    vector<VectorXd> labels;
    int max_count = -1;
    string name;
    path folderPath = "Data\\train";
    if (!exists(folderPath) || !is_directory(folderPath)) {
        cout << "Folder doesn't exist!" << endl;
        return -1;
    }
    vector<path>dir_paths;
    for (const auto& entry : directory_iterator(folderPath)) {
        if (is_directory(entry)) {
            dir_paths.push_back(entry.path());
        }
    }
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
            string Pcap_File = entry.path().string();
            Pcap_Header* ph = new Pcap_Header;
            Pcap_Packet_Header* pph = new Pcap_Packet_Header;

            ifstream pf;
            pf.open(Pcap_File, ios::in | ios::binary);
            if (!pf) {
                cout << "Open File Error!" << endl;
                return -1;
            }
            pf.read((char*)ph, sizeof(Pcap_Header));
            int count = 0;
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                char* buffer = (char*)malloc(pph->caplen);
                pf.read((char*)buffer, pph->caplen);
                count++;
                free(buffer);
            }
            if (count > max_count) {
                max_count = count;
                name = Pcap_File;
            }
        }
    }
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
            string Pcap_File = entry.path().string();
            Pcap_Header* ph = new Pcap_Header;
            Pcap_Packet_Header* pph = new Pcap_Packet_Header;
            TC_Protocol* tc_ptc;

            ifstream pf;
            pf.open(Pcap_File, ios::in | ios::binary);
            if (!pf) {
                cout << "Open File Error!" << endl;
                return -1;
            }
            pf.read((char*)ph, sizeof(Pcap_Header));
            vector<Feature>features;
            int count = 0;
            int max_len = 0;
            int min_len = INT_MAX;
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                char* buffer = (char*)malloc(pph->caplen);
                pf.read((char*)buffer, pph->caplen);
                tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));
                features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));
                free(buffer);
                count += pph->caplen;
                //cout << "pph len: " << pph->caplen << endl;
                if (pph->caplen > max_len) {
                    max_len = pph->caplen;
                }
                if (pph->caplen < min_len) {
                    min_len = pph->caplen;
                }
            }
            //cout << "Max len:" << max_len << endl;
            LoadData(features, features_matrix, labels, Label_Number(p.filename().string()), max_count, max_len, min_len, count / features.size());
            //cout << features_matrix[features_matrix.size() - 1] << endl;
        }
    }
    
    // 假设我们有10行5列的特征矩阵
    int matrix_rows = features_matrix[0].rows();
    int matrix_cols = features_matrix[0].cols();

    // 假设我们有10个不同的网站类别
    int output_dim = 10;

    // 创建CNN分类器(隐藏层大小为32)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.02);

    // 训练模型
     //cnn.train(features_matrix, labels, 100);

    // 保存模型
     //cnn.saveModel("test.pt");

    return 0;

    // 加载模型(如果已有训练好的模型)
     cnn.loadModel("test.pt");

    // 预测新数据
     vector<MatrixXd> test_features_matrix;
     folderPath = "Data\\test";
     if (!exists(folderPath) || !is_directory(folderPath)) {
         cout << "Folder doesn't exist!" << endl;
         return -1;
     }
     dir_paths.clear();
     for (const auto& entry : directory_iterator(folderPath)) {
         if (is_directory(entry)) {
             dir_paths.push_back(entry.path());
         }
     }
     for (const auto& p : dir_paths) {
         for (const auto& entry : directory_iterator(p)) {
             string Pcap_File = entry.path().string();
             Pcap_Header* ph = new Pcap_Header;
             Pcap_Packet_Header* pph = new Pcap_Packet_Header;

             ifstream pf;
             pf.open(Pcap_File, ios::in | ios::binary);
             if (!pf) {
                 cout << "Open File Error!" << endl;
                 return -1;
             }
             pf.read((char*)ph, sizeof(Pcap_Header));
             int count = 0;
             while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                 char* buffer = (char*)malloc(pph->caplen);
                 pf.read((char*)buffer, pph->caplen);
                 count++;
                 free(buffer);
             }
             if (count > max_count) {
                 max_count = count;
                 name = Pcap_File;
             }
         }
     }
     int correct = 0;
     int wrong = 0;
     for (const auto& p : dir_paths) {
         for (const auto& entry : directory_iterator(p)) {
             string Pcap_File = entry.path().string();
             Pcap_Header* ph = new Pcap_Header;
             Pcap_Packet_Header* pph = new Pcap_Packet_Header;
             TC_Protocol* tc_ptc;

             ifstream pf;
             pf.open(Pcap_File, ios::in | ios::binary);
             if (!pf) {
                 cout << "Open File Error!" << endl;
                 return -1;
             }
             pf.read((char*)ph, sizeof(Pcap_Header));
             vector<Feature>features;
             int count = 0;
             int max_len = 0;
             int min_len = INT_MAX;
             while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                 char* buffer = (char*)malloc(pph->caplen);
                 pf.read((char*)buffer, pph->caplen);
                 tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));
                 features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));
                 free(buffer);
                 count += pph->caplen;
                 //cout << "pph len: " << pph->caplen << endl;
                 if (pph->caplen > max_len) {
                     max_len = pph->caplen;
                 }
                 if (pph->caplen < min_len) {
                     min_len = pph->caplen;
                 }
             }
             //cout << "Max len:" << max_len << endl;
             LoadData(features, test_features_matrix, labels, Label_Number(p.filename().string()), max_count, max_len, min_len, count / features.size());
             //cout << features_matrix[features_matrix.size() - 1] << endl;
             int predicted_class = cnn.predict(test_features_matrix[test_features_matrix.size() - 1]) + 1;
             if (predicted_class == Label_Number(p.filename().string())) {
                 correct++;
             }
             else {
                 wrong++;
             }
         }
     }
     cout << "Accuracy:" << fixed << setprecision(2) << 1.0 * (correct) / (correct + wrong) * 100 << "%" << endl;

    return 0;
}