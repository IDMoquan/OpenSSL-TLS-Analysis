#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    vector<MatrixXf> features_matrix;
    vector<int> labels;
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
            Ethernet2* e2;
            Protocol* ptc;
            TC_Protocol* tc_ptc;

            ifstream pf;
            pf.open(Pcap_File, ios::in | ios::binary);
            if (!pf) {
                cout << "Open File Error!" << endl;
                return -1;
            }
            pf.read((char*)ph, sizeof(Pcap_Header));
            vector<Feature>features;
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                char* buffer = (char*)malloc(pph->caplen);
                pf.read((char*)buffer, pph->caplen);
                e2 = (Ethernet2*)buffer;
                ptc = (Protocol*)(buffer + sizeof(Ethernet2));
                tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));
                features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));
                free(buffer);
            }
            LoadData(features, features_matrix, labels, Label_Number(p.filename().string()), max_count);
        }
    }
    int input_rows = max_count;
    int input_cols = 2;
    CNN cnn(input_rows, input_cols, 2, 3, 2, 10);   // 输入大小为 n 行 2 列，卷积核大小3x2，1个卷积核，池化大小2，10个类别
    //训练
    for (int epoch = 0; epoch < 45; epoch++) {
        float loss = 0.0f;
        float learning_rate = 0.001;
        int size = features_matrix.size();
        cout << "Epoch:" << epoch + 1 << endl;
        for (size_t i = 0; i < size; ++i) {
            MatrixXf output = cnn.forward(features_matrix[i]);
            output = softmax(output);
            int predicted_class;
            output.row(0).maxCoeff(&predicted_class);
            predicted_class++;
            vector<float>current_label;
            int right_class = labels[i];
            for (int i = 0; i < 10; i++) {
                if (i + 1 == right_class) {
                    current_label.push_back(1);
                }
                else {
                    current_label.push_back(0);
                }
            }
            auto delta_loss = cnn.cross_entropy(output, current_label);
            loss += delta_loss.second;
            cnn.backward(delta_loss.second, delta_loss.first, learning_rate);
        }
        cout << "Average loss:" << loss / size << endl;
    }
    cout << "Train finished!" << endl;
    int res = 0;
    vector<MatrixXf> test_features_matrix;
    vector<int> test_labels;
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
            Ethernet2* e2;
            Protocol* ptc;
            TC_Protocol* tc_ptc;

            ifstream pf;
            pf.open(Pcap_File, ios::in | ios::binary);
            if (!pf) {
                cout << "Open File Error!" << endl;
                return -1;
            }
            pf.read((char*)ph, sizeof(Pcap_Header));
            vector<Feature>features;
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                char* buffer = (char*)malloc(pph->caplen);
                pf.read((char*)buffer, pph->caplen);
                e2 = (Ethernet2*)buffer;
                ptc = (Protocol*)(buffer + sizeof(Ethernet2));
                tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));
                features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));
                free(buffer);
            }
            LoadData(features, test_features_matrix, test_labels, Label_Number(p.filename().string()), max_count);
        }
    }
    for (size_t i = 0; i < test_features_matrix.size(); ++i) {
        MatrixXf output = cnn.forward(test_features_matrix[i]);
        output = softmax(output);
        cout << "Predicted output for sample " << i << ": " << endl << output << endl;
        int predicted_class;
        output.row(0).maxCoeff(&predicted_class);
        predicted_class++;
        vector<float>current_label;
        cout << predicted_class << endl;
        if (predicted_class == test_labels[i]) {
            res++;
        }
    }
    cout << "Accurrcy: " << fixed << setprecision(2) << 1.0 * res * 100 / test_features_matrix.size() << "%" << endl;
    return 0;
}