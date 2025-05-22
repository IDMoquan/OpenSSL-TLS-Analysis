#pragma once
#include <filesystem>
#include "Head.h"
//#define Pcap_File "./Data/2.pcap"
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
            //std::cout << "Directory: " << entry.path() << " " << entry.path().filename() << std::endl;
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
    //cout << name << ":" << max_count << endl;
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
            string Pcap_File = entry.path().string();
            //cout << Pcap_File << endl;
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
            //printPcapFileHeader(ph);
            vector<Feature>features;
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                //pf.read((char*)pph, sizeof(Pcap_Packet_Header));
                //printPcapHeader(pph);
                char* buffer = (char*)malloc(pph->caplen);
                pf.read((char*)buffer, pph->caplen);
                e2 = (Ethernet2*)buffer;
                ptc = (Protocol*)(buffer + sizeof(Ethernet2));
                tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));
                //printPcap(e2, sizeof(Ethernet2));
                //printPcap(ptc, sizeof(Protocol));
                //printPcap(tc_ptc, sizeof(TC_Protocol));
                //printPcap(buffer, pph->caplen);
                //cout << (short)ptc->destination_address.a1 << "." << (short)ptc->destination_address.a2 << "." << (short)ptc->destination_address.a3 << "." << (short)ptc->destination_address.a4 << endl;
                //cout << "端口：" << ntohs((short)tc_ptc->destination_port) << endl;
                //printPcap(&(tc_ptc->destination_port), sizeof(short));
                features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));
                free(buffer);
                //CNN(features);
            }
            LoadData(features, features_matrix, labels, Label_Number(p.filename().string()), max_count);
        }
    }
    int input_rows = max_count;
    int input_cols = 2;
    CNN cnn(input_rows, input_cols, 3, 1, 2, 10);   // 输入大小为 n 行 2 列，卷积核大小3x2，1个卷积核，池化大小2，10个类别
    //训练
    for (int epoch = 0; epoch < 50; epoch++) {
        float loss = 0.0f;
        float learning_rate = 0.0005;
        int size = features_matrix.size();
        cout << "Epoch:" << epoch + 1 << endl;
        //cnn.display_weights();
        for (size_t i = 0; i < size; ++i) {
            MatrixXf output = cnn.forward(features_matrix[i]);
            output = softmax(output);
            //cout << "Predicted output for sample " << i << ": " << endl << output << endl;
            int predicted_class;
            output.row(0).maxCoeff(&predicted_class);
            predicted_class++;
            vector<float>current_label;
            //cout << predicted_class << endl;
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
    for (size_t i = 0; i < features_matrix.size(); ++i) {
        MatrixXf output = cnn.forward(features_matrix[i]);
        output = softmax(output);
        //cout << "Predicted output for sample " << i << ": " << endl << output << endl;
        int predicted_class;
        output.row(0).maxCoeff(&predicted_class);
        predicted_class++;
        vector<float>current_label;
        cout << predicted_class << endl;
        if (predicted_class == labels[i]) {
            res++;
        }
    }
    cout << "Accurrcy: " << fixed << setprecision(2) << 1.0 * res * 100 / features_matrix.size() << "%" << endl;
    return 0;
}