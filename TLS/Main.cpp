#pragma once
#include <filesystem>
#include "Head.h"
//#define Pcap_File "./Data/2.pcap"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;

int train() {
    path folderPath = "Data\\train";
    if (!exists(folderPath) || !is_directory(folderPath)) {
        cout << "Folder doesn't exist!" << endl;
        return -1;
    }
    vector<path>dir_paths;
    for (const auto& entry : directory_iterator(folderPath)) {
        if (is_directory(entry)) {
            std::cout << "Directory: " << entry.path() << " " << entry.path().filename() << std::endl;
            dir_paths.push_back(entry.path());
        }
    }
    vector<MatrixXf> features_matrix;
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
            string Pcap_File = entry.path().string();
            cout << Pcap_File << endl;
            Pcap_Header* ph = new Pcap_Header;
            Pcap_Packet_Header* pph = new Pcap_Packet_Header;
            Ethernet2* e2 = new Ethernet2;
            Protocol* ptc = new Protocol;
            TC_Protocol* tc_ptc = new TC_Protocol;
            int label;

            ifstream pf;
            pf.open(Pcap_File, ios::in | ios::binary);
            if (!pf) {
                cout << "Open File Error!" << endl;
                return -1;
            }
            pf.read((char*)ph, sizeof(Pcap_Header));
            printPcapFileHeader(ph);
            vector<Feature>features;
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
                //pf.read((char*)pph, sizeof(Pcap_Packet_Header));
                //printPcapHeader(pph);
                void* buffer = (void*)malloc(pph->caplen);
                pf.read((char*)e2, sizeof(Ethernet2));
                pf.read((char*)ptc, sizeof(Protocol));
                pf.read((char*)tc_ptc, sizeof(TC_Protocol));
                pf.read((char*)buffer, pph->caplen - sizeof(Ethernet2) - sizeof(Protocol) - sizeof(TC_Protocol));
                //printPcap(buffer, pph->caplen);
                features.push_back(Feature(pph->caplen, tc_ptc->destination_port));
                free(buffer);
            }
            vector<int> labels;
            LoadData(features, features_matrix, labels, label);

            //CNN(features);
            delete ph;
            delete pph;
            delete e2;
            delete ptc;
            delete tc_ptc;
        }
        cout << features_matrix;
    }
    
}

int main(){
    train();
    return 0;
}