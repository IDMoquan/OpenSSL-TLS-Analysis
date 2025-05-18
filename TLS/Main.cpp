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
            //std::cout << "Directory: " << entry.path() << " " << entry.path().filename() << std::endl;
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
                //cout << "¶Ë¿Ú£º" << ntohs((short)tc_ptc->destination_port) << endl;
                //printPcap(&(tc_ptc->destination_port), sizeof(short));
                features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));
                free(buffer);
                //CNN(features);
            }
            vector<int> labels;
            LoadData(features, features_matrix, labels, Label_Number(p.filename().string()));
        }
        //cout << features_matrix;
    }
    
}

int main(){
    train();
    return 0;
}