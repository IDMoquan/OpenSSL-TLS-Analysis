#include<iostream>
#include<fstream>
#include<queue>
#include "Head.h"
#define Pcap_File "./Data/2.pcap"
#define MAX_ETH_FRAME 1514

using namespace std;


int main(){
    Pcap_Header* ph = new Pcap_Header;
    Pcap_Packet_Header* pph = new Pcap_Packet_Header;
    Ethernet2* e2 = new Ethernet2;
    Protocol* ptc = new Protocol;
    TC_Protocol* tc_ptc = new TC_Protocol;

    ifstream pf;
    pf.open(Pcap_File, ios::in | ios::binary);
    if(!pf){
        cout << "Open File Error!" << endl;
        return -1;
    }
    pf.read((char*)ph, sizeof(Pcap_Header));
    printPcapFileHeader(ph);
    while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
        //pf.read((char*)pph, sizeof(Pcap_Packet_Header));
        printPcapHeader(pph);
        void* buffer = (void*)malloc(pph->caplen);
        pf.read((char*)e2, sizeof(Ethernet2));
        pf.read((char*)ptc, sizeof(Protocol));
        pf.read((char*)tc_ptc, sizeof(TC_Protocol));
        pf.read((char*)buffer, pph->caplen - sizeof(Ethernet2) - sizeof(Protocol) - sizeof(TC_Protocol));
        printPcap(buffer, pph->caplen);
        Feature feature(pph->caplen, *ptc);
        free(buffer);
    }
    delete ph;
    delete pph;
    delete e2;
    delete ptc;
    delete tc_ptc;
    return 0;
}