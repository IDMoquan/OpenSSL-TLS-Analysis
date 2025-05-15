#include<iostream>
#include<fstream>
#include<queue>
#include "Head.h"
#define Pcap_File "./Data/test.pcap"

using namespace std;


int main(){
    Pcap_Header *ph = new Pcap_Header;
    Pcap_Packet_Header *pph = new Pcap_Packet_Header;

    ifstream pf;
    pf.open(Pcap_File, ios::in | ios::binary);
    if(!pf){
        cout << "Open File Error!" << endl;
        return -1;
    }
    pf.read((char*)ph, sizeof(Pcap_Header));
    printPcapFileHeader(ph);
    //while (pf.read((char*)pph, sizeof(Pcap_Packet_Header));) {
        pf.read((char*)pph, sizeof(Pcap_Packet_Header));
        printPcapHeader(pph);

    //}
    delete ph;
    delete pph;
    return 0;
}