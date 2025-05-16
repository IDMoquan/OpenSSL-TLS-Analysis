#include<WinSock2.h>
#pragma comment(lib, "ws2_32.lib")

typedef struct Pcap_Header {
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

typedef struct Pcap_Packet_Header {
    unsigned int timestamp_sec;     //时间戳高位(second)
    unsigned int timestamp_msec;    //时间戳低位(microsecond)
    unsigned int caplen;
    unsigned int len;
};

typedef struct Byte6 {
	char v1, v2, v3, v4, v5, v6;
};

typedef struct Address {
	char a1, a2, a3, a4;
};

typedef struct Ethernet2 {
	 Byte6 destination;
	 Byte6 source;
	 short type;
};

typedef struct Protocol {
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

typedef struct TC_Protocol {
	short source_port;
	short destination_port;
	int sequence_number;
	int acknowledge_number;
	short flags;
	short window;
	short checksum;
	short urgent_pointer;
};

void printPcapFileHeader(Pcap_Header *pfh){
	if (pfh==NULL) {
		return;
	}
	printf("=====================\n"
		   "magic:0x%0x\n"
		   "version_major:%u\n"
		   "version_minor:%u\n"
		   "thiszone:%d\n"
		   "sigfigs:%u\n"
		   "snaplen:%u\n"
		   "linktype:%u\n"
		   "=====================\n",
		   pfh->magic,
		   pfh->major,
		   pfh->minor,
		   pfh->thiszone,
		   pfh->sigfigs,
		   pfh->snaplen,
		   pfh->linktype);
}

void printPcapHeader(Pcap_Packet_Header* ph) {
	if (ph == NULL) {
		return;
	}
	printf("=====================\n"
		"ts.timestamp_s:%u\n"
		"ts.timestamp_ms:%u\n"
		"capture_len:%u\n"
		"len:%d\n"
		"=====================\n",
		ph->timestamp_sec,
		ph->timestamp_msec,
		ph->caplen,
		ph->len);
}


void printPcap(void* data, size_t size) {
	unsigned  short iPos = 0;
	//int * p = (int *)data;
	//unsigned short* p = (unsigned short *)data;
	if (data == NULL) {
		return;
	}

	printf("\n==data:0x%x,len:%lu=========", data, size);

	for (iPos = 0; iPos < size / sizeof(unsigned short); iPos++) {
		//printf(" %x ",(int)( * (p+iPos) ));
		//unsigned short a = ntohs(p[iPos]);

		unsigned short a = ntohs(*((unsigned short*)data + iPos));
		if (iPos % 8 == 0) printf("\n");
		if (iPos % 4 == 0) printf(" ");

		printf("%04x", a);


	}
	/*
	 for (iPos=0; iPos <= size/sizeof(int); iPos++) {
		//printf(" %x ",(int)( * (p+iPos) ));
		int a = ntohl(p[iPos]);

		//int a = ntohl( *((int *)data + iPos ) );
		if (iPos %4==0) printf("\n");

		printf("%08x ",a);


	}
	 */
	printf("\n============\n");
}

typedef struct Direction{
	Address source;
	Address destination;
}

Class Feature{
private:
	int size;
	Direction direction;
public:
	Feature(int s, Direction d) : size(s), direction(d){}
	
};