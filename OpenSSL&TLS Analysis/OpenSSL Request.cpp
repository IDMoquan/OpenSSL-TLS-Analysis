#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <openssl/ssl.h>
#include <openssl/err.h>
#include "Head.h"
#pragma comment(lib, "libssl.lib")
#pragma comment(lib, "libcrypto.lib")
#pragma comment(lib, "ws2_32.lib")
using namespace std;

int main() {
    // 初始化 SSL 库
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    const SSL_METHOD* meth = SSLv23_client_method();
    SSL_CTX* ctx = SSL_CTX_new(meth);
    if (ctx == NULL) {
        ERR_print_errors_fp(stderr);
        cout << "SSL_CTX_new error !" << endl;
        return -1;
    }

    // 初始化 Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return -1;
    }

    // 创建 socket
    SOCKET client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (client == INVALID_SOCKET) {
        cout << "socket error !" << endl;
        return -1;
    }

    // 设置目标服务器
    string host = "www.zhihu.com";
    cout << host << endl;
    unsigned short port = 443;
    hostent* ip = gethostbyname(host.c_str());  

    sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    sin.sin_addr = *(in_addr*)ip->h_addr_list[0];

    struct in_addr addr;
    memcpy(&addr, ip->h_addr_list[0], sizeof(struct in_addr));
    std::cout << host << "'s IP: " << inet_ntoa(addr) << std::endl;
    //return 0;

    // 连接到服务器
    if (connect(client, (sockaddr*)&sin, sizeof(sin)) == SOCKET_ERROR) {
        cout << "connect error 1" << endl;
        return -1;
    }

    // 创建 SSL 对象并绑定到 socket
    SSL* ssl = SSL_new(ctx);
    if (ssl == NULL) {
        cout << "SSL NEW error" << endl;
        return -1;
    }
	SSL_set_fd(ssl, static_cast<int>(client));

    // 建立 SSL 连接
    int ret = SSL_connect(ssl);
    if (ret == -1) {
        cout << "SSL ACCEPT error " << endl;
        return -1;
    }

    // 构造 HTTP 请求
    stringstream stream;
    stream << "GET / HTTP/1.1\r\n";
    stream << "Host: " << host << "\r\n";
    stream << "Accept: */*\r\n";
    stream << "Connection: Close\r\n";
    stream << "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134\r\n";
    stream << "\r\n";

    string s = stream.str();
    const char* sendData = s.c_str();

    // 发送请求
	ret = SSL_write(ssl, sendData, static_cast<int>(strlen(sendData)));
    if (ret == -1) {
        cout << "SSL write error !" << endl;
        return -1;
    }

    // 接收响应
    char* rec = new char[1024 * 1024];
    int start = 0;
    while ((ret = SSL_read(ssl, rec + start, 1024)) > 0) {
        start += ret;
    }
    rec[start] = 0;
    cout << Utf8ToGbk(rec) << endl;

    // 清理资源
    SSL_shutdown(ssl);
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    closesocket(client);
    WSACleanup();

    return 0;
}