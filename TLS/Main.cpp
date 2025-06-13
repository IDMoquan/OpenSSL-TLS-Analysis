#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    //��ȡѵ�������������󡢱�ǩ������Ȩ�ؾ����С
    auto train_data = ReadTrain("Data\\train");
    
    // ��ȡ���������С
    int matrix_rows = train_data.first[0].rows();
    int matrix_cols = train_data.first[0].cols();

    // ������վ������
    int output_dim = 10;

    // ����CNN������(Ȩ�ؾ����С�����ز��С�����������ѧϰ��)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.05);

    // ѵ��ģ��(ѵ�����ݡ���ǩ��ѵ���ִ�)
    //cnn.train(train_data.first , train_data.second.first, 1000);

    // ����ģ��
    //cnn.saveModel("1000 epoches.pt");

    // ����ģ��
     cnn.loadModel("100 epoches.pt");

    // Ԥ��������(����·�����������������С)
     double res = MainPredict("Data\\test", cnn, train_data.second.second);

     //���Ԥ����ȷ��
     if (res >= 90) {
         cout << GREEN;
     }
     else {
         cout << RED;
     }
     cout << endl << "Accuracy:" << fixed << setprecision(2) << res << "%" << endl << WHITE;

    return 0;
}