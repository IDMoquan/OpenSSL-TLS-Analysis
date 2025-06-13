#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    auto train_data = ReadTrain("Data\\train");     //��ȡѵ�������������󡢱�ǩ������Ȩ�ؾ����С
    
    // ��ȡ���������С
    int matrix_rows = train_data.first[0].rows();
    int matrix_cols = train_data.first[0].cols();

    // ������վ������
    int output_dim = 10;

    // ����CNN������(Ȩ�ؾ����С�����ز��С�����������ѧϰ��)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.05);

    // ѵ��ģ��(ѵ�����ݡ���ǩ��ѵ���ִ�)
     cnn.train(train_data.first , train_data.second.first, 100);

    // ����ģ��
     cnn.saveModel("100 epoches.pt");

    // ����ģ��
     cnn.loadModel("100 epoches.pt");

    // Ԥ��������(����·�����������������С)
     double res = MainPredict("Data\\test", cnn, train_data.second.second);

     //���Ԥ����ȷ��
     cout << "Accuracy:" << fixed << setprecision(2) << res << "%" << endl;

    return 0;
}