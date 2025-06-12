#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    auto train_data = ReadTrain("Data\\train");
    
    // ����������10��5�е���������
    int matrix_rows = train_data.first[0].rows();
    int matrix_cols = train_data.first[0].cols();

    // ����������10����ͬ����վ���
    int output_dim = 10;

    // ����CNN������(���ز��СΪ32)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.05);

    // ѵ��ģ��
     //cnn.train(train_data.first , train_data.second.first, 5);

    // ����ģ��
     //cnn.saveModel("test.pt");

    //return 0;

    // ����ģ��(�������ѵ���õ�ģ��)
     cnn.loadModel("100 epoches.pt");

    // Ԥ��������    
     double res = MainPredict("Data\\test", cnn, train_data.second.second);
     cout << "Accuracy:" << fixed << setprecision(2) << res << "%" << endl;

    return 0;
}