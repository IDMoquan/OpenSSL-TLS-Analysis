#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    auto train_data = ReadTrain("Data\\train");
    
    // 假设我们有10行5列的特征矩阵
    int matrix_rows = train_data.first[0].rows();
    int matrix_cols = train_data.first[0].cols();

    // 假设我们有10个不同的网站类别
    int output_dim = 10;

    // 创建CNN分类器(隐藏层大小为32)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.05);

    // 训练模型
     //cnn.train(train_data.first , train_data.second.first, 5);

    // 保存模型
     //cnn.saveModel("test.pt");

    //return 0;

    // 加载模型(如果已有训练好的模型)
     cnn.loadModel("100 epoches.pt");

    // 预测新数据    
     double res = MainPredict("Data\\test", cnn, train_data.second.second);
     cout << "Accuracy:" << fixed << setprecision(2) << res << "%" << endl;

    return 0;
}