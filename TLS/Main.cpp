#pragma once
#include <filesystem>
#include "Head.h"
#define MAX_ETH_FRAME 1514
using namespace std;
using namespace Eigen;
using namespace filesystem;


int main(){
    auto train_data = ReadTrain("Data\\train");     //获取训练数据特征矩阵、标签向量、权重矩阵大小
    
    // 获取特征矩阵大小
    int matrix_rows = train_data.first[0].rows();
    int matrix_cols = train_data.first[0].cols();

    // 设置网站类别个数
    int output_dim = 10;

    // 创建CNN分类器(权重矩阵大小、隐藏层大小、分类个数、学习率)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.05);

    // 训练模型(训练数据、标签、训练轮次)
     cnn.train(train_data.first , train_data.second.first, 100);

    // 保存模型
     cnn.saveModel("100 epoches.pt");

    // 加载模型
     cnn.loadModel("100 epoches.pt");

    // 预测新数据(数据路径、分类器、矩阵大小)
     double res = MainPredict("Data\\test", cnn, train_data.second.second);

     //输出预测正确率
     cout << "Accuracy:" << fixed << setprecision(2) << res << "%" << endl;

    return 0;
}