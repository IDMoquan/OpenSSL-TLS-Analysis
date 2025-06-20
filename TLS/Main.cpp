﻿#include "Head.h"
using namespace std;


int main(){
    //获取训练数据特征矩阵、标签向量、权重矩阵大小
    auto train_data = ReadTrain("Data\\train");
	stringstream pt_filename;
    
    //获取特征矩阵大小
	int matrix_rows = static_cast<int>(train_data.first[0].rows());
    int matrix_cols = static_cast<int>(train_data.first[0].cols());

    //设置网站类别个数
    int output_dim = 10;
	
	//设置训练轮数
	int epochs = 100;
	pt_filename << ".\\Models\\" << epochs << " epoches.pt";	

    //创建CNN分类器(权重矩阵大小、隐藏层大小、分类个数、学习率)
    CNN cnn(matrix_rows, matrix_cols, 32, output_dim, 0.05);

    //训练模型(训练数据、标签、训练轮次)
    cnn.train(train_data.first , train_data.second, epochs);

    //保存模型
    cnn.saveModel(pt_filename.str());
	return 0;

    //加载模型
     cnn.loadModel(pt_filename.str());

    //预测新数据(数据路径、分类器、矩阵大小)
     double res = MainPredict("Data\\test", cnn, cnn.GetRows());

     //输出预测正确率
     if (res >= 90) {
         cout << GREEN;
     }
     else {
         cout << RED;
     }
     cout << endl << "Accuracy:" << fixed << setprecision(2) << res << "%" << endl << WHITE;

    return 0;
}