#pragma once
#include "Head.h"

//Feature类构造函数
Feature::Feature(unsigned int& s, const short& port) : size(s) {
    //通过端口判断传输方向
    if (port == 443) {
        direction_send = 1;
    }
    else {
        direction_send = 0;
    }
}

//获取大小
unsigned int Feature::GetSize() const {
    return size;
}

//获取方向
bool Feature::GetDirection() const {
    return direction_send;
}

//激活函数1
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//激活函数2
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

//CNN类构造函数
CNN::CNN(int matrix_rows, int matrix_cols, int hidden_dim, int output_dim, double lr, bool reproducible)
    : rows(matrix_rows), cols(matrix_cols), hidden_size(hidden_dim), output_size(output_dim), learning_rate(lr) {

    //初始化随机数生成器
    mt19937 gen;

    //固定种子，确保实验可重复
    if (reproducible) {
        unsigned int seed = 42;
        gen = mt19937(seed);
    }
    //不可预测的随机数生成
    else {
        //种子生成
        random_device rd;           
        vector<unsigned int> seeds(mt19937::state_size);
        generate_n(seeds.data(), seeds.size(), ref(rd));
        seed_seq seed_seq(seeds.begin(), seeds.end());
        //生成随机数
        gen = mt19937(seed_seq);
    }

    //使用Xavier初始化
    double fan_in = rows * cols;                        //输入神经元数量
    double fan_out = hidden_size;                       //输出神经元大小
    double stddev = sqrt(2.0 / (fan_in + fan_out));     //计算标准差
    normal_distribution<> dist(0.0, stddev);            //定义正态分布

    //输入到隐藏层的权重: hidden_size x (rows * cols)
    weights_input_hidden = MatrixXd::Zero(hidden_size, rows * cols);
    bias_hidden = VectorXd::Zero(hidden_size);

    //隐藏层到输出层的权重: output_size x hidden_size
    weights_hidden_output = MatrixXd::Zero(output_size, hidden_size);
    bias_output = VectorXd::Zero(output_size);

    //随机初始化权重
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            weights_input_hidden(i, j) = dist(gen);     //输入-隐藏层权重从正态分布中采样
        }
        bias_hidden(i) = dist(gen);                     //偏置相同操作
    }

    //同上操作
    fan_in = hidden_size;
    fan_out = output_size;
    stddev = sqrt(2.0 / (fan_in + fan_out));
    normal_distribution<> dist2(0.0, stddev);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            weights_hidden_output(i, j) = dist2(gen);
        }
        bias_output(i) = dist2(gen);
    }
}

//前向传播
VectorXd CNN::forward(const MatrixXd& input) {
    VectorXd input_vec = Map<const VectorXd>(input.data(), rows * cols);    //将矩阵转换为向量
    VectorXd hidden = weights_input_hidden * input_vec + bias_hidden;       //输入层到隐藏层

	//应用激活函数
    for (int i = 0; i < hidden.size(); ++i) {
        hidden(i) = sigmoid(hidden(i));
    }

    VectorXd output = weights_hidden_output * hidden + bias_output;         //隐藏层到输出层

	//应用激活函数
    for (int i = 0; i < output.size(); ++i) {
        output(i) = sigmoid(output(i));
    }

	return output;      //返回输出向量
}

//反向传播
void CNN::backward(const MatrixXd& input, const VectorXd& target) {
    VectorXd input_vec = Map<const VectorXd>(input.data(), rows * cols);                    //将矩阵转换为向量

	//前向传播计算隐藏层和输出层
    VectorXd hidden_input = weights_input_hidden * input_vec + bias_hidden;
    VectorXd hidden_output = hidden_input.unaryExpr([](double x) { 
         return sigmoid(x); 
    });

    VectorXd final_input = weights_hidden_output * hidden_output + bias_output;
    VectorXd final_output = final_input.unaryExpr([](double x) {
        return sigmoid(x); 
    });

    //计算输出层误差
    VectorXd output_error = final_output - target;
    VectorXd output_delta = output_error.cwiseProduct(
        final_input.unaryExpr([](double x) { 
            return sigmoid_derivative(x); 
        })
    );

    //计算隐藏层误差
    VectorXd hidden_error = weights_hidden_output.transpose() * output_delta;
    VectorXd hidden_delta = hidden_error.cwiseProduct(
        hidden_input.unaryExpr([](double x) { 
            return sigmoid_derivative(x); 
        }
    ));

    //更新权重和偏置
    weights_hidden_output -= learning_rate * output_delta * hidden_output.transpose();
    bias_output -= learning_rate * output_delta;

    weights_input_hidden -= learning_rate * hidden_delta * input_vec.transpose();
    bias_hidden -= learning_rate * hidden_delta;
}

//训练模型
void CNN::train(const vector<MatrixXd>& featureMatrices, const vector<VectorXd>& labels, int epochs) {
	int num_samples = featureMatrices.size();               //初始化样本数量

	//训练epochs轮
    for (int epoch = 0; epoch < epochs; epoch++) {
		double total_loss = 0.0;                            //初始化总损失

		//遍历每个样本
        for (int i = 0; i < num_samples; ++i) {
            //确保矩阵大小正确
            if (featureMatrices[i].rows() != rows || featureMatrices[i].cols() != cols) {
                cerr << "Error: Matrix " << i << " has incorrect dimensions!" << endl;
                continue;
            }

            //前向传播
            VectorXd prediction = forward(featureMatrices[i]);

            //计算损失
            VectorXd error = prediction - labels[i];
            double loss = error.squaredNorm() / 2.0;
            total_loss += loss;

            //反向传播
            backward(featureMatrices[i], labels[i]);
        }

        //输出平均损失
        double avg_loss = total_loss / num_samples;
        cout << "Epoch " << epoch + 1 << "/" << epochs << ", Average Loss: " << avg_loss << endl;
    }
}

// 预测
int CNN::predict(const MatrixXd& featureMatrix) {
    // 确保矩阵大小正确
    if (featureMatrix.rows() != rows || featureMatrix.cols() != cols) {
        cerr << "Error: Input matrix has incorrect dimensions!" << endl;
        return -1;
    }

    // 前向传播
    VectorXd output = forward(featureMatrix);

    // 找出最大值的索引
    int max_index = 0;
    double max_value = output(0);

    for (int i = 1; i < output.size(); ++i) {
        if (output(i) > max_value) {
            max_value = output(i);
            max_index = i;
        }
    }

    return max_index;
}

// 保存模型
void CNN::saveModel(const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        // 保存网络结构
        file << rows << " " << cols << " " << hidden_size << " " << output_size << endl;

        // 保存权重和偏置
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < rows * cols; ++j) {
                file << weights_input_hidden(i, j) << " ";
            }
            file << bias_hidden(i) << endl;
        }

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                file << weights_hidden_output(i, j) << " ";
            }
            file << bias_output(i) << endl;
        }

        file.close();
        cout << "Model saved to " << filename << endl;
    }
    else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

// 加载模型
void CNN::loadModel(const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        // 加载网络结构
        file >> rows >> cols >> hidden_size >> output_size;

        // 调整矩阵大小
        weights_input_hidden.resize(hidden_size, rows * cols);
        bias_hidden.resize(hidden_size);
        weights_hidden_output.resize(output_size, hidden_size);
        bias_output.resize(output_size);

        // 加载权重和偏置
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < rows * cols; ++j) {
                file >> weights_input_hidden(i, j);
            }
            file >> bias_hidden(i);
        }

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                file >> weights_hidden_output(i, j);
            }
            file >> bias_output(i);
        }

        file.close();
        cout << "Model loaded from " << filename << endl;
    }
    else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

void LoadData(const vector<Feature>& f, vector<MatrixXd>& features, vector<VectorXd>& labels, const int& label, int len, int max_size, int min_size, int avr_size) {
    int size = static_cast<int>(f.size());
    MatrixXd feature(len, 5);

    // 计算特征最大值用于归一化
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        max_val = max(max_val, static_cast<float>(f[i].GetSize()));
    }
    max_val = max(max_val, static_cast<float>(max_size));

    for (int i = 0; i < size; i++) {
        feature(i, 0) = static_cast<float>(f[i].GetSize()) / max_val;    // 归一化
        feature(i, 1) = f[i].GetDirection();
        feature(i, 2) = static_cast<float>(max_size) / max_val;         // 归一化
        feature(i, 3) = static_cast<float>(min_size) / max_val;         // 归一化
        feature(i, 4) = static_cast<float>(avr_size) / max_val;          // 归一化
    }
    for (int i = size; i < len; i++) {
        feature(i, 0) = 0.0f;
        feature(i, 1) = 0.0f;
        feature(i, 2) = 0.0f;
        feature(i, 3) = 0.0f;
        feature(i, 4) = 0.0f;
    }
    features.push_back(feature);
    VectorXd temp_label(10);
    for (int i = 0; i < 10; i++) {
        if (i + 1 == label) {
            temp_label[i] = 1;
        }
        else {
            temp_label[i] = 0;
        }
    }
    labels.push_back(temp_label);
}

//通过文件夹名称返回对应标签
int Label_Number(string label) {
    if (label == "bd")  return 1;       //百度
    if (label == "bz")  return 2;       //B站
    if (label == "csdn")  return 3;     //Csdn
    if (label == "gh")  return 4;       //Github
    if (label == "iqy")  return 5;      //爱奇艺
    if (label == "my")  return 6;       //猫眼
    if (label == "qd")  return 7;       //起点
    if (label == "tb")  return 8;       //淘宝
    if (label == "wb")  return 9;       //微博
    if (label == "zh")  return 10;      //知乎
    else return -1;
}

string Website_Name(int label) {
    if (label == 1)  return "www.baidu.com";        //百度
    if (label == 2)  return "www.bilibili.com";     //B站
    if (label == 3)  return "www.csdn.com";         //Csdn
    if (label == 4)  return "www.github.com";       //Github
    if (label == 5)  return "www.iqiyi.com";        //爱奇艺
    if (label == 6)  return "www.maoyan.com";       //猫眼
    if (label == 7)  return "www.qidian.com";       //起点
    if (label == 8)  return "www.taobao.com";       //淘宝
    if (label == 9)  return "www.weibo.com";        //微博
    if (label == 10)  return "www.zhihu.com";       //知乎
    else return "null";
}

//读取训练数据主函数
pair<vector<MatrixXd>, pair<vector<VectorXd>, int>> ReadTrain(path folderPath) {
    //初始化变量
    vector<MatrixXd> features_matrix;                           //特征值矩阵动态数组
    vector<VectorXd> labels;                                    //标签记录数组
    vector<path>dir_paths;                                      //待搜索文件夹路径
    int max_count = -1;                                         //包数量最大值
    string name;                                                //数量最大的包名称

    //检查路径是否存在
    if (!exists(folderPath) || !is_directory(folderPath)) {
        throw "Invalid Path!";
    }

    //遍历第一级目录将文件夹放入待搜索数组
    for (const auto& entry : directory_iterator(folderPath)) {
        //判断是文件夹
        if (is_directory(entry)) {
            dir_paths.push_back(entry.path());
        }
    }
    
    //按文件夹搜索pcap文件以获取最大包数量
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
            //初始化变量
            string Pcap_File = entry.path().string();                   //文件名
            Pcap_Header* ph = new Pcap_Header;                          //Pcap头指针
            Pcap_Packet_Header* pph = new Pcap_Packet_Header;           //Pcap包头指针
            ifstream pf;                                                //文件
            int count = 0;                                              //

            pf.open(Pcap_File, ios::in | ios::binary);                  //以二进制方式读取Pcap文件
            //判断是否成功读取
            if (!pf) {
                throw "Open File Error!";
            }
            pf.read((char*)ph, sizeof(Pcap_Header));                    //从文件中读取指定大小区域的数据为Pcap头
			//读取包头
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
				char* buffer = (char*)malloc(pph->caplen);              //开辟内存空间
				pf.read((char*)buffer, pph->caplen);                    //读取包大小
				count++;                                                //更新包数量
				free(buffer);                                           //释放内存空间
            }
			//判断包数量是否大于最大值
            if (count > max_count) {
                //更新最大包信息
                max_count = count;
                name = Pcap_File;
            }
        }
    }
    //按文件夹搜索pcap文件
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
			//初始化变量
            vector<Feature>features;                            //特征值矩阵
			string Pcap_File = entry.path().string();           //Pcap文件名
			Pcap_Header* ph = new Pcap_Header;                  //Pcap头指针
			Pcap_Packet_Header* pph = new Pcap_Packet_Header;   //Pcap包头指针
			TC_Protocol* tc_ptc;                                //TC_Protocol指针
            ifstream pf;                                        //文件
            long long count = 0;                                //字段长度总和
			int max_len = 0;                                    //最大字段长度
			int min_len = INT_MAX;						        //最小字段长度  

			//打开文件
            pf.open(Pcap_File, ios::in | ios::binary);
			//判断是否成功打开
            if (!pf) {
                throw "Open File Error!";
            }
			//读取Pcap头
            pf.read((char*)ph, sizeof(Pcap_Header));
			//读取包头
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
				char* buffer = (char*)malloc(pph->caplen);                                              //开辟内存空间
				pf.read((char*)buffer, pph->caplen);                                                    //读取包大小 
				tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));                 //获取TC_Protocol指针
				features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));       //将特征值添加到特征矩阵
				free(buffer);				                                                            //释放内存空间  
				count += pph->caplen;                                                                   //更新字段长度总和
				//更新最大最小字段长度
                if (pph->caplen > max_len) {
                    max_len = pph->caplen;
                }
                if (pph->caplen < min_len) {
                    min_len = pph->caplen;
                }
            }
			//加载数据
            LoadData(features, features_matrix, labels, Label_Number(p.filename().string()), max_count, max_len, min_len, count / features.size());
        }
    }
	return make_pair(features_matrix, make_pair(labels, max_count));        //返回特征矩阵和标签矩阵以及最大包数量
}

//主预测函数
double MainPredict(path folderPath, CNN& cnn, int max_count) {
	//初始化变量
	vector<MatrixXd> test_features_matrix;                      //测试数据特征值矩阵动态数组
	vector<VectorXd> labels;                                    //标签记录数组    
	vector<path>dir_paths;  			                        //待搜索文件夹路径数组               
	int correct = 0;                                            //正确预测数量    
	int wrong = 0;                                              //错误预测数量
	int number = 0;                                             //包数量

	//检查路径是否存在
    if (!exists(folderPath) || !is_directory(folderPath)) {
        throw "Folder doesn't exist!";
    }
	//遍历第一级目录将文件夹放入待搜索数组
    for (const auto& entry : directory_iterator(folderPath)) {
        if (is_directory(entry)) {
            dir_paths.push_back(entry.path());
        }
    }
	//按文件夹搜索pcap文件
    for (const auto& p : dir_paths) {
        for (const auto& entry : directory_iterator(p)) {
			//初始化变量
			vector<Feature>features;                                    //特征值矩阵
			string Pcap_File = entry.path().string();                   //获取Pcap文件名
			Pcap_Header* ph = new Pcap_Header;                          //Pcap头指针
			Pcap_Packet_Header* pph = new Pcap_Packet_Header;           //Pcap包头指针
			TC_Protocol* tc_ptc;                                        //TC_Protocol指针
			int count = 0;                                              //字段长度总和
			int max_len = 0;									        //最大字段长度        
			int min_len = INT_MAX;                                      //最小字段长度
			ifstream pf;                                                //文件
			number++;                                                   //更新Pcap文件数量

			//打开文件
			pf.open(Pcap_File, ios::in | ios::binary);                  //以二进制方式读取Pcap文件
			//判断是否成功打开
            if (!pf) {
                cout << "Open File Error!" << endl;
                return -1;
            }
			pf.read((char*)ph, sizeof(Pcap_Header));                    //读取包头
			//读取Pcap包头
            while (pf.read((char*)pph, sizeof(Pcap_Packet_Header))) {
				char* buffer = (char*)malloc(pph->caplen);                                          //开辟内存空间
				pf.read((char*)buffer, pph->caplen);                                                //读取包字段长度                         
				tc_ptc = (TC_Protocol*)(buffer + sizeof(Ethernet2) + sizeof(Protocol));             //获取TC_Protocol指针
				features.push_back(Feature(pph->caplen, (short)ntohs(tc_ptc->destination_port)));   //将特征值添加到特征矩阵
				free(buffer);                                                                       //释放内存空间  
				count += pph->caplen;                                                               //更新字段长度总和
				//更新最大最小字段长度
                if (pph->caplen > max_len) {
                    max_len = pph->caplen;
                }
                if (pph->caplen < min_len) {
                    min_len = pph->caplen;
                }
            }
			//加载数据
            LoadData(features, test_features_matrix, labels, Label_Number(p.filename().string()), max_count, max_len, min_len, count / features.size());
			
            //预测结果
            int predicted_class = cnn.predict(test_features_matrix[test_features_matrix.size() - 1]) + 1;

			//输出结果
            cout << WHITE << "Sample " << setw(3) << std::left << number << setw(3) << ":" << WHITE;
            cout << "Predicted: " << YELLOW << setw(20) << std::left << Website_Name(predicted_class) << WHITE;
            cout << "Actual: " << CYAN<< setw(20) << std::left << Website_Name(Label_Number(p.filename().string()));
            if (predicted_class == Label_Number(p.filename().string())) {
                correct++;
                cout << GREEN << setw(10) <<"[Correct]";
            }
            else {
                wrong++;
                cout << RED << setw(10) << "[Wrong]";
            }
            cout << endl;
        }
    }
	//返回预测准确率
    return 1.0 * correct / (correct + wrong) * 100;
}