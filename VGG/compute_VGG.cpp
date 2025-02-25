#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include "./gemm.h"
#include "./src/Soc.h"
#include <systemc.h>
#include <chrono>
#include <unordered_map>
#include <string>
#include <utility>
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1
#define POOL_SIZE 2
#define POOL_STRIDE 2
using DataType = float;
// 创建全局对象
Soc<DataType>* soc;
Gemm<DataType>* gemm;

// 初始化函数
void init_gemm() {
    soc = new Soc<DataType>("soc");
    gemm = new Gemm<DataType>("gemm");
    gemm->socket.bind(soc->target_socket);
    sc_start(SC_ZERO_TIME);
}


//首先读取input,3维vector,channel first格式
std::vector<std::vector<std::vector<DataType>>> read_input(const std::string& filename,int channel_in,int input_height,int input_width) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::vector<std::vector<std::vector<DataType>>> input(channel_in, 
        std::vector<std::vector<DataType>>(input_height, 
            std::vector<DataType>(input_width, 0.0)));
    
    // 按照channel first格式读取数据
    for (int c = 0; c < channel_in; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                if (!(file >> input[c][h][w])) {
                    throw std::runtime_error("文件格式错误或数据不完整");
                }
            }
        }
    }
    
    return input;
}
//读取卷积权重数据，卷积权重数据为4维vector,(channel_out,channel_in,kernel_size,kernel_size)
std::vector<std::vector<std::vector<std::vector<DataType>>>> read_conv_weight(const std::string& filename,int channel_out,int channel_in) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }       
    std::vector<std::vector<std::vector<std::vector<DataType>>>> weight(channel_out, 
        std::vector<std::vector<std::vector<DataType>>>(channel_in, 
            std::vector<std::vector<DataType>>(KERNEL_SIZE, 
                std::vector<DataType>(KERNEL_SIZE, 0.0))));
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int c_in = 0; c_in < channel_in; ++c_in) {
            for (int h = 0; h < KERNEL_SIZE; ++h) {
                for (int w = 0; w < KERNEL_SIZE; ++w) {
                    if (!(file >> weight[c_out][c_in][h][w])) {
                        throw std::runtime_error("文件格式错误或数据不完整");
                    }
                }
            }
        }
    }
    return weight;
}
//读取卷积的偏置数据，偏置数据为1维vector,维度为channel_out
std::vector<DataType> read_conv_bias(const std::string& filename,int channel_out) {
    std::ifstream file(filename);
    std::vector<DataType> bias(channel_out, 0.0);
    for (int i = 0; i < channel_out; ++i) {
        if (!(file >> bias[i])) {
            throw std::runtime_error("文件格式错误或数据不完整");
        }
    }
    return bias;
}

//读取全连接层的权重数据，权重数据为2维vector,(channel_in,channel_out)
std::vector<std::vector<DataType>> read_fc_weight(const std::string& filename,int channel_in,int channel_out) {
    std::ifstream file(filename);
    std::vector<std::vector<DataType>> weight(channel_in, std::vector<DataType>(channel_out, 0.0));
    for (int i = 0; i < channel_in; ++i) {
        for (int j = 0; j < channel_out; ++j) {
            if (!(file >> weight[i][j])) {
                throw std::runtime_error("文件格式错误或数据不完整");
            }
        }
    }
    return weight;
}
//读取全连接层的偏置数据，偏置数据为1维vector,维度为channel_out
std::vector<DataType> read_fc_bias(const std::string& filename,int channel_out) {
    std::ifstream file(filename);
    std::vector<DataType> bias(channel_out, 0.0);
    for (int i = 0; i < channel_out; ++i) {
        if (!(file >> bias[i])) {
            throw std::runtime_error("文件格式错误或数据不完整");
        }
    }
    return bias;
}
//根据卷积核的大小，步长，填充，计算im2col
void im2col(const std::vector<std::vector<std::vector<DataType>>>& image, std::vector<std::vector<DataType>>& image_2d,
            const std::vector<std::vector<std::vector<std::vector<DataType>>>>& weight,std::vector<std::vector<DataType>>& weight_2d,
            int channel_out,int channel_in,int input_height,int input_width) {
    //image_2d的维度为(kernel_size*kernel_size*channel_in,output_height*output_width)
    //将输入图像的每个局部区域（与卷积核大小相同)展开为列向量,所有列向量组合成一个大的矩阵
    // 计算输出特征图的大小
    int output_height = (input_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // 初始化image_2d
    int rows = KERNEL_SIZE * KERNEL_SIZE * channel_in;
    int cols = output_height * output_width;
    image_2d.resize(rows, std::vector<DataType>(cols, 0.0));
    
    // 创建带padding的图像
    std::vector<std::vector<std::vector<DataType>>> padded_image(
        channel_in,
        std::vector<std::vector<DataType>>(
            input_height + 2 * PADDING,
            std::vector<DataType>(input_width + 2 * PADDING, 0.0)
        )
    );
    
    // 填充图像
    for (int c = 0; c < channel_in; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                padded_image[c][h + PADDING][w + PADDING] = image[c][h][w];
            }
        }
    }
    
    // 执行im2col操作
    int col_idx = 0;
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            int row_idx = 0;
            for (int c = 0; c < channel_in; ++c) {
                for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                    for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                        image_2d[row_idx][col_idx] = 
                            padded_image[c][h * STRIDE + kh][w * STRIDE + kw];
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
    //weight_2d的维度为(channel_out,kernel_size*kernel_size*channel_in)
    // 初始化weight_2d
    weight_2d.resize(channel_out, std::vector<DataType>(KERNEL_SIZE * KERNEL_SIZE * channel_in, 0.0));
    
    // 重排卷积核权重
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        int idx = 0;
        for (int c_in = 0; c_in < channel_in; ++c_in) {
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    weight_2d[c_out][idx++] = weight[c_out][c_in][kh][kw];
                }
            }
        }
    }
}
//计算卷积
void conv(const std::vector<std::vector<DataType>>& input_2d, 
          const std::vector<std::vector<DataType>>& weight_2d, 
          const std::vector<DataType>& bias, 
          std::vector<std::vector<DataType>>& conv_result,
          int channel_out,int channel_in,int input_height,int input_width) {
    
    int output_height = (input_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // 初始化输出矩阵 [64 x 50176]
    conv_result.resize(channel_out, std::vector<DataType>(output_height * output_width, 0.0));

    // // 正确的矩阵乘法
    // for (int c_out = 0; c_out < channel_out; ++c_out) {
    //     for (int col = 0; col < output_height * output_width; ++col) {
    //         DataType sum = 0.0;
    //         // 对每个特征点，计算im2col后的一列与权重的内积
    //         for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE * channel_in; ++k) {
    //             sum += input_2d[k][col] * weight_2d[c_out][k];
    //         }
    //         conv_result[c_out][col] = sum + bias[c_out];
    //         //
    //     }
    // }

    //conv_result加入bias
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int col = 0; col < output_height * output_width; ++col) {
            conv_result[c_out][col] += bias[c_out];
        }
    }   
    //cout << "=============开始GEMM计算=============="<<endl;
    //加一个时间戳，记录真实执行时间，真实的时间不是sc_time_stamp()，而是系统时间
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm->compute_matrix(
        weight_2d,                    // A矩阵
        input_2d,                     // B矩阵
        conv_result,                  // C矩阵
        channel_out,                  // M
        output_height * output_width, // N
        KERNEL_SIZE * KERNEL_SIZE * channel_in);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    cout << "GEMM计算时间: " << dec << duration.count() << "秒" << endl;
    //relu激活函数
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int col = 0; col < output_height * output_width; ++col) {
            conv_result[c_out][col] = std::max(static_cast<DataType>(0.0), conv_result[c_out][col]);
        }
    }
    //write_matrix_in_file_2d(conv_result, "conv_result_gemm.txt", channel_out, output_height * output_width);
}
//把卷积结果由2维展开为3维
void conv_result_to_3d(const std::vector<std::vector<DataType>>& conv_result, std::vector<std::vector<std::vector<DataType>>>& conv_result_3d,
                        int channel_out,int input_height,int input_width) {
    int output_height = (input_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    conv_result_3d.resize(channel_out, std::vector<std::vector<DataType>>(output_height, std::vector<DataType>(output_width, 0.0)));
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                conv_result_3d[c_out][h][w] = conv_result[c_out][h * output_width + w];
            }
        }
    }
}
//计算结果存储到blockn_convn_output.txt
void save_conv_result(const std::vector<std::vector<std::vector<DataType>>>& conv_result_3d,const std::string& layer) {
    std::ofstream file("./temp/" + layer + "_output.txt");
    if (!file.is_open()) {
        throw std::runtime_error("无法打开输出文件");
    }
    
    // 对3维数据进行遍历
    for (const auto& channel : conv_result_3d) {         // 遍历每个输出通道
        for (const auto& row : channel) {                // 遍历每一行
            for (const auto& val : row) {                // 遍历每一列
                file << val << " ";
            }
            file << std::endl;
        }
    }
    
    file.close();
}

//保存fc_result
void save_fc_result(const std::vector<DataType>& fc_result,const std::string& layer) {
    std::ofstream file("./temp/" + layer + "_output_对照组.txt");
    if (!file.is_open()) {
        throw std::runtime_error("无法打开输出文件");
    }
    for (const auto& val : fc_result) {
        file << val << std::endl;
    }
    file.close();
}
//池化层,根据VGG的论文，池化层使用最大池化，池化核大小为2x2，步长为2
void pooling(const std::vector<std::vector<std::vector<DataType>>>& conv_result_3d,
            std::vector<std::vector<std::vector<DataType>>>& pooling_result_3d,
            int channel_out, int input_height, int input_width) {
    
    // 计算输出尺寸
    int output_height = (input_height) / POOL_STRIDE;
    int output_width = (input_width) / POOL_STRIDE;
    
    // 调整输出tensor的大小
    pooling_result_3d.resize(channel_out, 
        std::vector<std::vector<DataType>>(output_height, 
            std::vector<DataType>(output_width, 0.0)));
    
    // 对每个通道进行池化操作
    for (int c = 0; c < channel_out; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                // 在2x2窗口中找最大值
                DataType max_val = -std::numeric_limits<DataType>::infinity();
                
                // 遍历池化窗口
                for (int ph = 0; ph < POOL_SIZE; ++ph) {
                    for (int pw = 0; pw < POOL_SIZE; ++pw) {
                        // 计算输入特征图上的位置
                        int in_h = h * POOL_STRIDE + ph;
                        int in_w = w * POOL_STRIDE + pw;
                        
                        // 确保在输入范围内
                        if (in_h < input_height && in_w < input_width) {
                            max_val = std::max(max_val, conv_result_3d[c][in_h][in_w]);
                        }
                    }
                }
                
                // 存储最大值
                pooling_result_3d[c][h][w] = max_val;
            }
        }
    }
}
//将3维的pooling_result展开为1维
void pooling_result_to_1d(const std::vector<std::vector<std::vector<DataType>>>& pooling_result_3d,std::vector<DataType>& pooling_result_1d) {
    for (const auto& channel : pooling_result_3d) {
        for (const auto& row : channel) {
            for (const auto& val : row) {
                pooling_result_1d.push_back(val);
            }
        }
    }
}
//全连接层,输入conv_result是1维，输出fc_result是1维
void fc_layer(const std::vector<DataType>& conv_result_1d,std::vector<DataType>& fc_result,
        const std::vector<std::vector<DataType>>& weight,const std::vector<DataType>& bias,
        int channel_in,int channel_out) {

    //fc_result的维度为channel_out
    fc_result.resize(channel_out, 0.0);

    // //普通计算：fc_result = conv_result_1d * weight + bias
    // for (int i = 0; i < channel_out; ++i) {
    //     DataType sum = 0.0;
    //     for (int j = 0; j < channel_in; ++j) {
    //         sum += conv_result_1d[j] * weight[j][i];
    //     }
    //     fc_result[i] = sum + bias[i];
    // }

    //gemm计算
    //fc_result加入bias
    for (int i = 0; i < channel_out; ++i) {
        fc_result[i] += bias[i];
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm->compute_fc(
        conv_result_1d,     //[channel_in]
        weight,            //[channel_in,channel_out]
        fc_result,        //[channel_out]
        channel_in,        // K
        channel_out);      // N
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    cout << "GEMM计算时间: " << dec << duration.count() << "秒" << endl;

}
//softmax
void softmax(const std::vector<DataType>& fc_result,std::vector<DataType>& softmax_result) {
    DataType sum = 0.0;
    for (const auto& val : fc_result) {
        sum += std::exp(val);
    }
    for (const auto& val : fc_result) {
        softmax_result.push_back(std::exp(val) / sum);
    }
}   
// 加载类别名称
std::vector<std::string> load_class_names(const std::string& synset_file, const std::string& class_file) {
    std::unordered_map<std::string, std::string> synset_to_desc;
    std::ifstream synset_f(synset_file);
    std::string line;

    // 加载类别编号到类别描述的映射
    while (std::getline(synset_f, line)) {
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string synset_id = line.substr(0, space_pos);
            std::string desc = line.substr(space_pos + 1);
            synset_to_desc[synset_id] = desc;
        }
    }

    // 加载类别编号列表
    std::vector<std::string> class_names;
    std::ifstream class_f(class_file);
    while (std::getline(class_f, line)) {
        class_names.push_back(synset_to_desc[line]);
    }

    return class_names;
}

// 打印 Top-5 预测结果
void print_top5(const std::vector<DataType>& softmax_result, const std::vector<std::string>& class_names) {
    std::vector<std::pair<DataType, int>> sorted_result(softmax_result.size());
    for (size_t i = 0; i < softmax_result.size(); ++i) {
        sorted_result[i] = {softmax_result[i], i};
    }

    // 按概率从大到小排序
    std::sort(sorted_result.begin(), sorted_result.end(), std::greater<std::pair<DataType, int>>());

    // 打印 Top-5 类别名称
    std::cout << "Top 5 predictions:" << std::endl;
    for (size_t i = 0; i < 5 && i < sorted_result.size(); ++i) {
        int class_id = sorted_result[i].second;
        DataType prob = sorted_result[i].first;
        std::cout << i + 1 << ": " << class_names[class_id] << " (probability: " << prob << ")" << std::endl;
    }
}
//读入conv_result
void read_conv_result(
    const std::string& filename, 
    std::vector<DataType>& conv_result,
    int channels, int height, int width
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    // 预分配空间
    conv_result.resize(channels*height*width);

    // 按通道读取数据
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                file >> conv_result[c*height*width+h*width+w];
            }
        }
    }
    file.close();
} 
int process_vgg() {
    // 卷积层列表
    std::vector<std::string> conv_layers = {
        "block1_conv1", "block1_conv2","pooling",
        "block2_conv1", "block2_conv2","pooling",
        "block3_conv1", "block3_conv2", "block3_conv3","pooling",
        "block4_conv1", "block4_conv2", "block4_conv3","pooling",
        "block5_conv1", "block5_conv2", "block5_conv3","pooling"
    };
    std::map<std::string,std::vector<int>> layer_info;//{channel_out,channel_in,input_height,input_width}
    layer_info["block1_conv1"] = {64,3,224,224};
    layer_info["block1_conv2"] = {64,64,224,224};
    layer_info["block2_conv1"] = {128,64,112,112};
    layer_info["block2_conv2"] = {128,128,112,112};
    layer_info["block3_conv1"] = {256,128,56,56};
    layer_info["block3_conv2"] = {256,256,56,56};
    layer_info["block3_conv3"] = {256,256,56,56};
    layer_info["block4_conv1"] = {512,256,28,28};
    layer_info["block4_conv2"] = {512,512,28,28};
    layer_info["block4_conv3"] = {512,512,28,28};
    layer_info["block5_conv1"] = {512,512,14,14};
    layer_info["block5_conv2"] = {512,512,14,14};
    layer_info["block5_conv3"] = {512,512,14,14};
    //fc:[channel_in,channel_out]
    layer_info["fc1"] = {25088,4096};
    layer_info["fc2"] = {4096,4096};
    layer_info["predictions"] = {4096,1000};
     // 读取输入图像
    std::vector<std::vector<std::vector<DataType>>> input = read_input("./image_1.txt", 3, 224, 224);

    std::vector<DataType> conv_result_1d;
    // 用于存储当前层的输出
    std::vector<std::vector<std::vector<DataType>>> current_output;

    for (const auto& layer : conv_layers) {
        if (layer == "pooling") {
            // 创建新的输出tensor而不是直接修改输入
            std::vector<std::vector<std::vector<DataType>>> pooling_output;
            
            // 获取当前输入的维度
            int current_channels = input.size();
            int current_height = input[0].size();
            int current_width = input[0][0].size();
            // 执行池化操作
            pooling(input, pooling_output, 
                   current_channels, 
                   current_height, 
                   current_width);
            
            // 更新输入为池化后的结果
            input = std::move(pooling_output);  // 使用移动语义避免拷贝
            //保存池化层输出
            save_conv_result(input, layer);
            std::cout << "保存了" << layer << "层的输出" << std::endl;
            continue;
        }
        
        // 卷积层处理
        std::string weight_file = "./weights/" + layer + "_weights.txt";
        std::vector<std::vector<std::vector<std::vector<DataType>>>> weights = 
            read_conv_weight(weight_file, layer_info[layer][0], layer_info[layer][1]);
            
        std::string bias_file = "./weights/" + layer + "_biases.txt";
        std::vector<DataType> biases = read_conv_bias(bias_file, layer_info[layer][0]);
        
        // im2col
        std::vector<std::vector<DataType>> input_2d;
        std::vector<std::vector<DataType>> weight_2d;
        im2col(input, input_2d, weights, weight_2d, 
               layer_info[layer][0], layer_info[layer][1], 
               layer_info[layer][2], layer_info[layer][3]);
        
        // conv
        std::vector<std::vector<DataType>> conv_result;
        conv(input_2d, weight_2d, biases, conv_result, 
             layer_info[layer][0], layer_info[layer][1], 
             layer_info[layer][2], layer_info[layer][3]);
        
        // conv_result_to_3d
        std::vector<std::vector<std::vector<DataType>>> conv_result_3d;
        conv_result_to_3d(conv_result, conv_result_3d, 
                         layer_info[layer][0], 
                         layer_info[layer][2], layer_info[layer][3]);
        // 更新输入为当前层的输出
        input = std::move(conv_result_3d);
        
        // 保存当前层输出
        save_conv_result(input, layer);
        std::cout << "保存了" << layer << "层的输出" << std::endl;
    }
    pooling_result_to_1d(input, conv_result_1d);
    // //读入pooling_output.txt，直接开始fc1层的计算
    // read_conv_result("./temp/pooling_output.txt", conv_result_1d,512,7,7);
    //全连接层fc1
    std::string weight_file = "./weights/fc1_weights.txt";
    std::vector<std::vector<DataType>> fc_weights = read_fc_weight(weight_file, layer_info["fc1"][0], layer_info["fc1"][1]);
    std::string bias_file = "./weights/fc1_biases.txt";
    std::vector<DataType> fc_biases = read_fc_bias(bias_file, layer_info["fc1"][1]);
    std::vector<DataType> fc1_result;
    fc_layer(conv_result_1d, fc1_result, fc_weights, fc_biases, layer_info["fc1"][0],layer_info["fc1"][1]);
    save_fc_result(fc1_result, "fc1");
    std::cout << "保存了fc1层的输出" << std::endl;
    //全连接层fc2
    weight_file = "./weights/fc2_weights.txt";
    fc_weights = read_fc_weight(weight_file, layer_info["fc2"][0], layer_info["fc2"][1]);
    bias_file = "./weights/fc2_biases.txt";
    fc_biases = read_fc_bias(bias_file, layer_info["fc2"][1]);
    std::vector<DataType> fc2_result;
    fc_layer(fc1_result, fc2_result, fc_weights, fc_biases, layer_info["fc2"][0],layer_info["fc2"][1]);
    save_fc_result(fc2_result, "fc2");
    std::cout << "保存了fc2层的输出" << std::endl;
    //全连接层fc3
    weight_file = "./weights/predictions_weights.txt";
    fc_weights = read_fc_weight(weight_file, layer_info["predictions"][0], layer_info["predictions"][1]);
    bias_file = "./weights/predictions_biases.txt";
    fc_biases = read_fc_bias(bias_file, layer_info["predictions"][1]);
    std::vector<DataType> predictions_result;
    fc_layer(fc2_result, predictions_result, fc_weights, fc_biases, layer_info["predictions"][0],layer_info["predictions"][1]);
    save_fc_result(predictions_result, "fc3");
    std::cout << "保存了fc3层的输出" << std::endl;
    //softmax
    std::vector<DataType> softmax_result;
    softmax(predictions_result, softmax_result);
    //保存softmax_result
    save_fc_result(softmax_result, "predictions");
    std::cout << "保存了softmax_result层的输出" << std::endl;
        // 加载类别名称
    std::string synset_file = "imagenet_synsets.txt";
    std::string class_file = "imagenet_classes.txt";
    std::vector<std::string> class_names = load_class_names(synset_file, class_file);
    // 打印 Top-5 预测结果
    print_top5(softmax_result, class_names);

    return 0;
}

// SystemC的入口函数
int sc_main(int argc, char* argv[]) {
    //添加一个时间戳
    auto vgg_start_time = std::chrono::high_resolution_clock::now();
    // 初始化GEMM
    init_gemm();
    // 执行VGG处理
    process_vgg();
    auto vgg_end_time = std::chrono::high_resolution_clock::now();
    auto vgg_duration = std::chrono::duration_cast<std::chrono::seconds>(vgg_end_time - vgg_start_time);
    std::cout << "VGG执行时间: " << vgg_duration.count() << "秒" << std::endl;
    // 清理资源
    delete soc;
    delete gemm;
    sc_stop();
    return 0;
}