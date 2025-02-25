#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>

#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1
#define POOL_SIZE 2
#define POOL_STRIDE 2


//首先读取input,3维vector,channel first格式
std::vector<std::vector<std::vector<double>>> read_input(const std::string& filename,int channel_in,int input_height,int input_width) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::vector<std::vector<std::vector<double>>> input(channel_in, 
        std::vector<std::vector<double>>(input_height, 
            std::vector<double>(input_width, 0.0)));
    
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
std::vector<std::vector<std::vector<std::vector<double>>>> read_conv_weight(const std::string& filename,int channel_out,int channel_in) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }       
    std::vector<std::vector<std::vector<std::vector<double>>>> weight(channel_out, 
        std::vector<std::vector<std::vector<double>>>(channel_in, 
            std::vector<std::vector<double>>(KERNEL_SIZE, 
                std::vector<double>(KERNEL_SIZE, 0.0))));
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
std::vector<double> read_conv_bias(const std::string& filename,int channel_out) {
    std::ifstream file(filename);
    std::vector<double> bias(channel_out, 0.0);
    for (int i = 0; i < channel_out; ++i) {
        if (!(file >> bias[i])) {
            throw std::runtime_error("文件格式错误或数据不完整");
        }
    }
    return bias;
}
//根据卷积核的大小，步长，填充，计算im2col
void im2col(const std::vector<std::vector<std::vector<double>>>& image, std::vector<std::vector<double>>& image_2d,
            const std::vector<std::vector<std::vector<std::vector<double>>>>& weight,std::vector<std::vector<double>>& weight_2d,
            int channel_out,int channel_in,int input_height,int input_width) {
    //image_2d的维度为(kernel_size*kernel_size*channel_in,output_height*output_width)
    //将输入图像的每个局部区域（与卷积核大小相同)展开为列向量,所有列向量组合成一个大的矩阵
    // 计算输出特征图的大小
    int output_height = (input_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // 初始化image_2d
    int rows = KERNEL_SIZE * KERNEL_SIZE * channel_in;
    int cols = output_height * output_width;
    image_2d.resize(rows, std::vector<double>(cols, 0.0));
    
    // 创建带padding的图像
    std::vector<std::vector<std::vector<double>>> padded_image(
        channel_in,
        std::vector<std::vector<double>>(
            input_height + 2 * PADDING,
            std::vector<double>(input_width + 2 * PADDING, 0.0)
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
    weight_2d.resize(channel_out, std::vector<double>(KERNEL_SIZE * KERNEL_SIZE * channel_in, 0.0));
    
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
void conv(const std::vector<std::vector<double>>& image_2d, 
          const std::vector<std::vector<double>>& weight_2d, 
          const std::vector<double>& bias, 
          std::vector<std::vector<double>>& conv_result,
          int channel_out,int channel_in,int input_height,int input_width) {
    
    int output_height = (input_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // 初始化输出矩阵 [64 x 50176]
    conv_result.resize(channel_out, std::vector<double>(output_height * output_width, 0.0));
    
    // 正确的矩阵乘法
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int col = 0; col < output_height * output_width; ++col) {
            double sum = 0.0;
            // 对每个特征点，计算im2col后的一列与权重的内积
            for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE * channel_in; ++k) {
                sum += image_2d[k][col] * weight_2d[c_out][k];
            }
            conv_result[c_out][col] = sum + bias[c_out];
            //
        }
    }
    //relu激活函数
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int col = 0; col < output_height * output_width; ++col) {
            conv_result[c_out][col] = std::max(0.0, conv_result[c_out][col]);
        }
    }
}
//把卷积结果由2维展开为3维
void conv_result_to_3d(const std::vector<std::vector<double>>& conv_result, std::vector<std::vector<std::vector<double>>>& conv_result_3d,
                        int channel_out,int input_height,int input_width) {
    int output_height = (input_height + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (input_width + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    conv_result_3d.resize(channel_out, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));
    for (int c_out = 0; c_out < channel_out; ++c_out) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                conv_result_3d[c_out][h][w] = conv_result[c_out][h * output_width + w];
            }
        }
    }
}
//计算结果存储到blockn_convn_output.txt
void save_conv_result(const std::vector<std::vector<std::vector<double>>>& conv_result_3d,const std::string& layer) {
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
//池化层,根据VGG的论文，池化层使用最大池化，池化核大小为2x2，步长为2
void pooling(const std::vector<std::vector<std::vector<double>>>& conv_result_3d,
            std::vector<std::vector<std::vector<double>>>& pooling_result_3d,
            int channel_out, int input_height, int input_width) {
    
    // 计算输出尺寸
    int output_height = (input_height) / POOL_STRIDE;
    int output_width = (input_width) / POOL_STRIDE;
    
    // 调整输出tensor的大小
    pooling_result_3d.resize(channel_out, 
        std::vector<std::vector<double>>(output_height, 
            std::vector<double>(output_width, 0.0)));
    
    // 对每个通道进行池化操作
    for (int c = 0; c < channel_out; ++c) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                // 在2x2窗口中找最大值
                double max_val = -std::numeric_limits<double>::infinity();
                
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
int main() {
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
     // 读取输入图像
    std::vector<std::vector<std::vector<double>>> input = read_input("./image_1.txt", 3, 224, 224);
    
    // 用于存储当前层的输出
    std::vector<std::vector<std::vector<double>>> current_output;
    
    for (const auto& layer : conv_layers) {
        if (layer == "pooling") {
            // 创建新的输出tensor而不是直接修改输入
            std::vector<std::vector<std::vector<double>>> pooling_output;
            
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
            
            continue;
        }
        
        // 卷积层处理
        std::string weight_file = "./weights/" + layer + "_weights.txt";
        std::vector<std::vector<std::vector<std::vector<double>>>> weights = 
            read_conv_weight(weight_file, layer_info[layer][0], layer_info[layer][1]);
            
        std::string bias_file = "./weights/" + layer + "_biases.txt";
        std::vector<double> biases = read_conv_bias(bias_file, layer_info[layer][0]);
        
        // im2col
        std::vector<std::vector<double>> input_2d;
        std::vector<std::vector<double>> weight_2d;
        im2col(input, input_2d, weights, weight_2d, 
               layer_info[layer][0], layer_info[layer][1], 
               layer_info[layer][2], layer_info[layer][3]);
        
        // conv
        std::vector<std::vector<double>> conv_result;
        conv(input_2d, weight_2d, biases, conv_result, 
             layer_info[layer][0], layer_info[layer][1], 
             layer_info[layer][2], layer_info[layer][3]);
        
        // conv_result_to_3d
        std::vector<std::vector<std::vector<double>>> conv_result_3d;
        conv_result_to_3d(conv_result, conv_result_3d, 
                         layer_info[layer][0], 
                         layer_info[layer][2], layer_info[layer][3]);
        
        // 更新输入为当前层的输出
        input = std::move(conv_result_3d);
        
        // 保存当前层输出
        save_conv_result(input, layer);
        std::cout << "保存了" << layer << "层的输出" << std::endl;

    }
    
    return 0;
}