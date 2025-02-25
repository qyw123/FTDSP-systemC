#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#define IMAGE_HEIGHT 224
#define IMAGE_WIDTH 224
#define CHANNEL_IN 3
#define CHANNEL_OUT 64
#define KERNEL_SIZE 3
#define STRIDE 1
#define PADDING 1

//计算im2col
//首先读取image_1.txt,image是3维vector,channel first格式
std::vector<std::vector<std::vector<double>>> read_image(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::vector<std::vector<std::vector<double>>> image(CHANNEL_IN, 
        std::vector<std::vector<double>>(IMAGE_HEIGHT, 
            std::vector<double>(IMAGE_WIDTH, 0.0)));
    
    // 按照channel first格式读取数据
    for (int c = 0; c < CHANNEL_IN; ++c) {
        for (int h = 0; h < IMAGE_HEIGHT; ++h) {
            for (int w = 0; w < IMAGE_WIDTH; ++w) {
                if (!(file >> image[c][h][w])) {
                    throw std::runtime_error("文件格式错误或数据不完整");
                }
            }
        }
    }
    
    return image;
}
//读取卷积权重数据，卷积权重数据为4维vector,(channel_out,channel_in,kernel_size,kernel_size)
std::vector<std::vector<std::vector<std::vector<double>>>> read_conv_weight(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }       
    std::vector<std::vector<std::vector<std::vector<double>>>> weight(CHANNEL_OUT, 
        std::vector<std::vector<std::vector<double>>>(CHANNEL_IN, 
            std::vector<std::vector<double>>(KERNEL_SIZE, 
                std::vector<double>(KERNEL_SIZE, 0.0))));
    for (int c_out = 0; c_out < CHANNEL_OUT; ++c_out) {
        for (int c_in = 0; c_in < CHANNEL_IN; ++c_in) {
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
std::vector<double> read_conv_bias(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<double> bias(CHANNEL_OUT, 0.0);
    for (int i = 0; i < CHANNEL_OUT; ++i) {
        if (!(file >> bias[i])) {
            throw std::runtime_error("文件格式错误或数据不完整");
        }
    }
    return bias;
}
//根据卷积核的大小，步长，填充，计算im2col
void im2col(const std::vector<std::vector<std::vector<double>>>& image, std::vector<std::vector<double>>& image_2d,
            const std::vector<std::vector<std::vector<std::vector<double>>>>& weight,std::vector<std::vector<double>>& weight_2d) {
    //image_2d的维度为(kernel_size*kernel_size*channel_in,output_height*output_width)
    //将输入图像的每个局部区域（与卷积核大小相同)展开为列向量,所有列向量组合成一个大的矩阵
    // 计算输出特征图的大小
    int output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // 初始化image_2d
    int rows = KERNEL_SIZE * KERNEL_SIZE * CHANNEL_IN;
    int cols = output_height * output_width;
    image_2d.resize(rows, std::vector<double>(cols, 0.0));
    
    // 创建带padding的图像
    std::vector<std::vector<std::vector<double>>> padded_image(
        CHANNEL_IN,
        std::vector<std::vector<double>>(
            IMAGE_HEIGHT + 2 * PADDING,
            std::vector<double>(IMAGE_WIDTH + 2 * PADDING, 0.0)
        )
    );
    
    // 填充图像
    for (int c = 0; c < CHANNEL_IN; ++c) {
        for (int h = 0; h < IMAGE_HEIGHT; ++h) {
            for (int w = 0; w < IMAGE_WIDTH; ++w) {
                padded_image[c][h + PADDING][w + PADDING] = image[c][h][w];
            }
        }
    }
    
    // 执行im2col操作
    int col_idx = 0;
    for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
            int row_idx = 0;
            for (int c = 0; c < CHANNEL_IN; ++c) {
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
    weight_2d.resize(CHANNEL_OUT, std::vector<double>(KERNEL_SIZE * KERNEL_SIZE * CHANNEL_IN, 0.0));
    
    // 重排卷积核权重
    for (int c_out = 0; c_out < CHANNEL_OUT; ++c_out) {
        int idx = 0;
        for (int c_in = 0; c_in < CHANNEL_IN; ++c_in) {
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
          std::vector<std::vector<double>>& conv_result) {
    
    int output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    
    // 初始化输出矩阵 [64 x 50176]
    conv_result.resize(CHANNEL_OUT, std::vector<double>(output_height * output_width, 0.0));
    
    // 正确的矩阵乘法
    for (int c_out = 0; c_out < CHANNEL_OUT; ++c_out) {
        for (int col = 0; col < output_height * output_width; ++col) {
            double sum = 0.0;
            // 对每个特征点，计算im2col后的一列与权重的内积
            for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE * CHANNEL_IN; ++k) {
                sum += image_2d[k][col] * weight_2d[c_out][k];
            }
            conv_result[c_out][col] = sum + bias[c_out];
        }
    }
}
//把卷积结果由2维展开为3维
void conv_result_to_3d(const std::vector<std::vector<double>>& conv_result, std::vector<std::vector<std::vector<double>>>& conv_result_3d) {
    int output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    int output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1;
    conv_result_3d.resize(CHANNEL_OUT, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));
    for (int c_out = 0; c_out < CHANNEL_OUT; ++c_out) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                conv_result_3d[c_out][h][w] = conv_result[c_out][h * output_width + w];
            }
        }
    }
}
//计算结果存储到block1_conv1_output.txt
void save_conv_result(const std::vector<std::vector<std::vector<double>>>& conv_result_3d) {
    std::ofstream file("./block1_conv1_output.txt");
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

int main() {
    std::vector<std::vector<std::vector<double>>> image = read_image("./image_1.txt");
    std::vector<std::vector<std::vector<std::vector<double>>>> weight = read_conv_weight("./weights/block1_conv1_weights.txt");
    std::vector<double> bias = read_conv_bias("./weights/block1_conv1_biases.txt");
    std::vector<std::vector<double>> image_2d;
    std::vector<std::vector<double>> weight_2d; 
    std::vector<std::vector<double>> conv_result;
    im2col(image, image_2d, weight, weight_2d);
    std::cout << "image_2d shape:" << std::endl;
    std::cout << image_2d.size() << " x " << image_2d[0].size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "weight_2d:" << std::endl;
    std::cout << weight_2d.size() << " x " << weight_2d[0].size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    conv(image_2d, weight_2d, bias, conv_result);
    std::cout << "conv_result shape:" << std::endl;
    std::cout << conv_result.size() << " x " << conv_result[0].size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::vector<std::vector<std::vector<double>>> conv_result_3d;
    conv_result_to_3d(conv_result, conv_result_3d);
    std::cout << "conv_result_3d shape:" << std::endl;
    std::cout << conv_result_3d.size() << " x " << conv_result_3d[0].size() << " x " << conv_result_3d[0][0].size() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    save_conv_result(conv_result_3d);


    return 0;
}