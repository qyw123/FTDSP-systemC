#ifndef TOOLS_H
#define TOOLS_H

#include "const.h"
//计算传输数据所需时钟周期
uint64_t calculate_clock_cycles(uint64_t data_size, uint64_t data_width) {
    return (data_size + data_width - 1) / data_width;
}
//三个向量合并为一个向量
template <typename T>
void merge_vectors(const vector<T>& vec1, const vector<T>& vec2, const vector<T>& vec3, vector<T>& merged_vec) {
    //vec1,vec2,vec3的元素个数必须相等
    if(vec1.size() != vec2.size() || vec1.size() != vec3.size()){
        SC_REPORT_ERROR("merge_vectors", "三个向量的元素个数不相等");
        return;
    }
    merged_vec.resize(vec1.size() + vec2.size() + vec3.size());
    for(int i = 0; i < vec1.size(); i++) {
        memcpy(&merged_vec[i*3], &vec1[i], sizeof(T));
        memcpy(&merged_vec[i*3 + 1], &vec2[i], sizeof(T));
        memcpy(&merged_vec[i*3 + 2], &vec3[i], sizeof(T));
    }
}
//一个向量拆分为三个向量
template <typename T>
void split_vector(const vector<T>& merged_vec, vector<T>& vec1, vector<T>& vec2, vector<T>& vec3) {
    vec1.resize(merged_vec.size() / 3);
    vec2.resize(merged_vec.size() / 3);
    vec3.resize(merged_vec.size() / 3);
    for(int i = 0; i < merged_vec.size() / 3; i++) {
        memcpy(&vec1[i], &merged_vec[i*3], sizeof(T));
        memcpy(&vec2[i], &merged_vec[i*3 + 1], sizeof(T));
        memcpy(&vec3[i], &merged_vec[i*3 + 2], sizeof(T));
    }
}
namespace dmi_utils {  
    // 读取DMI数据的通用函数    
    template<typename T>
    void read_from_dmi(uint64_t addr, std::vector<T>& values, 
                      const tlm::tlm_dmi& dmi, unsigned int data_num,
                      const std::string& module_name = "DMI_Utils") {
        const unsigned int bytes_per_block = DDR_DATA_WIDTH;
        const unsigned int elements_per_block = bytes_per_block / sizeof(T);

        if (addr + data_num * sizeof(T) - 1 <= dmi.get_end_address()) {
            values.resize(data_num);
            unsigned char* dmi_addr = dmi.get_dmi_ptr() + (addr - dmi.get_start_address());
            unsigned int total_blocks = (data_num + elements_per_block - 1) / elements_per_block;

            for (unsigned int block = 0; block < total_blocks; ++block) {
                unsigned int block_start = block * elements_per_block;
                unsigned int block_end = std::min(block_start + elements_per_block, data_num);
                
                for (unsigned int i = block_start; i < block_end; ++i) {
                    memcpy(&values[i], dmi_addr + i * sizeof(T), sizeof(T));
                }
                
                wait(dmi.get_read_latency());
            }
        } else {
            SC_REPORT_ERROR(module_name.c_str(), "DMI read failed: Address out of range");
        }
    }

    // 写入DMI数据的通用函数
    template<typename T>
    void write_to_dmi(uint64_t start_addr, uint64_t& end_addr, 
                     const std::vector<T>& values, const tlm::tlm_dmi& dmi, 
                     unsigned int data_num, const std::string& module_name = "DMI_Utils") {
        const unsigned int bytes_per_block = DDR_DATA_WIDTH;
        const unsigned int elements_per_block = bytes_per_block / sizeof(T);

        if (data_num != values.size()) {
            SC_REPORT_ERROR(module_name.c_str(), "Mismatch between data_num and values size");
            return;
        }

        end_addr = start_addr + data_num * sizeof(T) - 1;

        if (end_addr <= dmi.get_end_address() && start_addr >= dmi.get_start_address()) {
            unsigned char* dmi_addr = dmi.get_dmi_ptr() + (start_addr - dmi.get_start_address());
            unsigned int total_blocks = (data_num + elements_per_block - 1) / elements_per_block;

            for (unsigned int block = 0; block < total_blocks; ++block) {
                unsigned int block_start = block * elements_per_block;
                unsigned int block_end = std::min(block_start + elements_per_block, data_num);

                for (unsigned int i = block_start; i < block_end; ++i) {
                    memcpy(dmi_addr + i * sizeof(T), &values[i], sizeof(T));
                }
                
                wait(dmi.get_write_latency());
            }
        } else {
            SC_REPORT_ERROR(module_name.c_str(), "DMI write failed: Address out of range");
        }
    }
}

// 加载文件数据到 vector
template <typename T>
void load_from_file(std::vector<T>& data_buffer, std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        SC_REPORT_ERROR("load_from_file", "Failed to open file.");
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        T value;
        while (stream >> value) {
            data_buffer.push_back(value);
        }
    }
    file.close();
}

template <typename T>
void write_matrix_in_file(std::vector<T>& data_buffer, const std::string& file_path, int rows, int cols) {
    ofstream outfile(file_path);
    if (!outfile.is_open()) {
        SC_REPORT_ERROR("write_in_file", "Failed to open file.");
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outfile << data_buffer[i * cols + j] << " ";
        }
        outfile << endl;
    }
    outfile.close();
}
template <typename T>
void write_matrix_in_file_2d(std::vector<std::vector<T>>& data_buffer, const std::string& file_path, int rows, int cols) {
    ofstream outfile(file_path);
    if (!outfile.is_open()) {
        SC_REPORT_ERROR("write_in_file", "Failed to open file.");
        return; 
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outfile << data_buffer[i][j] << " ";
        }
        outfile << endl;
    }
}
// 记录矩阵形状
template <typename T>
void record_matrix_shape(const std::string& file_path, int& rows, int& cols) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        SC_REPORT_ERROR("record_matrix_shape", "Failed to open file.");
        return;
    }
    if(rows == 0 && cols == 0){
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        int current_cols = 0;
        T value;
        while (stream >> value) {
            ++current_cols;
        }
        if (cols == 0) {
            cols = current_cols;
        } else if (cols != current_cols) {
            SC_REPORT_ERROR("record_matrix_shape", "Inconsistent column sizes.");
            return;
        }
        ++rows;
    }
    file.close();
    }
    else{
        SC_REPORT_INFO("record_matrix_shape", "矩阵形状提前已设定");
        return;
    }
}

template <typename T>
void convertTo2D(const vector<T>& input, vector<vector<T>>& output, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            output[i][j] = input[i * cols + j];
        }
    }
}
template <typename T>
void convertTo1D(const vector<vector<T>>& input, vector<T>& output) {
    int rows = input.size();
    int cols = input[0].size();
    output.resize(rows * cols);
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            output[i * cols + j] = input[i][j];
        }
    }
}
// 矩阵乘法并保存结果到文件的函数
template <typename T>
void multiplyAndSaveMatrices(const vector<vector<T>>& mat1, 
                             const vector<vector<T>>& mat2, 
                             const string& filename) {
    if (mat1[0].size() != mat2.size()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    size_t rows = mat1.size();
    size_t cols = mat2[0].size();
    size_t inner = mat2.size();

    vector<vector<double>> result(rows, vector<double>(cols, 0.0));

    // 矩阵乘法
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    // 保存结果到文件
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file.");
    }

    for (const auto& row : result) {
        for (const auto& elem : row) {
            file << elem << " ";
        }
        file << "\n";
    }

    file.close();
}

template <typename T>
void check_all_zero(const vector<T> buffer){
    for(int i = 0; i < buffer.size(); i++){
        if(buffer[i] != 0){
            //cout << "buffer is not all zero"<<endl;
            return;
        }
    }
    cout << "buffer is all zero!!!!!!!!!!!!!!!!!!!!"<<endl;
    return;
}
#endif
