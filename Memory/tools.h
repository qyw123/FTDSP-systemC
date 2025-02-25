#ifndef TOOLS_H
#define TOOLS_H

#include "const.h"

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

// 记录矩阵形状
template <typename T>
void record_matrix_shape(const std::string& file_path, int& rows, int& cols) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        SC_REPORT_ERROR("record_matrix_shape", "Failed to open file.");
        return;
    }
    rows = 0;
    cols = 0;
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

#endif
