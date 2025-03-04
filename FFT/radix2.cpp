#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <fstream>
#include <ctime>      // 用于 clock_gettime
#include <sys/time.h> // POSIX 系统的高精度时间
#include "cnpy.h"  // 使用 cnpy 读取 .npy 文件

using namespace std;
using Complex = complex<double>;
using Clock = chrono::high_resolution_clock;

// 记录 CPU 执行时间
double get_cpu_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

// 交换二进制反转索引的数据位置
void bit_reverse_swap(vector<Complex>& data) {
    int n = data.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            swap(data[i], data[j]);
        }
    }
}

// 基 2 FFT 计算
void fft(vector<Complex>& data) {
    int n = data.size();
    bit_reverse_swap(data);
    
    for (int len = 2; len <= n; len *= 2) {
        double angle = -2.0 * M_PI / len;
        Complex wlen(cos(angle), sin(angle));
        
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int j = 0; j < len / 2; ++j) {
                Complex u = data[i + j];
                Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// 从 .npy 文件读取复数数据
vector<Complex> read_input(const string& filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    double* raw_data = arr.data<double>();
    vector<Complex> data(arr.shape[0] / 2);
    
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = Complex(raw_data[2 * i], raw_data[2 * i + 1]);
    }
    return data;
}

// 将 FFT 计算结果保存到 .npy 文件
void write_output(const string& filename, const vector<Complex>& data) {
    vector<double> flat_data(data.size() * 2);
    for (size_t i = 0; i < data.size(); ++i) {
        flat_data[2 * i] = data[i].real();
        flat_data[2 * i + 1] = data[i].imag();
    }
    cnpy::npy_save(filename, flat_data.data(), {data.size(), 2}, "w");
}

// 检查文件是否存在
bool file_exists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

int main() {
    ofstream time_log("fft_execution_times.txt");
    time_log << "Filename ExecutionTime(ms) CPUTime(ms)\n";
    
    for (int exp = 10; exp < 21; ++exp) {
        string input_file = "./data/fft_" + to_string(exp) + "_2_input.npy";
        string output_file = "./data/fft_" + to_string(exp) + "_2_output.npy";
        
        if (!file_exists(input_file)) {
            cerr << "File not found: " << input_file << endl;
            continue;
        }
        
        vector<Complex> data = read_input(input_file);
        
        auto start = Clock::now();
        double cpu_start = get_cpu_time_ms();  // 记录 CPU 开始时间

        fft(data);

        double cpu_end = get_cpu_time_ms();  // 记录 CPU 结束时间
        auto end = Clock::now();
        
        write_output(output_file, data);
        
        chrono::duration<double, milli> elapsed = end - start;
        double cpu_time_ms = cpu_end - cpu_start;

        time_log << input_file << " " << elapsed.count() << " " << cpu_time_ms << "\n";
        cout << "Processed " << input_file << " in " << elapsed.count() 
             << " ms (CPU Time: " << cpu_time_ms << " ms)" << endl;
    }
    
    time_log.close();

    return 0;
}
