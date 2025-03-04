import numpy as np
import os

def generate_fft_inputs():
    for exp in range(10, 30):#10~29
        n = 1 << exp  # 2^exp
        filename = f"fft_{exp}_2_input.npy"
        
        # 生成顺序的复数数据（实部为0~n-1, 虚部为0~n-1）
        real_part = np.arange(n, dtype=np.float32)
        imag_part = np.arange(n, dtype=np.float32)
        complex_data = np.empty((n * 2,), dtype=np.float32)
        complex_data[0::2] = real_part
        complex_data[1::2] = imag_part
        
        # 保存为 .npy 文件
        np.save(filename, complex_data)
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_fft_inputs()