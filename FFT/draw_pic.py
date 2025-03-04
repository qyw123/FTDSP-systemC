import matplotlib.pyplot as plt

# 读取执行时间数据
x_values = []
y_values = []

with open("fft_execution_times.txt", "r") as f:
    next(f)  # 跳过标题行
    for line in f:
        parts = line.split()
        exponent = int(parts[0].split("_")[1])  # 解析 2^x 指数
        time_ms = float(parts[2])  # 解析执行时间（毫秒）

        if 10 <= exponent <= 29:  # 确保横坐标在 10 到 29 之间
            x_values.append(exponent)
            y_values.append(time_ms)

# 绘制折线图
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue', label="FFT Execution Time")
plt.xticks(range(10, 21))  # 设置横坐标刻度为整数
plt.xlabel("Exponent (2^x)")
plt.ylabel("Execution Time (ms)")
plt.title("FFT Execution Time vs Input Size")
plt.grid(True)
plt.legend()
plt.savefig("fft_execution_time.jpg", dpi=300)  # 增加分辨率
plt.show()
