#对比计算结果是否相同，两个txt文件内容是否一致
im2col_result_file_path = "../block1_conv1_output.txt"
pytorch_result_file_path = "./block1_conv1_python_output.txt"

def count_numbers_in_file(file_path):
    number_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            # 过滤掉空行
            if line.strip():
                # 分割并过滤掉空字符串
                numbers = [x for x in line.strip().split() if x]
                number_count += len(numbers)
    return number_count

# 读取并处理文件内容
def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            # 过滤掉空行
            if line.strip():
                # 分割并过滤掉空字符串，转换为float
                numbers.extend([float(x) for x in line.strip().split() if x])
    return numbers

# 读取两个文件的数字
im2col_result = read_numbers_from_file(im2col_result_file_path)
pytorch_result = read_numbers_from_file(pytorch_result_file_path)

# 打印两个文件的数字个数
print("im2col_result has", len(im2col_result), "numbers")
print("pytorch_result has", len(pytorch_result), "numbers")

# 确定两个文件中元素个数是否相同
if len(im2col_result) != len(pytorch_result):
    print("im2col_result and pytorch_result have different number of elements")
    exit()

# 逐个元素对比
for i in range(len(im2col_result)):
    if abs(im2col_result[i] - pytorch_result[i]) > 0.1:
        print(f"im2col_result[{i}] = {im2col_result[i]} is not equal to pytorch_result[{i}] = {pytorch_result[i]}")
        break

print("im2col_result and pytorch_result are the same")