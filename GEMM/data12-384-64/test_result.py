#本代码用于检查MatrixC_output.txt与真实结果是否一致
import numpy as np
#工具函数
def read_matrix_from_file(file_path):
    with open(file_path, "r") as file:
        matrix = [list(map(float, line.split())) for line in file]
    #打印矩阵的shape
    #print("matrix shape:",np.array(matrix).shape)
    return np.array(matrix)

def check_result(result_truth,output):
    #打印真实结果的shape
    print("result_truth shape:",result_truth.shape)
    #打印真实结果的最后五个元素
    # print("result_truth last five elements:",result_truth[-5:])
    #打印输出结果的shape
    print("output shape:",output.shape)
    #打印输出结果的最后五个元素
    # print("output last five elements:",output[-5:])
    #检查数值是否一致,误差为1e-2
    if np.allclose(result_truth,output,atol=1e-2):
        print("结果一致")
    else:   
        print("结果不一致")

if __name__ == "__main__":

    #读取真实结果
    result_truth = read_matrix_from_file("./Matrix_C_truth.txt")

    #3.读取VDSP计算结果MatrixC
    MatrixC_file_path = "./MatrixC_output.txt"
    MatrixC = read_matrix_from_file(MatrixC_file_path)

    #4.检查结果是否一致
    check_result(result_truth,MatrixC)






