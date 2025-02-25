#本代码用于生成数据
import numpy as np

#生成A矩阵,大小为12*384
A = np.random.randn(12, 384)
#另存为txt文件
np.savetxt("matrixA_input.txt", A, fmt="%f")
#生成B矩阵,大小为384*64
B = np.random.randn(384, 64)
#另存为txt文件
np.savetxt("matrixB_input.txt", B, fmt="%f")
#生成C矩阵,大小为12*64
C = np.random.randn(12, 64)
#另存为txt文件
np.savetxt("matrixC_input.txt", C, fmt="%f")
#生成Matrix_C_truth.txt
#C=A*B+C
C_truth = A@B+C 
#另存为txt文件
np.savetxt("Matrix_C_truth.txt", C_truth, fmt="%f")



