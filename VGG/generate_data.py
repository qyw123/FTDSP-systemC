#把JPEG图片转化为224*224*3的矩阵
#import cv2
import numpy as np

def load_image(image_path):
    # 读取并调整大小
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    
    # 转换为RGB（因为OpenCV读取的是BGR格式）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为float32并归一化到[0,1]
    image = image.astype(np.float32) / 255.0
    
    # ImageNet数据集的标准化参数
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 标准化
    image = (image - mean) / std
    
    # 转换为channel first格式 (H,W,C) -> (C,H,W)
    image = image.transpose(2, 0, 1)
    
    return image

#查看权重数据文件txt的形状:(rows,cols)
def check_weight_shape(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        rows = len(lines)
        # 检查第一行的元素数量作为列数
        cols = len(lines[0].strip().split())
        print(f"权重矩阵形状: ({rows}, {cols})")
        return rows, cols


# 使用示例
if __name__ == "__main__":
    # image_path = './val/ILSVRC2012_val_00000001.JPEG'
    # image = load_image(image_path)
    # print(f"Image shape: {image.shape}")  # 应该是(3, 224, 224)
    # print(f"Value range - Min: {image.min():.3f}, Max: {image.max():.3f}")
    
    # # 保存预处理后的数据
    # with open('image_1.txt', 'w') as f:
    #     for i in range(image.shape[0]):  # 遍历通道
    #         for j in range(image.shape[1]):  # 遍历高度
    #             for k in range(image.shape[2]):  # 遍历宽度
    #                 f.write(f"{image[i, j, k]:.6f} ")
    #             f.write('\n')
    weight_path = './weights/fc1_weights.txt'
    check_weight_shape(weight_path)