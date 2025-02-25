#计算第一层卷积后的结果
#输入图片为../val/ILSVRC2012_val_00000001.JPEG
#权重数据为../weights/vgg16_weights_th_dim_ordering_th_kernels.h5
#输出结果为./block1_conv1_python_output.txt

#第一步读入图片,分辨率为224*224，通道为3
import cv2
import h5py
import torch
import torch.nn.functional as F

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    return image

# 提取第一层卷积层的权重和偏置
def extract_first_conv_layer_weights(h5_file):
    # VGG16 的第一层卷积层的权重和偏置路径
    weights_path = 'block1_conv1/block1_conv1_W:0'
    biases_path = 'block1_conv1/block1_conv1_b:0'

    if weights_path not in h5_file:
        raise ValueError(f"Weights '{weights_path}' not found in the HDF5 file.")
    if biases_path not in h5_file:
        raise ValueError(f"Biases '{biases_path}' not found in the HDF5 file.")

    # 提取权重和偏置
    weights = h5_file[weights_path][()]
    biases = h5_file[biases_path][()]

    return weights, biases

#把卷积结果（torch.tensor转化为float）写入txt
def write_conv_result(conv_result):
    with open('block1_conv1_python_output.txt', 'w') as f:
        for i in range(conv_result.shape[0]):
            for j in range(conv_result.shape[1]):
                for k in range(conv_result.shape[2]):
                    #把卷积结果（torch.tensor）转化为float
                    conv_result_float = float(conv_result[i, j, k])
                    #设置float小数点后最多三位
                    conv_result_float = round(conv_result_float, 4)
                    f.write(str(conv_result_float))
                    f.write(' ')
                f.write('\n')
    print("conv_result has been written to block1_conv1_python_output.txt")
    f.close()

if __name__ == "__main__":
    # 加载 HDF5 文件,读入权重和偏置
    h5_file_path = "../weights/vgg16_weights_th_dim_ordering_th_kernels.h5"
    with h5py.File(h5_file_path, 'r') as h5_file:
        weights, biases = extract_first_conv_layer_weights(h5_file)
    #读入图片
    image_path = '../val/ILSVRC2012_val_00000001.JPEG'
    image = load_image(image_path)

    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)
    print("image shape:",image.shape)
    weights = torch.tensor(weights, dtype=torch.float32)
    biases = torch.tensor(biases, dtype=torch.float32)
    print("weights shape:",weights.shape)
    print("biases shape:",biases.shape)
    #调用torch的卷积函数计算第一层的卷积结果
    conv_result = F.conv2d(image, weights, bias=biases, stride=1, padding=1)
    print("conv_result shape:",conv_result.shape)
    #把卷积结果写入txt
    write_conv_result(conv_result)

