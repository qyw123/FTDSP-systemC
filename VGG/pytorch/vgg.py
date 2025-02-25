import cv2
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_image(image_path):
    # 读取并预处理图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为float32并归一化
    image = image.astype(np.float32) / 255.0
    # 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    # 转换为CHW格式
    image = image.transpose(2, 0, 1)
    # 添加batch维度
    image = np.expand_dims(image, 0)
    # 确保返回float32类型的tensor
    return torch.from_numpy(image).float()  # 显式转换为float

def load_weights(h5_file_path):
    weights = {}
    with h5py.File(h5_file_path, 'r') as f:
        # 卷积层
        conv_layers = [
            'block1_conv1', 'block1_conv2',
            'block2_conv1', 'block2_conv2',
            'block3_conv1', 'block3_conv2', 'block3_conv3',
            'block4_conv1', 'block4_conv2', 'block4_conv3',
            'block5_conv1', 'block5_conv2', 'block5_conv3'
        ]
        # 全连接层
        fc_layers = ['fc1', 'fc2', 'predictions']
        
        # 加载卷积层权重
        for layer in conv_layers:
            weights[f'{layer}_W'] = torch.from_numpy(f[f'{layer}/{layer}_W:0'][()]).float()
            weights[f'{layer}_b'] = torch.from_numpy(f[f'{layer}/{layer}_b:0'][()]).float()
            
        # 加载全连接层权重并转置
        for layer in fc_layers:
            # 转置全连接层的权重矩阵
            weights[f'{layer}_W'] = torch.from_numpy(f[f'{layer}/{layer}_W:0'][()]).float().t()
            weights[f'{layer}_b'] = torch.from_numpy(f[f'{layer}/{layer}_b:0'][()]).float()
            
            # 打印权重形状以便调试
            print(f"{layer} weights shape: {weights[f'{layer}_W'].shape}")
            print(f"{layer} bias shape: {weights[f'{layer}_b'].shape}")
    
    return weights

def vgg16_forward(x, weights):
    # Block 1
    x = F.conv2d(x, weights['block1_conv1_W'], weights['block1_conv1_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block1_conv2_W'], weights['block1_conv2_b'], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    # Block 2
    x = F.conv2d(x, weights['block2_conv1_W'], weights['block2_conv1_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block2_conv2_W'], weights['block2_conv2_b'], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    # Block 3
    x = F.conv2d(x, weights['block3_conv1_W'], weights['block3_conv1_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block3_conv2_W'], weights['block3_conv2_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block3_conv3_W'], weights['block3_conv3_b'], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    # Block 4
    x = F.conv2d(x, weights['block4_conv1_W'], weights['block4_conv1_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block4_conv2_W'], weights['block4_conv2_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block4_conv3_W'], weights['block4_conv3_b'], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    # Block 5
    x = F.conv2d(x, weights['block5_conv1_W'], weights['block5_conv1_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block5_conv2_W'], weights['block5_conv2_b'], padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights['block5_conv3_W'], weights['block5_conv3_b'], padding=1)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    # 展平
    x = x.view(x.size(0), -1)

    # 全连接层
    x = F.linear(x, weights['fc1_W'], weights['fc1_b'])
    x = F.relu(x)
    x = F.linear(x, weights['fc2_W'], weights['fc2_b'])
    x = F.relu(x)
    x = F.linear(x, weights['predictions_W'], weights['predictions_b'])
    
    # 输出softmax概率
    x = F.softmax(x, dim=1)
    
    return x

def write_predictions(predictions, output_file):
    # 先将tensor移到CPU，再转换为numpy数组
    predictions = predictions.cpu().detach().numpy()
    np.savetxt(output_file, predictions, fmt='%.6f')
def load_class_names(synset_file, class_file):
    # 加载类别编号到类别描述的映射
    with open(synset_file, 'r') as f:
        synset_to_desc = {}
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                synset_to_desc[parts[0]] = parts[1]
    
    # 加载类别编号列表
    with open(class_file, 'r') as f:
        class_ids = [line.strip() for line in f]
    
    # 将类别编号映射到类别描述
    class_names = [synset_to_desc.get(class_id, "Unknown") for class_id in class_ids]
    return class_names

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图像
    image_path = '../val/ILSVRC2012_val_00000001.JPEG'
    image = load_image(image_path)
    
    # 加载权重
    h5_file_path = "../weights/vgg16_weights_th_dim_ordering_th_kernels.h5"
    weights = load_weights(h5_file_path)
    
    # 将数据和权重移到设备上
    image = image.to(device)
    for k, v in weights.items():
        weights[k] = v.to(device)
    
    # 前向传播
    with torch.no_grad():
        predictions = vgg16_forward(image, weights)
    
    # 保存预测结果
    write_predictions(predictions, './vgg16_predictions.txt')
    
    # 加载类别名称
    synset_file = "../imagenet_synsets.txt"
    class_file = "../imagenet_classes.txt"
    class_names = load_class_names(synset_file, class_file)
    
    # 打印 Top-5 预测结果
    probs, top5_indices = torch.topk(predictions[0], 5)
    print("Top 5 predictions:")
    for i, (prob, idx) in enumerate(zip(probs, top5_indices)):
        print(f"{i + 1}: {class_names[idx]} (probability: {prob:.4f})")