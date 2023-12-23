import os
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from tqdm import tqdm


def convert_to_hmm(file_path, output_dir):
    # 加载Grad-CAM saliency map文件
    saliency_map = np.load(file_path)

    # 将Grad-CAM saliency map转换为观测序列
    observations = saliency_map.flatten()

    # 创建HMM模型
    model = hmm.CategoricalHMM(n_components=2)  # 设置隐藏状态的数量为2

    # 训练HMM模型
    model.fit(observations.reshape(-1, 1))

    # 预测隐藏状态序列
    hidden_states = model.predict(observations.reshape(-1, 1))

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(saliency_map, cmap='hot')
    plt.title('Grad-CAM Saliency Map')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.scatter(range(len(hidden_states)), hidden_states, c=hidden_states, cmap='cool')
    plt.title('Hidden State Sequence')
    plt.xlabel('Time')
    plt.ylabel('Hidden State')
    plt.ylim([-0.5, 1.5])

    plt.tight_layout()

    # 保存可视化结果到新目录下
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name.replace('.npy', '.png'))
    plt.savefig(output_path)
    plt.close()

def main():
    # 指定目录和输出目录
    input_dir = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData'
    output_dir = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/GeneratedHMM'

    # 遍历目录下的所有.npy文件
    for file_name in tqdm(os.listdir(input_dir)[0:2]):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_dir, file_name)
            convert_to_hmm(file_path, output_dir)

if __name__ == '__main__':
    main()