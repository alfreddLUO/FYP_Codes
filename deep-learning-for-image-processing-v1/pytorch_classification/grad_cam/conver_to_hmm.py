import numpy as np
from hmmlearn import hmm
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm




def convert_saliency_to_hmm(saliency_path):
    # 加载saliency数据
    saliency_data = np.load(saliency_path)
    print(saliency_data.shape)
    # 将saliency数据转化为观测序列
    # observations = saliency_data.flatten().reshape(-1, 1)
    observations = np.reshape(saliency_data, (-1, 3)).flatten()
    n_samples=100

    sampled_observations = np.random.choice(list(observations), size=n_samples, replace=False)

    print(sampled_observations.shape)
    # 创建GaussianHMM模型
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, init_params="")

    # 手动设置初始概率
    model.startprob_ = np.array([0.7, 0.3])

    res = np.isnan(model.startprob_).any()
    print(res)
    model.startprob_ = np.nan_to_num(model.startprob_, nan=0.0)
    model.startprob_ /= np.sum(model.startprob_)
    print(model.startprob_)
    restored_observations = np.reshape(sampled_observations, (-1,2))
    restored_observations = np.nan_to_num(restored_observations, nan=0.0)
    # 使用观测序列训练GaussianHMM模型
    model.fit(restored_observations)

    # 预测隐藏状态序列
    hidden_states = model.predict(sampled_observations)

    return hidden_states

def visualize_hidden_states(hidden_states):
    # 创建时间步列表
    time_steps = np.arange(len(hidden_states))

    # 绘制隐藏状态序列
    plt.plot(time_steps, hidden_states, color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Hidden State')
    plt.title('Hidden States')
    plt.show()

def convert_all_saliency_to_hmm(saliency_dir, output_dir):
    # 获取目录下所有saliency数据文件的路径
    saliency_files = glob.glob(os.path.join(saliency_dir, '*.npy'))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个saliency数据文件
    for saliency_file in tqdm(saliency_files[0:2]):
        # 转化为GaussianHMM
        hidden_states = convert_saliency_to_hmm(saliency_file)

        # 可视化隐藏状态序列
        visualize_hidden_states(hidden_states)

        # 生成输出文件路径
        output_file = os.path.join(output_dir, os.path.basename(saliency_file))

        # 保存隐藏状态序列
        np.save(output_file, hidden_states)


# 示例使用
input_dir = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData'
output_dir = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/GeneratedHMM'

convert_all_saliency_to_hmm(input_dir, output_dir)