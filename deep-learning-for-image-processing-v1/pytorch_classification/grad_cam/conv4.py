import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# 加载Grad-CAM生成的saliency map数据
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")
# 定义空间分割的参数
n_segments = 3  # 将saliency map分割成3个部分
saliency_map = saliency_map.reshape(-1, 3)
segment_length = saliency_map.shape[0] // n_segments

# 分割saliency map数据为多个子序列
subsequences = []
for i in range(n_segments):
    start_index = i * segment_length
    end_index = (i + 1) * segment_length
    subsequence = saliency_map[start_index:end_index]
    subsequences.append(subsequence)

# 创建多个HMM模型并拟合
hmm_models = []
for subsequence in subsequences:
    # 对观测序列进行标准化处理
    scaler = StandardScaler()
    observations_normalized = scaler.fit_transform(subsequence)

    # 创建HMM模型并拟合
    hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
    hmm_model.fit(observations_normalized)

    hmm_models.append(hmm_model)

# 生成新样本
n_samples = 100
generated_samples_hmm = []
for hmm_model in hmm_models:
    # 生成新样本
    generated_samples_hmm_normalized, _ = hmm_model.sample(n_samples)
    generated_samples_hmm_subsequence = scaler.inverse_transform(generated_samples_hmm_normalized)
    generated_samples_hmm.append(generated_samples_hmm_subsequence)

# 可视化生成的样本和原始数据
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, subsequence in enumerate(subsequences):
    plt.scatter(subsequence[:, 0], subsequence[:, 1], alpha=0.5, label=f"Original Data - Segment {i+1}")

for i, generated_samples_subsequence in enumerate(generated_samples_hmm):
    plt.scatter(generated_samples_subsequence[:, 0], generated_samples_subsequence[:, 1], alpha=0.5, label=f"Generated Samples - Segment {i+1}")

plt.legend()
plt.title("HMM Model - Spatial Segmentation")
plt.show()