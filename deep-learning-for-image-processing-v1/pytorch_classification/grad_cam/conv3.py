import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载Grad-CAM生成的saliency map数据
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")

# 将saliency map转化为观测序列并归一化处理
observations = saliency_map.reshape(-1, 3)
observations_normalized = observations / np.max(observations)

# 训练高斯混合模型
gmm_model = GaussianMixture(n_components=3, covariance_type="diag")
gmm_model.fit(observations_normalized)

# 生成新的样本
n_samples = 100
generated_samples_normalized = gmm_model.sample(n_samples)[0]
generated_samples = generated_samples_normalized * np.max(observations)

# 可视化生成的样本和原始数据
plt.figure(figsize=(10, 6))
plt.scatter(observations[:, 0], observations[:, 1], color='blue', alpha=0.5, label='Original Data')
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='red', alpha=0.5, label='Generated Samples')
plt.legend()
plt.title("Gaussian Mixture Model")

transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
initial_state_probabilities = np.array([0.5, 0.5])

# 对观测序列进行标准化处理
scaler = StandardScaler()
observations_normalized = scaler.fit_transform(observations)

# 创建HMM模型并拟合
hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
hmm_model.startprob_ = initial_state_probabilities
hmm_model.transmat_ = transition_matrix
hmm_model.fit(observations_normalized)

# 生成新样本
generated_samples_hmm_normalized, _ = hmm_model.sample(n_samples)
generated_samples_hmm = scaler.inverse_transform(generated_samples_hmm_normalized)

# 可视化生成的样本和原始数据
plt.figure(figsize=(10, 6))
plt.scatter(observations[:, 0], observations[:, 1], color='blue', alpha=0.5, label='Original Data')
plt.scatter(generated_samples_hmm[:, 0], generated_samples_hmm[:, 1], color='red', alpha=0.5, label='Generated Samples (HMM)')
plt.legend()
plt.title("HMM Model")

plt.show()