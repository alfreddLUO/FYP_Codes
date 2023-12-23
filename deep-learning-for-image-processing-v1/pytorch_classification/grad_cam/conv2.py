import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 加载Grad-CAM生成的saliency map数据
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")

# 将saliency map转化为观测序列
print(saliency_map)
saliency_map_2d = saliency_map.reshape(-1, 3)
# observations = saliency_map_2d[:, 0]
# 使用PCA将三维数据降维到二维
# pca = PCA(n_components=2)
# observations = pca.fit_transform(saliency_map)

observations = saliency_map_2d

# observations = saliency_map

# 训练高斯混合模型
gmm_model = GaussianMixture(n_components=3, covariance_type="diag")
# gmm_model.fit(observations.reshape(-1, 1))
gmm_model.fit(observations)

# 生成新的样本
n_samples = 100
generated_samples = gmm_model.sample(n_samples)[0]
print(gmm_model.sample(n_samples))

# 可视化生成的样本和原始数据
plt.figure(figsize=(10, 6))
plt.scatter(observations[:, 0], observations[:, 1], color='blue', alpha=0.5, label='Original Data')
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='red', alpha=0.5, label='Generated Samples')
plt.legend()
plt.title("Gaussian Mixture Model")
# plt.show()


transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
initial_state_probabilities = np.array([0.5, 0.5])
hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
hmm_model.startprob_ = initial_state_probabilities
hmm_model.transmat_ = transition_matrix
hmm_model.fit(observations)






# # 可视化生成的样本和原始数据
# plt.figure(figsize=(10, 6))
# # plt.scatter(observations[:, 0], observations[:, 1], color='blue', alpha=0.5, label='Original Data')
# plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='red', alpha=0.5, label='Generated Samples')
# plt.legend()
# plt.title("Gaussian Mixture Model")
# plt.show()





# # 可视化生成的样本和原始数据
# plt.figure(figsize=(10, 6))
# # plt.hist(observations, bins=50, alpha=0.5, label='Original Data')
# plt.hist(generated_samples, bins=50, alpha=0.5, label='Generated Samples')
# plt.legend()
# plt.title("Gaussian Mixture Model")
# plt.show()