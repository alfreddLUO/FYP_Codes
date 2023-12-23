import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# 加载Grad-CAM生成的saliency map数据
# 加载saliency map数据
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")
# 查看数组的形状和元素值
print("数组形状:", saliency_map.shape)
print("数组元素值范围:", saliency_map.min(), "-", saliency_map.max())
# 可视化saliency map
plt.imshow(saliency_map, cmap="hot")
plt.title("Saliency Map")
plt.colorbar()
# plt.show()
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")

print("hello")

# 将saliency map转化为观测序列
saliency_map_2d = saliency_map.reshape(-1, 3)
observations = saliency_map_2d[:, 0]
# 训练高斯HMM模型
hmm_model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
n_samples=100
sampled_observations = np.random.choice(list(observations), size=n_samples, replace=False)
sampled_observations = np.reshape(sampled_observations,(-1,2))
print("hello")

hmm_model.startprob_ = np.array([0.7, 0.3])

hmm_model.fit(sampled_observations)
print("hello")
observations = np.reshape(observations,(-1,2))
observations = np.nan_to_num(observations, nan=0.0)
# 预测观测序列的隐藏状态

hidden_states = hmm_model.predict(observations)
print("hello")

# 可视化隐藏状态和saliency map
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(hidden_states, color="blue")
plt.title("Hidden States")
plt.subplot(2, 1, 2)
plt.imshow(saliency_map, cmap="hot")
plt.title("Saliency Map")
plt.show()