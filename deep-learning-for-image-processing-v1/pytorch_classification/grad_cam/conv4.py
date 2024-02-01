import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# 加载Grad-CAM生成的saliency map数据
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")

print(saliency_map.shape)
# print(saliency_map)
flattened_map = saliency_map.reshape(-1, 3)  # Reshape to (412*263, 3)
weights = np.linalg.norm(flattened_map, axis=1)  # Compute weights based on Euclidean norm (or any other weight calculation)
sorted_indices = np.argsort(weights)[::-1]
sorted_saliency = flattened_map[sorted_indices]
# print(sorted_saliency)
# Divide the sorted saliency map into three regions
total_pixels = sorted_saliency.shape[0]
region_size = total_pixels // 3

regions = [sorted_saliency[i:i+region_size] for i in range(0, total_pixels, region_size)]

# Printing the divided regions
for i, region in enumerate(regions):
    print(f"Region {i+1}: {region}")


# Step 3: Assign regions to hidden states
num_states = 3
region_assignments = np.zeros(total_pixels, dtype=int)  # Initialize region assignments

for i, region in enumerate(regions):
    region_assignments[sorted_indices[i:i+region_size]] = i  # Assign each pixel in the region to the corresponding hidden state

# Step 4: Define transition probabilities based on spatial relationships
transition_matrix = np.zeros((num_states, num_states))  # Initialize transition matrix

# Define spatial relationships between regions
spatial_relationships = [[0, 0.5, 0.5],
                         [0.5, 0, 0.5],
                         [0.5, 0.5, 0]]

for i in range(num_states):
    for j in range(num_states):
        if i != j:
            transition_matrix[i, j] = spatial_relationships[i][j]

print("Region Assignments:")
print(region_assignments)

print("\nTransition Matrix:")
print(transition_matrix)

# Step 5: Set emission probabilities based on saliency values
emission_probabilities = np.zeros((num_states, region_size))  # Initialize emission probabilities

for i, region in enumerate(regions):
    region_saliency_values = saliency_map[region]  # Extract saliency values within the region
    region_indices = np.ravel_multi_index((region[:, 0], region[:, 1]), saliency_map.shape[:2])  # Convert region coordinates to linear indices
    flat_saliency_values = region_saliency_values.flatten()  # Flatten the saliency values
    region_size = len(flat_saliency_values)  # Get the size of the region
    emission_probabilities[i, :region_size] = flat_saliency_values[:region_size]  # Assign the flattened saliency values to the corresponding hidden state

# Step 6: Train the HMM
model = hmm.MultinomialHMM(n_components=num_states, n_iter=100)
model.fit(emission_probabilities)

# Step 7: Predict hidden states for new saliency maps
new_saliency_map = ...  # Load or generate new saliency map
new_emission_probabilities = np.zeros((num_states, total_pixels))  # Initialize emission probabilities for new saliency map

for i, region in enumerate(regions):
    region_saliency_values = new_saliency_map[region]  # Extract saliency values within the region
    region_indices = np.ravel_multi_index((region[:, 0], region[:, 1]), new_saliency_map.shape[:2])  # Convert region coordinates to linear indices
    flat_saliency_values = region_saliency_values.flatten()  # Flatten the saliency values
    new_emission_probabilities[i, region_indices[:len(flat_saliency_values)]] = flat_saliency_values  # Assign the flattened saliency values to the corresponding hidden state

predicted_states = model.predict(new_emission_probabilities)

print("Predicted Hidden States:")
print(predicted_states)

# print(saliency_map.shape)
# sorted_indices = np.argsort(saliency_map, axis=None)[::-1]
# print(sorted_indices)
# sorted_saliency = saliency_map.flatten()[sorted_indices]
# print(sorted_saliency)

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