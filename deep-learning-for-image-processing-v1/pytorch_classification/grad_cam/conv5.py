import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Load saliency map data generated by Grad-CAM
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")

# Define parameters for spatial segmentation
n_segments = 3  # Divide the saliency map into 3 segments
saliency_map = saliency_map.reshape(-1, 3)
segment_length = saliency_map.shape[0] // n_segments

# Split the saliency map data into multiple subsequences
subsequences = []
for i in range(n_segments):
    start_index = i * segment_length
    end_index = (i + 1) * segment_length
    subsequence = saliency_map[start_index:end_index]
    subsequences.append(subsequence)

# Create and fit multiple HMM models
hmm_models = []
for subsequence in subsequences:
    # Standardize the observation sequence
    scaler = StandardScaler()
    observations_normalized = scaler.fit_transform(subsequence)

    # Create and fit an HMM model
    hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
    hmm_model.fit(observations_normalized)

    hmm_models.append(hmm_model)

# Generate new samples
n_samples = 100
generated_samples_hmm = []
for hmm_model in hmm_models:
    # Generate new samples
    generated_samples_hmm_normalized, _ = hmm_model.sample(n_samples)
    generated_samples_hmm_subsequence = scaler.inverse_transform(generated_samples_hmm_normalized)
    generated_samples_hmm.append(generated_samples_hmm_subsequence)

# Visualize the generated samples and original data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for i, subsequence in enumerate(subsequences):
    plt.scatter(subsequence[:, 0], subsequence[:, 1], alpha=0.5, label=f"Original Data - Segment {i+1}")

for i, generated_samples_subsequence in enumerate(generated_samples_hmm):
    plt.scatter(generated_samples_subsequence[:, 0], generated_samples_subsequence[:, 1], alpha=0.5, label=f"Generated Samples - Segment {i+1}")

plt.legend()
plt.title("HMM Model - Spatial Segmentation")
plt.show()