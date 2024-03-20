import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# Load the saliency map from the specified file path
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")
print(saliency_map.shape)
# Sample coordinates from the saliency map
sampled_coordinates = np.random.choice(saliency_map.shape[0], size=min(300, saliency_map.shape[0]), replace=False)

# Use the sampled coordinates to obtain the corresponding samples from the saliency map
samples = saliency_map[sampled_coordinates, :, :]

# Reshape the samples if needed
samples = samples.reshape(-1, 3)  # Or reshape to the appropriate shape

# Fit the GMM to the samples to estimate the GMM parameters
gmm_model = GaussianMixture(n_components=3)
gmm_model.fit(samples)
print(gmm_model)
# Access the estimated GMM parameters
estimated_means = gmm_model.means_
print("estimated means: ", estimated_means)
estimated_covariances = gmm_model.covariances_
print("estimated covariances: ", estimated_covariances)
estimated_weights = gmm_model.weights_
print("estimated weights: ", estimated_weights)

# Generate samples from the GMM
generated_samples, _ = gmm_model.sample(1000)  # Unpack the generated samples

# Plot the original samples
plt.scatter(samples[:, 0], samples[:, 1], label='Original Samples')

# Plot the generated samples
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label='Generated Samples')

# Plot the GMM component means
plt.scatter(gmm_model.means_[:, 0], gmm_model.means_[:, 1], c='red', marker='x', label='GMM Means')

plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Visualization')
plt.show()