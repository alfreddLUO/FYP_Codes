from hmmlearn import hmm
import numpy as np

# Load the saliency map from the specified file path
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")


# Initialize an empty list to store the sampled coordinates
sampled_coordinates = []

# Define the region size around each sample to mask out
region_size = 5

# Sample the coordinates from the saliency map
while True:
    # Sample a coordinate from the saliency map
    sampled_coordinate = np.random.choice(saliency_map.shape[0], size=2, replace=False)

    # Mask out the saliency map in the region around the sampled coordinate
    start_x = max(0, sampled_coordinate[0] - region_size)
    end_x = min(saliency_map.shape[0], sampled_coordinate[0] + region_size + 1)
    start_y = max(0, sampled_coordinate[1] - region_size)
    end_y = min(saliency_map.shape[1], sampled_coordinate[1] + region_size + 1)
    saliency_map[start_x:end_x, start_y:end_y] = 0

    # Append the sampled coordinate to the list
    sampled_coordinates.append(sampled_coordinate)

    # Break the loop if the desired number of samples is reached
    if len(sampled_coordinates) == 1000:  # Adjust the desired number of samples as needed
        break

# Convert the list of sampled coordinates to a numpy array
sampled_coordinates = np.array(sampled_coordinates)

# Fit the HMM model to the sampled coordinates
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.fit(sampled_coordinates)

# Predict the hidden states for the sampled coordinates
hidden_states = model.predict(sampled_coordinates)

print("Hidden States:", hidden_states)