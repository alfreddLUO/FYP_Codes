import numpy as np
import itertools
from hmmlearn import hmm

from sklearn.preprocessing import StandardScaler

# 加载Grad-CAM生成的saliency map数据
import numpy as np
import itertools
from hmmlearn import hmm

def calculate_transition_probability(region1, region2):
    # Define your own logic to calculate transition probabilities based on spatial relationships
    return 0.5  # Placeholder value

# Load the saliency map
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")
print(saliency_map.shape)

# Step 1: Sort the saliency map based on the weights in descending order
sorted_saliency_map = np.sort(saliency_map.flatten())[::-1]

# Step 2: Divide the sorted saliency map into three equal-sized regions
num_regions = 3
region_size = len(sorted_saliency_map) // num_regions
regions = np.split(sorted_saliency_map, num_regions)

# Step 3: Assign each region to a hidden state in the HMM
num_states = num_regions
hidden_states = np.arange(num_states)

# Step 4: Define the transition probabilities between the hidden states based on the spatial relationships between the regions
transition_matrix = np.zeros((num_states, num_states))
for i, j in itertools.product(range(num_states), repeat=2):
    if i != j:
        # Define your own logic to calculate transition probabilities based on spatial relationships
        transition_matrix[i, j] = calculate_transition_probability(regions[i], regions[j])

# Step 5: Set the emission probabilities for each hidden state based on the saliency values within the corresponding region
emission_probabilities = np.zeros((num_states, region_size))
for i, region in enumerate(regions):
    region_saliency_values = region[:region_size]  # Choose the top region_size saliency values
    emission_probabilities[i, :len(region_saliency_values)] = region_saliency_values

# Step 6: Train the HMM using an appropriate algorithm (e.g., Baum-Welch algorithm) to estimate the model parameters
training_data = np.random.randint(0, 10, size=(100, num_regions * region_size))  # Placeholder training data (integer symbols)
training_data = training_data.astype(int)  # Convert training data to integers
model = hmm.CategoricalHMM(n_components=num_states)
model.transmat_ = transition_matrix
model.emissionprob_ = emission_probabilities
model.fit(training_data)  # Provide your own training data

# Step 7: Predict the hidden states (regions) for new saliency maps based on their weights and spatial relationships
new_saliency_map = np.random.rand(412, 263, 3)  # Placeholder new saliency map
sorted_new_saliency_map = np.sort(new_saliency_map.flatten())[::-1]
new_regions = np.split(sorted_new_saliency_map, num_regions)


print(emission_probabilities.shape)
print(new_regions[0][:region_size].shape)
new_regions_indices = []
print(num_regions)
print(new_regions)
new_regions_indices = []
for i in range(num_regions):
    region = new_regions[i][:region_size]
    if len(region) > 0:
        # Find the index only if the region is not empty
        indices = np.where(emission_probabilities[i] == region)[0]
        if len(indices) > 0:
            new_regions_indices.append(indices[0])
print(new_regions_indices)
new_regions_indices = np.array(new_regions_indices, dtype=int)  # Convert to integer type
print(new_regions_indices)

new_regions_indices = np.reshape(new_regions_indices, (-1, 1))  # Reshape to (num_regions, 1)
predicted_states = model.predict(new_regions_indices)

print(predicted_states)

