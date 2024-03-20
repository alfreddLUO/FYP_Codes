import numpy as np
from sklearn.mixture import GaussianMixture

def learn_gmm(saliency_map, n_components):
    print("saliency_map shape:", saliency_map.shape)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(saliency_map)
    likelihoods = gmm.score_samples(saliency_map)
    print("likelihoods shape:", likelihoods.shape)
    return gmm

def estimate_transition_matrix(saliency_map, gmm):
    n_components = gmm.n_components
    n_samples = saliency_map.shape[0]
    transition_matrix = np.zeros((n_components, n_components))

    # Compute likelihoods for each sample
    likelihoods = gmm.score_samples(saliency_map)

    for i in range(1, n_samples):
        prev_likelihood = likelihoods[i - 1]
        curr_likelihood = likelihoods[i]
        prev_likelihood = np.atleast_1d(prev_likelihood)
        curr_likelihood = np.atleast_1d(curr_likelihood)
        print("prev_likelihood shape:", prev_likelihood.shape)
        print("curr_likelihood shape:", curr_likelihood.shape)

        # Compute the change in accuracy for each ROI transition
        accuracy_increase = np.zeros((n_components, n_components))
        for j in range(n_components):
            for k in range(n_components):
                accuracy_increase[j, k] = curr_likelihood[k] - prev_likelihood[j]

        # Find the ROI transition that leads to the largest increase in accuracy
        max_accuracy_increase = np.max(accuracy_increase)
        max_accuracy_increase_idx = np.unravel_index(np.argmax(accuracy_increase), accuracy_increase.shape)

        # Update the transition matrix
        transition_matrix[max_accuracy_increase_idx] += 1

    # Normalize the transition matrix
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

    return transition_matrix

def construct_hmm(saliency_map, n_components):
    gmm = learn_gmm(saliency_map, n_components)
    transition_matrix = estimate_transition_matrix(saliency_map, gmm)

    # Create the HMM model
    hmm_model = {
        'gmm': gmm,
        'transition_matrix': transition_matrix
    }

    return hmm_model

# Example usage
# Load the saliency map from the specified file path
saliency_map = np.load("/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData/10000.npy")
# saliency_map = saliency_map.reshape(-1, saliency_map.shape[-1])
# saliency_map = saliency_map.reshape(-1, 1)
saliency_map = saliency_map.reshape(-1, saliency_map.shape[-1])

n_components = 3  # Number of components for the GMM

hmm_model = construct_hmm(saliency_map, n_components)

print("HMM Model:")
print(hmm_model)