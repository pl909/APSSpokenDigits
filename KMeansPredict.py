import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def load_data_flatten(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    labels = []
    digit = 0
    current_block = []
    block_count = 0

    for line in lines:
        if line.strip():  # If the line is not blank
            features = list(map(float, line.split()))[:10]  # Take only first 10 coefficients
            current_block.append(features)
        else:  # End of a block
            if current_block:
                data.extend(current_block)  # Flatten the block into individual frames
                labels.extend([digit] * len(current_block))  # Extend labels
                current_block = []
                block_count += 1

            if block_count == 660:  # 660 blocks per digit
                digit += 1
                block_count = 0

    # Check if the last block is processed
    if current_block:
        data.extend(current_block)
        labels.extend([digit] * len(current_block))

    return np.array(data), np.array(labels)

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    labels = []
    digit = 0
    current_block = []
    block_count = 0  # To keep track of the number of blocks

    for line in lines:
        if line.strip():  # If the line is not blank
            features = list(map(float, line.split()))[:10]  # Take only first 10 coefficients
            current_block.append(features)
        else:  # End of a block
            if current_block:
                data.append(current_block)
                labels.append(digit)
                current_block = []
                block_count += 1
            if block_count == 220:  # 660 blocks per digit
                digit += 1
                block_count = 0  # Reset the block count for the next digit

    # Check if the last block is processed
    if current_block:
        data.append(current_block)
        labels.append(digit)

    return data, labels


# Load the dataset
file_path = 'Train_Arabic_Digit.txt'
data, labels = load_data_flatten(file_path)

print("Loaded data size:", data.shape)  # Debug statement

# Number of clusters (and components for GMMs)
n_clusters = 4  # Adjust as needed

# Initialize lists to store means and covariances for each digit
means_list = []
covariances_list = []

gmm_components = {
    0: 3, 1: 3, 2: 5, 3: 4, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3
}


# Train KMeans and calculate covariances for each digit
for digit in range(10):
    digit_data = data[np.array(labels) == digit]
    nclusters = gmm_components[digit]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(digit_data)
    # Calculate mean vectors
    means = kmeans.cluster_centers_
    # Calculate covariance matrices
    covariances = []
    for cluster in range(n_clusters):
        cluster_data = digit_data[kmeans.labels_ == cluster]
        covariance = np.cov(cluster_data, rowvar=False)
        covariances.append(covariance)

    means_list.append(means)
    covariances_list.append(covariances)

# ... [Previous code sections remain unchanged] ...

def compute_block_likelihood(block, means, covariances):
    # Compute the likelihood of the block for given GMM parameters
    total_likelihood = 0
    for frame in block:
        frame_likelihoods = []
        for mean, cov in zip(means, covariances):
            try:
                likelihood = multivariate_normal(mean=mean, cov=cov).pdf(frame)
            except np.linalg.LinAlgError as e:
                likelihood = 0
            frame_likelihoods.append(likelihood)
        # Check for numerical stability
        
        
        total_likelihood += np.log(np.sum(frame_likelihoods) + 1e-9)  # Add a small value for numerical stability
    return total_likelihood

# ... [Rest of the code for testing and accuracy computation] ...


# Load the test dataset
test_data, test_labels = load_data('Test_Arabic_Digit.txt')

# Predict labels for each block
predicted_labels = []
for block in test_data:
    block_likelihoods = [compute_block_likelihood(block, means, covs) for means, covs in zip(means_list, covariances_list)]
    predicted_digit = np.argmax(block_likelihoods)
    predicted_labels.append(predicted_digit)

# Calculate accuracy
accuracy = np.mean(np.array(predicted_labels) == np.array(test_labels))
print(f'Overall Accuracy: {accuracy}')

precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, labels=np.arange(10))

# Create a DataFrame to hold the precision, recall, and F1-score for each digit
metrics_df = pd.DataFrame({
    'Digit': np.arange(10),
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1
})


"""
# Set the 'Digit' column as the index
metrics_df.set_index('Digit', inplace=True)

print(metrics_df)

all_digits = np.arange(10)
conf_matrix = confusion_matrix(test_labels, predicted_labels)
# Plotting the confusion matrix with annotations for all cells
# Plot using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for K-Means GMM Classifier Testing Set")
plt.savefig("confusionmatrixKmeans")
plt.show()
"""