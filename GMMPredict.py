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



def compute_block_likelihood(block, gmm):
    # Compute the log likelihood of each frame in the block for a given GMM
    log_likelihoods = [gmm.score_samples([frame])[0] for frame in block]
    # Sum the log likelihoods to get the total log likelihood of the block
    total_log_likelihood = np.sum(log_likelihoods)
    return total_log_likelihood

gmm_components = {
    0: 3, 1: 3, 2: 5, 3: 4, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3
}

def plot_mfcc_gmm_overlay(test_1data, test_1labels, predicted_1labels, gmms1, digit=7):
    test_data = test_1data
    test_labels = test_1labels
    predicted_labels = predicted_1labels
    gmms = gmms1
    # Find the index of the first block of the specified digit in the test set
    block_indices = [i for i, label in enumerate(test_labels) if label == digit]
    first_block_index = block_indices[0]  # Get the first occurrence

    # Extract the first block of data for the specified digit
    first_block_data = np.array(test_1data[first_block_index])

    # Get the GMM for the specified digit
    gmm = gmms[digit]

    # Predict the label for the first block
    block_likelihoods = [compute_block_likelihood(first_block_data, gmm) for gmm in gmms]
    predicted_digit = np.argmax(block_likelihoods)
    correct_label = predicted_digit == test_labels[first_block_index]

    # Create scatter plots for MFCC comparisons
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mfcc_pairs = [(0, 1), (1, 2), (0, 2)]
    for i, (mfcc_x, mfcc_y) in enumerate(mfcc_pairs):
        ax = axes[i]
        ax.scatter(first_block_data[:, mfcc_x], first_block_data[:, mfcc_y], c='blue', label='MFCC Data')
        
        # Create a meshgrid for the contour plots
        x = np.linspace(first_block_data[:, mfcc_x].min(), first_block_data[:, mfcc_x].max(), num=100)
        y = np.linspace(first_block_data[:, mfcc_y].min(), first_block_data[:, mfcc_y].max(), num=100)
        X, Y = np.meshgrid(x, y)
        XX = np.column_stack([X.ravel(), Y.ravel()])
        
        # Pad the XX array with zeros to match the number of features the GMM expects
        XX_padded = np.pad(XX, ((0, 0), (0, 8)), 'constant', constant_values=0)
        
        # Compute the score samples
        Z = -gmm.score_samples(XX_padded).reshape(X.shape)
        
        # Plot GMM contours
        ax.contour(X, Y, Z, levels=14, linewidths=0.5, colors='red')

        # ... (the rest of the plotting code remains unchanged) ...

    plt.tight_layout()
    plt.show()

# Example usage:
# This assumes that 'test_data', 'test_labels', 'predicted_labels', and 'gmms' are already defined.
# plot_mfcc_gmm_overlay(test_data, test_labels, predicted_labels, gmms, digit=7)


# Load the training dataset
file_path = 'Train_Arabic_Digit.txt'
data, labels = load_data_flatten(file_path)

# Initialize a list to store GMMs for each digit
gmms = []

# Train a GMM for each digit with specified number of components
for digit in range(10):
    digit_data = data[np.array(labels) == digit]
    n_components = gmm_components[digit]
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(digit_data)
    gmms.append(gmm)

# Load the test dataset
test_data, test_labels = load_data('Test_Arabic_Digit.txt')


# Predict labels for each block in the test set
predicted_labels = []
for block in test_data:
    block_likelihoods = [compute_block_likelihood(block, gmm) for gmm in gmms]
    predicted_digit = np.argmax(block_likelihoods)
    predicted_labels.append(predicted_digit)

plot_mfcc_gmm_overlay(test_data, test_labels, predicted_labels, gmms)

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

# Set the 'Digit' column as the index
metrics_df.set_index('Digit', inplace=True)

print(metrics_df)

all_digits = np.arange(10)
conf_matrix = confusion_matrix(test_labels, predicted_labels)
# Plotting the confusion matrix with annotations for all cells
# Plot using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for E-M GMM Classifier Testing Set")
plt.savefig("confusionmatrixGMM")
plt.show()

