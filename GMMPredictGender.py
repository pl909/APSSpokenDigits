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
    gender = 0
    for line in lines:
        if line.strip():  # If the line is not blank
            features = list(map(float, line.split()))[:10]  # Take only first 10 coefficients
            features.append(gender)
            current_block.append(features)
        else:  # End of a block
            if current_block:
                data.extend(current_block)  # Flatten the block into individual frames
                labels.extend([digit] * len(current_block))  # Extend labels
                current_block = []
                block_count += 1
            if block_count == 330:
                gender = 1 - gender
            if block_count == 660:  # 660 blocks per digit
                digit += 1
                gender = 0
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
    gender = 0
    digit = 0
    current_block = []
    block_count = 0  # To keep track of the number of blocks

    for line in lines:
        if line.strip():  # If the line is not blank
            features = list(map(float, line.split()))[:10]  # Take only first 10 coefficients
            features.append(gender)
            current_block.append(features)
        else:  # End of a block
            if current_block:
                data.append(current_block)
                labels.append(digit)
                current_block = []
                block_count += 1
            if block_count == 110:
                gender = 1 - gender
            if block_count == 220:  # 660 blocks per digit
                digit += 1
                gender = 0
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
    0: 6, 1: 6, 2: 10, 3: 8, 4: 6, 5: 6, 6: 10, 7: 8, 8: 8, 9: 8
}


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
plt.title("Confusion Matrix for E-M GMM Classifier + Gender Testing Set")
plt.savefig("confusionmatrixGenderGMM")
plt.show()

