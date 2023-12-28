import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    labels = []
    digit = 0
    block_count = 0

    for line in lines:
        if line.strip():  # If the line is not blank
            features = list(map(float, line.strip().split()))
            data.append(features)
            labels.append(digit)
        else:  # End of a block
            block_count += 1
            if block_count == 660:  # 660 blocks per digit
                digit += 1
                block_count = 0

    return np.array(data), np.array(labels)

# Load the dataset
data, labels = load_data('Train_Arabic_Digit.txt')

# Standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# Initialize PCA
pca = PCA()

# Fit PCA on the standardized data
pca.fit(data_std)

# Calculate the cumulative sum of explained variance ratio
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Output the cumulative variance
print(cumulative_variance)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by Each Component')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)

# Save the plot to a file
plt.savefig('.png')