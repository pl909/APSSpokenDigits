import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Read blocks function
def read_blocks(filename):
    blocks = []
    block_data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  
                block_data.append([float(value) for value in line.split()])
            elif block_data:  
                blocks.append(pd.DataFrame(block_data))
                block_data = []  
    if block_data:  
        blocks.append(pd.DataFrame(block_data))
    return blocks

# Function to perform K-Means clustering and fit a GMM for a given digit's data
def fit_gmm_for_digit(digit_data, num_components=3):
    mfcc_pairs = [(1, 2), (1, 3), (2, 3)]
    gmm_models = {}
    
    for mfcc1, mfcc2 in mfcc_pairs:
        # Perform K-Means clustering on the MFCC pair, fit GMM and store it.
        kmeans = KMeans(n_clusters=num_components, random_state=0).fit(digit_data[[mfcc1, mfcc2]])
        gmm = GaussianMixture(n_components=num_components, means_init=kmeans.cluster_centers_, random_state=0).fit(digit_data[[mfcc1, mfcc2]])
        gmm_models[(mfcc1, mfcc2)] = gmm
    
    return gmm_models

# Function to plot GMM contours
def plot_gmm_contours(gmm, mfcc1, mfcc2):
    x = np.linspace(-15, 15, 100)
    y = np.linspace(-15, 15, 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    
    # Calculate the score samples (log probability under the model)
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X.shape)
    
    # Plot contours
    plt.contour(X, Y, Z, levels=18, linewidths=1)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', zorder=3, marker='x', s=100, label='Centers')
    for i, (center_x, center_y) in enumerate(gmm.means_):
        plt.text(center_x, center_y, str(i), color='blue', fontsize=12, ha='center', va='center')
    plt.xlabel(f'MFCC {mfcc1}')
    plt.ylabel(f'MFCC {mfcc2}')
    plt.title(f'GMM Contour Plot for MFCC {mfcc1} vs MFCC {mfcc2}')

# Load  blocks
blocks = read_blocks('Train_Arabic_Digit.txt')

#  660 blocks per digit
num_digits = 10
blocks_per_digit = 660

# Organize the blocks by digit
digits_dataframes = [pd.concat(blocks[i*blocks_per_digit:(i+1)*blocks_per_digit], ignore_index=True) for i in range(num_digits)]

# Fit GMMs for each digit and plot the contour plots
for digit in range(num_digits):
    gmm_models = fit_gmm_for_digit(digits_dataframes[digit])
    plt.figure(figsize=(15, 5))
    for i, (mfcc_pair, gmm) in enumerate(gmm_models.items()):
        plt.subplot(1, 3, i + 1)
        plot_gmm_contours(gmm, *mfcc_pair)
    plt.suptitle(f'Digit {digit} - GMM Contour Plots')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
