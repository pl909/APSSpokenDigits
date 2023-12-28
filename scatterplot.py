import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
            features = list(map(float, line.split()))
            current_block.append(features)
        else:  # End of a block
            if current_block:
                data.append(current_block)
                labels.append(digit)
                current_block = []
                block_count += 1

            if block_count == 660:  # 660 blocks per digit
                digit += 1
                block_count = 0  # Reset the block count for the next digit

    # Check if the last block is processed
    if current_block:
        data.append(current_block)
        labels.append(digit)

    return data, labels


file_path = 'Train_Arabic_Digit.txt'
data, labels = load_data(file_path)
start_index = 7 * 660
end_index = start_index + 660
data_digit_8 = [data[i] for i in range(start_index, end_index) if labels[i] == 7]

# Flatten the data for digit 8
flattened_data_digit_8 = [feature for sublist in data_digit_8 for feature in sublist]

# Convert the data to a DataFrame
df_digit_8 = pd.DataFrame(flattened_data_digit_8, columns=[f'MFCC_{i+1}' for i in range(13)])

# Selecting only the first 6 MFCCs for pair-wise scatter plot
df_subset_digit_8 = df_digit_8.iloc[:, :3]


# Using seaborn's PairGrid to customize the pair-wise plot
grid = sns.PairGrid(df_subset_digit_8)
grid = grid.map_upper(sns.scatterplot, alpha=0.5)
grid = grid.map_lower(sns.kdeplot, cmap="Blues_d")
grid = grid.map_diag(sns.kdeplot, lw=3, legend=False)

plt.savefig('lada.png')
plt.show()
