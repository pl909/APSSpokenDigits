import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to read blocks from the file
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


blocks = read_blocks('Train_Arabic_Digit.txt')

# Code below graphs cepstral coefficient as a function of analysis window index for one cepstral coefficient and multiple blocks

plt.figure(figsize=(10, 8))


for block_number in range(10):
    block_data = blocks[block_number]
    plt.subplot(5, 2, block_number + 1)  
    plt.plot(block_data.iloc[:, 0])  
    plt.title(f'Block {block_number + 1}')

plt.tight_layout()
plt.show()


# The code below graphs a scatter plot between two cepstral coefficients for one block in "0"

block_number = 0  
block_data = blocks[block_number] 


x_coeff = 1  
y_coeff = 5  

sns.scatterplot(x=block_data.iloc[:, x_coeff], y=block_data.iloc[:, y_coeff])

plt.xlabel(f'Cepstral Coefficient {x_coeff + 1}')
plt.ylabel(f'Cepstral Coefficient {y_coeff + 1}')

plt.show()


# The code below graphs MFCCs for different different cepstral coefficients for a block of the spoken digit zero

plt.figure(figsize=(10, 8))
for i in range(13):
    plt.subplot(5, 3, i+1)  
    plt.plot(block_data.iloc[:, i])
    plt.title(f'MFCC {i+1}')

plt.tight_layout()
plt.show()

'''
sns.pairplot(block_data)
sns.set_context("talk")
plt.show()
'''


