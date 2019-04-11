import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import display_data as dd
import lr_cost_function as lCF
# import oneVsAll as ova
# import predictOneVsAll as pova


plt.ion()

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 input images of Digits
num_labels = 10         # 10 labels, from 0 to 9
                        # Note that we have mapped "0" to label 10

# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))  # 随机排列序列
selected = X[rand_indices[0:100], :]

dd.display_data(selected)

input('Program paused. Press ENTER to continue')

# ===================== Part 2-a: Vectorize Logistic Regression =====================
# In this part of the exercise, you will reuse your logistic regression
# code from the last exercise. Your task here is to make sure that your
# regularized logistic regression implementation is vectorized. After
# that, you will implement one-vs-all classification for the handwritten
# digit dataset
#

# Test case for lr_cost_function
print('Testing lr_cost_function()')





