import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plot_data import *
import cost_function_reg as cfr
import plot_decision_boundary as pdb
import predict as predict
import map_feature as mf


plt.ion()
# Load data
# The first two columns contain the exam scores and the third column contains the label.
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

plot_data(X, y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])

input('Program paused. Press ENTER to continue')

# ===================== Part 1: Regularized Logistic Regression =====================
# In this part, you are given a dataset with data points that are not
# linearly separable. However, you would still like to use logistic
# regression to classify the data points.

# To do so, you introduce more feature to use -- in particular, you add
# polynomial features to our data matrix (similar to polynomial regression)
#

# Add polynomial features

# Note that map_feature also adds a column of ones for us, so the intercept
# term is handled

X = mf.map_feature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lmd = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost, grad = cfr.cost_function_reg(initial_theta, X, y, lmd)



