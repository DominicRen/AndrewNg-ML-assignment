import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plot_data import *
import cost_function as cf
# import plotDecisionBoundary as pdb
# import predict as predict
from sigmoid import *


plt.ion()
# Load data
# The first two columns contain the exam scores and the third column contains the label.
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

# ===================== Part 1: Plotting =====================
print('Plotting Data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plot_data(X, y)

plt.axis([30, 100, 30, 100])
plt.legend(['Admitted', 'Not admitted'], loc=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Compute Cost and Gradient =====================
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. You need to complete the code in
# costFunction.py

# Setup the data array appropriately, and add ones for the intercept term
(m, n) = X.shape

# Add intercept term
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = cf.cost_function(initial_theta, X, y)

np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})

print('Cost at initial theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')






