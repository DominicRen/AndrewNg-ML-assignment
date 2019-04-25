import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from skimage import io
from skimage import img_as_float
# import runkMeans as km
import find_closest_centroids as fc
# import computeCentroids as cc
# import kMeansInitCentroids as kmic


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Find Closest Centroids =====================
# To help you implement K-means, we have divided the learning algorithm
# into two functions -- find_closest_centroids and compute_centroids. In this
# part, you should complete the code in the findClosestCentroids.py
#

print('Finding closest centroids.')

# Load an example dataset that we will be using
data = scio.loadmat('ex7data2.mat')
X = data['X']

# Select an initial set of centroids
k = 3  # Three centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = fc.find_closest_centroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print('{}'.format(idx[0:3]))
print('(the closest centroids should be 0, 2, 1 respectively)')

input('Program paused. Press ENTER to continue')







