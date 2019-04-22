import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import plot_data as pd
import visualize_boundary as vb
import gaussian_kernel as gk


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and
# plot the data

print('Loading and Visualizing data ... ')

# Load from ex6data1:
data = scio.loadmat('ex6data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Training Linear SVM =====================
# The following code will train a linear SVM on the dataset and plot the
# decision boundary learned
#

print('Training Linear SVM')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)

c = 1000
clf = svm.SVC(c, kernel='linear', tol=1e-3)
clf.fit(X, y)

pd.plot_data(X, y)
vb.visualize_boundary(clf, X, 0, 4.5, 1.5, 5)

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Implementing Gaussian Kernel =====================
# You will now implement the Gaussian kernel to use
# with the SVM. You should now complete the code in gaussianKernel.py
#



