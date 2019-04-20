import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
# import linearRegCostFunction as lrcf
# import trainLinearReg as tlr
# import learningCurve as lc
# import polyFeatures as pf
# import featureNormalize as fn
# import plotFit as plotft
# import validationCurve as vc


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and pot
# the data.
#

# Load Training data
print('Loading and Visualizing data ...')

# Load from ex5data1:
data = scio.loadmat('ex5data1.mat')
X = data['X']
y = data['y'].flatten()
Xval = data['Xval']
yval = data['yval'].flatten()
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

m = y.size

# Plot training data
plt.figure()
plt.scatter(X, y, c='r', marker="x")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')

input('Program paused. Press ENTER to continue')


