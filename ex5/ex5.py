import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import linear_reg_cost_function as lrcf
import train_linear_reg as tlr
import learning_curve as lc
# import polyFeatures as pf
# import featureNormalize as fn
# import plotFit as plotft
# import validationCurve as vc


plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and plot
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
plt.ylabel('Water flowing out of the dam (y)')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Regularized Linear Regression Cost =====================
# You should now implement the cost function for regularized linear regression
#

theta = np.ones(2)
cost, _ = lrcf.linear_reg_cost_function(theta, np.c_[np.ones(m), X], y, 1)

print('Cost at theta = [1  1]: {:0.6f}\n(this value should be about 303.993192'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Regularized Linear Regression Gradient =====================
# You should now implement the gradient for regularized linear regression
#

theta = np.ones(2)
cost, grad = lrcf.linear_reg_cost_function(theta, np.c_[np.ones(m), X], y, 1)

print('Gradient at theta = [1  1]: {}\n(this value should be about [-15.303016  598.250744]'.format(grad))

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Train Linear Regression =====================
# Once you have implemented the cost and gradient correctly, the
# train_linear_reg function will use your cost function to train regularzized linear regression.
#
# Write Up Note : The data is non-linear, so this will not give a great fit.
#

# Train linear regression with lambda = 0
lmd = 0

theta = tlr.train_linear_reg(np.c_[np.ones(m), X], y, lmd)

# Plot fit over the data
plt.plot(X, np.dot(np.c_[np.ones(m), X], theta))

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Learning Curve for Linear Regression =====================
# Next, you should implement the learning_curve function.
#
# Write up note : Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf
#

lmd = 0
error_train, error_val = lc.learning_curve(np.c_[np.ones(m), X], y, np.c_[np.ones(Xval.shape[0]), Xval], yval, lmd)

plt.figure()
plt.plot(np.arange(m), error_train, np.arange(m), error_val)
plt.title('Learning Curve for Linear Regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

input('Program paused. Press ENTER to continue')



