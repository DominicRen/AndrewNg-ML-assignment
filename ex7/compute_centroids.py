import numpy as np


def compute_centroids(X, idx, K):

    (m, n) = X.shape
    centroids = np.zeros((K, n))

    # ===================== Your Code Here =====================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids[i]
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #

    for k in range(K):
        x_for_centroid_k = X[np.where(idx == k)]
        centroid_k = np.sum(x_for_centroid_k, axis=0) / x_for_centroid_k.shape[0]
        centroids[k] = centroid_k

    return centroids
