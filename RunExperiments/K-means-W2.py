import numpy as np
from pathlib import Path
import cv2
import math
import matplotlib.pyplot as plt
from numpy import genfromtxt
from pathlib import Path
import random
import ot

# Load images
# Define path to the data directory
data_dir = Path('ShapeData')
all_dir = [f for f in data_dir.glob("*.*")]

# Reshape to size dim x dim (try smaller size)
dim = 50

all_img = []
# Process each img
for direc in all_dir:
    temp = plt.imread(direc, 0)
    #convert to greyscale
    if len(temp.shape)>2:
        temp = temp[:,:,2]
    temp = cv2.resize(temp, (dim,dim))
    # Normalization, so that each temp is a distribution
    temp = temp / np.sum(temp)
    temp = temp.flatten()
    all_img.append(temp)
# Shuffle lists
random.shuffle(all_img)

# cost matrix: dim^2 by dim^2
M = []
# dim same as above
for i in range(dim):
    for j in range(dim):
        for m in range(dim):
            for n in range(dim):
                M.append(math.pow(i-m,2)+math.pow(j-n,2))
M = np.reshape(M, (dim*dim,dim*dim))
M = np.array(M)
# Initialize k centroids of size 1 x dim^2, by picking k samples randomly
# Number of clusters
k = 9
all_img = np.array(all_img)
# Randomly pick k centroids from dataset
rdm_idx = np.random.choice(range(all_img.shape[0]), k, replace=False)
W = all_img[rdm_idx,:]

# assign clusters
def expectation(X, W):
    '''
    X: N x d instead
    W: k x d instead
    '''
    # Get shapes
    N, d = X.shape
    k, _ = W.shape
    
    # Compute Distance and assign cluster in one step
    cluster_id = [] # N x 1
    # Compute distances from each sample to each cluster using W-2 distance
    for i in range(N):
        minDist = float('inf')
        index = -1
        for j in range(k):
            tempdist = ot.emd2(X[i],W[j],M)
            if tempdist<minDist:
                minDist = tempdist
                index = j
        cluster_id.append(index)

    cluster_id = np.array(cluster_id)
    
    # Matrix of one-hot vectors
    Z = np.zeros((N, k))
    Z[range(N),cluster_id] = 1
    return Z

def maximization(X, Z):
    '''
    X: N x d
    Z: N x k
    '''
    # Get shapes
    N, d = X.shape
    _, k = Z.shape

    # Compute updated centroids
    W_new = np.zeros((k, d))
    for cl_id in range(k):
        W_new[cl_id, :] = np.mean(X[(Z[: ,cl_id] == 1), :], axis=0)
    return W_new

def run_kmeans(X, K, max_iter = 100, eps = 1e-5, log_interval = 2):

    rdm_idx = np.random.choice(range(X.shape[0]), K, replace=False)
    W = X[rdm_idx,:]
    Z = None

    # Expectation-maximization loop
    for n_iter in range(max_iter):
        # Store previous W and Z
        Z_prev = Z
        W_prev = W
        
        # Expectation followed by maximization step
        Z = expectation(X, W)
        W = maximization(X, Z)

        # Terminate if change in Z and W is small
        if Z_prev is not None and ((Z_prev - Z) ** 2).sum() < eps and ((W_prev - W) ** 2).sum() < eps:
            print('Done at iteration: ', n_iter + 1)
            rec_error = np.linalg.norm(X - Z @ W, axis=1).sum()
            print ('Reconstruction error ', n_iter + 1,":", rec_error)
            break

        # Print reconstruction loss every log_interval steps
        if (n_iter + 1) % log_interval == 0 or (n_iter + 1) == max_iter or n_iter == 0:
            rec_error = np.linalg.norm(X - Z @ W, axis=1).sum()
            print ('Reconstruction error at iteration ', n_iter + 1,":", rec_error)
        print(rec_error)
    return W
    
W = run_kmeans(all_img, K=9)
np.savetxt("K_means_W2_barycenters.csv", W, delimiter = ",")
