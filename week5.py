#!/usr/bin/python

# Try to open imports
try:
    import sys
    import random
    import math
    import os
    import time
    import numpy as np
	import pandas as pd
	from random import sample 
    from matplotlib import pyplot as plt
    import matplotlib.colorbar
	from sklearn import datasets
	from sklearn.model_selection import StratifiedShuffleSplit
	from sklearn.model_selection import train_test_split
	from sklearn.neighbors import NearestNeighbors
	from sklearn.neighbors import KDTree
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import confusion_matrix	
    from sklearn.cross_validation import train_test_split
	from sklearn.decompostion import PCA as sklearnPCA
	from sklearn.model_selection import cross_val_score
	from scipy import stats
	from sklearn.metrics import mean_squared_error
	from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

# Error when importing
except ImportError:
    print('### ', ImportError, ' ###')
    # Exit program
    exit()
	
#Simple Random Smapling 
def srs(d, n):
      for i in range(n, n - 2):
	     j = numpy.random
		 d[i] = d[j]
	 s = numpy.random.permutationd(d[0:4])
	 return s
#Unifrom Simple Random Smapling
def srsUnifrom(a, b):
   o = random.uniform(a, b)
    return o
#Stratfied Sampling 		
def strafiedSampling(r):	
	o = StratifiedShuffleSplit(n_splits = 5, test_size = 0.5, random_state = r)
    return o

#Distance 
def euclideandistance2d(v1, v2):
    xd = v1.x - v2.x
    yd = v1.y - v2.y
    # return the distance
    return int(math.sqrt(xd * xd + yd * yd))

	
# Read input
def read():
	# Week 5
	X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
	y = np.array([0, 0, 0, 1, 1, 1])
	random_state = 1
	ss = strafiedSampling(random_state)
	ss.get_n_splits(X, y)
	print(sss)
	y_pred = KMeans(n_clusters=2, random_state = random_state).fit_predict(X)
	plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")
	transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters=3, random_state = random_state).fit_predict(X_aniso)
	plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c = y_pred)
	transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters  3, random_state = random_state).fit_predict(X_aniso)
    plt.subplot(222)
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c = y_pred)
	X_varied, y_varied = make_blobs(n_samples=n_samples,  cluster_std = [1.0, 2.5, 0.5], random_state = random_state)
    y_pred = KMeans(n_clusters = 3, random_state = random_state).fit_predict(X_varied)
	plt.subplot(223)
    plt.scatter(X_varied[:, 0], X_varied[:, 1], c = y_pred)
	X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3,
    random_state = random_state).fit_predict(X_filtered)	
    plt.subplot(224)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c = y_pred)
	plt.show()
				
	for train_index, test_index in sss.split(X, y):
	     print("TRAIN:", train_index, "TEST:", test_index)
         X_train = X[train_index]
		 X_test = X[test_index]
         y_train = y[train_index]
		 y_test = y[test_index]
     
	w = 4
    h = 3
    d = 70
    plt.figure(figsize = (w, h), dpi = d    		 
    color_map = plt.imshow(X)
    color_map.set_cmap("Blues_r")
    plt.colorbar()
    plt.savefig("out.png")

	
	
# Main
def main():
    # Read Input
    read()
    # Close Program
    exit()


# init
if __name__ == '__main__':
    # Begin
    main()


