#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:32:04 2019

@author: sariyanidi
"""
import os
import sys
sys.path.append(os.getcwd()+'/modules')
from syncref_pca import SyncRefPCA
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import numpy as  np

X = np.loadtxt('demo_data.dat')        
(T, K) = X.shape

# The input to SyncRef must be z-normalized sequences
# Gaussian smoothing is also recommended
for s in range(0, X.shape[1]):
    x = gaussian_filter1d(X[:,s], 2)
    X[:,s] = (x-x.mean())/(x.std()+1e-6)

Kfull = np.min([24, T, K])

Spca = SyncRefPCA(corr_threshold=0.85, num_clusters_to_search=4, Kfull=Kfull)
(obj, time_elapsed) = Spca.get_synchrony(X, rand_point_ratio=0.005, one_per_window=True)

print('Processed in %.2f seconds' % (time_elapsed))

plt.clf()
for i in obj[0]:
    x = X[:,i]
    plt.plot(x, color='b', alpha=0.1)
plt.show()
