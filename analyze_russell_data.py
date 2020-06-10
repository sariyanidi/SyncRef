#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:32:04 2019

@author: sariyanide
"""

import sys
import os
sys.path.append(os.getcwd()+'/modules')
from syncref_pca import SyncRefPCA
import matplotlib.pyplot as plt
import numpy as  np
import copy

corr_threshold = 0.85
time_length = 30

# Names of 1905 companies with (almost no) NaNs
company_names = np.load('data/company_names.npy',  allow_pickle=True)

results_dir = 'results/corr%0.2f_tpoints%02d' % (corr_threshold, time_length)
os.makedirs(results_dir, exist_ok=True)

Xc_orig = np.load('data/russel_data.npy', allow_pickle=True)

# Number of time points
T = Xc_orig.shape[0]

# Analyze each month separately
for tidx, time_begin in enumerate(range(0,T,time_length)):
    print('Working on "month"# %d' % tidx)
    png_path = '%s/sync_%03d.png' % (results_dir, tidx)
    txt_path = '%s/sync_%03d.txt' % (results_dir, tidx)

    if os.path.exists(png_path):
        print('already processed -- skipping')
        continue
    
    Xc = copy.deepcopy(Xc_orig)
    Xc = Xc[time_begin:(time_begin+time_length),:]
    
    Xcn = copy.deepcopy(Xc)
    
    # The input to SyncRef must be z-normalized sequences
    for s in range(0, Xcn.shape[1]):
        x = Xcn[:,s]
        Xcn[:,s] = (x-x.mean())/(x.std()+1e-6)
    
    Spca = SyncRefPCA(corr_threshold=corr_threshold,  num_clusters_to_search=4, Kfull=np.min([24, Xc.shape[0], time_length]), K=4)
    # obj = Spca.get_synchrony(Xcn, one_per_window=True)
    (obj, time_elapsed)  = Spca.get_synchrony(Xcn, one_per_window=True)
    
    print('Found %d companies (processing time: %.2f secs)' % (len(obj[0]), time_elapsed))
    
    plt.clf()
    for i in obj[0]:
        x = Xcn[:,i]
        plt.plot(x, color='b', alpha=0.1)
    # Save company names
    companies_ = sorted(company_names[obj[0]].tolist())
    np.savetxt(txt_path, companies_, delimiter="\n", fmt="%s")
    plt.savefig(png_path)
