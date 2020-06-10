#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:40:38 2020

@author: v
"""

from math import exp
import numpy as np

def codebook_neighbours_inner(double [:,:] TH, long W, long [:,:] codebook):
    
    cdef long Nthresh = TH.shape[1]-1
    cdef long K = TH.shape[0]
    cdef long Ncodes = Nthresh**K
    cdef double sm, t1, T1, t2, T2, med
    cdef long c1, c2, k
    cdef double[:,:] cneighbours = np.zeros((Ncodes, Ncodes))
    cdef double[:,:] neighbour = np.zeros((Ncodes,1))

    for c1 in range(Ncodes):
        
        for c2 in range(Ncodes):
            sm = 0
            
            for k in range(K):
                t1 = TH[k, codebook[c1, k]]
                T1 = TH[k, codebook[c1, k]+1]
                t2 = TH[k, codebook[c2, k]]
                T2 = TH[k, codebook[c2, k]+1]
            
                med = (t2+T2)/2
                
                if t1 < med < T1:
                    med = 0
                else:
                    sm += min([abs(t1-t2),abs(t1-T2),abs(T1-t2),abs(T1-T2)])**2
                    
                    
            cneighbours[c2,c1] = np.sqrt(sm)/W
        
    return cneighbours
