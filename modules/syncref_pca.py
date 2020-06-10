import os
import numpy as np
import numpy.matlib
import pickle
import miniball 
import math

import scipy.stats
from sklearn.neighbors import KernelDensity
import scipy.spatial
import sys
sys.path.append(os.path.dirname(__file__) + '/cython_modules')
from cython_functions import codebook_neighbours_inner

from time import time

plot_em = False
use_part_vid = True
    
def dec2base(n, b):
    if not n:
        return []
    return dec2base(n//b, b) + [n%b]


def dec2base_padded(n, b, L):
    d = dec2base(n, b)
    
    if L <= len(d):
        return d
    
    pad = L-len(d)
    d = [0 for x in range(pad)] + d
    return d


def save_cpickle(fpath, o):
    f = open(fpath, 'wb')
    pickle.dump(o, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    
    
def load_cpickle(fpath):
    f = open(fpath, 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    
    return loaded_obj




def find_thresholds(Z, K, Nthresh):

    TH = []
    
    for k in range(K):
        u = Z[:,k]
        minu = np.min(u)
        maxu = np.max(u)
        pad = (maxu-minu)/5
        xi = np.linspace(minu-pad, maxu+pad, 1000)
        u = u.reshape(-1, 1)
        xi = xi.reshape(-1, 1)
        
        kde = KernelDensity(kernel='epanechnikov').fit(u)
        log_dens = kde.score_samples(xi)
        f = np.exp(log_dens)
        f /= np.sum(f)

        F = np.cumsum(f)
        
        # The first threshold is where the distribution "starts"
        thetas = []
        thetas.append(xi[np.argmin(np.abs(F-0.0001))][0])
        for t in range(Nthresh):
            idx = np.argmin(np.abs(F-(t+1)*(1./Nthresh)))
            thetas.append(xi[idx][0])
            
        TH.append(thetas)
    
    TH = np.matrix(TH)
    
    return TH



def compute_integer_code(c, TH):
    K = TH.shape[0]
    base = TH.shape[1]-1
    int_code = 0
    
    for k in range(K):
        for t in range(TH.shape[1]-1):
            if c[k] < TH[k, t+1]:
                int_code += (base**(K-(k+1)))*(t)
                break
            
    return int(int_code+1)



def codebook_neighbours(TH, W):
    
    Nthresh = TH.shape[1]-1
    K = TH.shape[0]
    Ncodes = Nthresh**K
    
#    
    TH[:,0] *= 100
    TH[:,-1] *= 100
    TH_MEANS = 0.5*(TH[:,:-1]+TH[:,1:])
    
    codebook = np.zeros((Ncodes, K), dtype=int)
    centroids = np.zeros((Ncodes, K))
    centroid_thresholds = []
    
    for c in range(Ncodes):
        centroid_threshold = np.zeros((2,K))
        code = dec2base_padded(c, Nthresh, K)
        
        for k in range(K):
            val = code[k]
            codebook[c,k] = int(val)
            centroids[c,k] = TH_MEANS[k, val]
            centroid_threshold[0,k] = TH[k, val]
            centroid_threshold[1,k] = TH[k, val+1]
        
        centroid_thresholds.append(centroid_threshold)

    cneigbhbours = codebook_neighbours_inner(TH, W, codebook)
    
    return (cneigbhbours, centroid_thresholds)



# This is an inefficient way, replaced by the function above
def codebook_neighbours_oldway(TH, W):
    
    Nthresh = TH.shape[1]-1
    K = TH.shape[0]
    Ncodes = Nthresh**K
    
#    
    TH[:,0] *= 100
    TH[:,-1] *= 100
    TH_MEANS = 0.5*(TH[:,:-1]+TH[:,1:])
    
    codebook = np.zeros((Ncodes, K), dtype=int)
    centroids = np.zeros((Ncodes, K))
    centroid_thresholds = []
    
    for c in range(Ncodes):
        centroid_threshold = np.zeros((2,K))
        code = dec2base_padded(c, Nthresh, K)
        
        for k in range(K):
            val = code[k]
            codebook[c,k] = int(val)
            centroids[c,k] = TH_MEANS[k, val]
            centroid_threshold[0,k] = TH[k, val]
            centroid_threshold[1,k] = TH[k, val+1]
        
        centroid_thresholds.append(centroid_threshold)

    cneigbhbours = np.zeros((Ncodes, 1), dtype=object)

    for c1 in range(Ncodes):
        neighbour = np.zeros((Ncodes,1))
        
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
                    
                    
            neighbour[c2] = np.sqrt(sm)/W
        cneigbhbours[c1,0] = neighbour

    return (cneigbhbours, centroid_thresholds)





class SyncRefPCA:
    
    def __init__(self, K=4, Kfull=32, corr_threshold=0.85, Nthresh=4, num_clusters_to_search=8, super_intense_search=False):
        self.K = K
        self.Kfull = Kfull
        self.corr_threshold = corr_threshold
        
        self.Nthresh = Nthresh
        self.THs = {}
        self.cluster_distances = {}
        self.cluster_thresholds = {}
        self.T = None
        self.super_intense_search = super_intense_search
        
        self.num_clusters_to_search = num_clusters_to_search


    
    def compute_codes(self, X, return_signals=False):
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(X.T)
        pca.components_.shape
        
        Z = pca.transform(X.T).T
        

        self.T = X.shape[0]
        NS = X.shape[1] 
        self.THs = find_thresholds(Z.T, self.K, self.Nthresh)
        
        (self.cluster_distances, self.cluster_thresholds) = codebook_neighbours(self.THs, self.T)
        self.Ncodes = len(self.cluster_thresholds)
        
        Codes = -1*np.ones((NS), dtype=int)
        CodesRaw = -1*np.ones((NS, self.Kfull))
        
        if return_signals:
            Signals = np.zeros((NS), dtype=object)
        
        idx = 0
        for s in range(NS):
            idx += 1
            Codes[s] = compute_integer_code(Z[0:self.K,s], self.THs)-1
            if return_signals:
                x = X[:,s]
                xn = (x-x.mean())/(x.std()+1e-14)
                Signals[s] = xn
            
            CodesRaw[s,:] = Z[0:self.Kfull,s]

        if return_signals:
            return (Codes, CodesRaw, Signals)
        else:
            return (Codes, CodesRaw)
        
        
    
    def get_synchrony(self, X,rand_point_ratio=0.005,one_per_window=False, order_by='time'):
        time_start = time()

        obj = self.compute_codes(X, True)
        Codes = obj[0]
        CodesRaw = obj[1]
        Signals = obj[2]
        
        syncsets = []
        syncset_sizes = []
        idx = 0
        
        SignalsRaw = np.zeros((0,len(Signals[0])))
        for i in range(len(Signals)):
            SignalsRaw = np.concatenate((SignalsRaw, Signals[i].reshape(1,-1)), axis=0)

        
        (exp_clusters, clusters_to_search) = self.get_eclusters(Codes[:], CodesRaw[:], [])

        csyncsets = []
        csyncset_sizes = []            
        for ci in clusters_to_search:
            (_, inliers_set) = self.find_max_set_inliers_randomized(CodesRaw[:,0:self.Kfull], exp_clusters[ci], rand_point_ratio)
            
            if len(inliers_set) > 1:
                (num_max_inliers, rel_inliers) = self.find_max_set_inliers_randomized(SignalsRaw, exp_clusters[ci][inliers_set],rand_point_ratio=0.01)
                csyncsets.append(exp_clusters[ci][inliers_set][rel_inliers])
                csyncset_sizes.append(num_max_inliers)
            else:
                csyncsets.append([])
                csyncset_sizes.append(0)
            
        if len(csyncset_sizes)>0:
            if one_per_window:
                maxid = np.argmax(csyncset_sizes)
                inliers_set = csyncsets[maxid]
                
                
                if len(inliers_set)>1:
                    syncsets.append(inliers_set)
                    syncset_sizes.append(len(inliers_set))  
            else:
                for csi in range(len(csyncset_sizes)):
                    csyncset = csyncsets[csi]
                    inliers = exp_clusters[clusters_to_search[csi]][csyncset]
                    if len(inliers)>1:
                        syncsets.append(inliers)
                        syncset_sizes.append(len(inliers))
            
        syncset_sizes = np.array(syncset_sizes)
        
        if order_by == 'syncset_size':
            indices = np.argsort(syncset_sizes)[::-1]
        elif order_by == 'time':
            indices = range(0, len(syncset_sizes))
            
        top_syncsets = []
        for idx in indices:
            top_syncsets.append(syncsets[idx])
        
        time_elapsed = time() - time_start
        # print('%d   \t /   \t (total time:  %.2f secs)' %  (len(top_syncsets[0]), time_elapsed))

        return (top_syncsets, time_elapsed) #, sync_signal_sets)


    


    def get_expanded_cluster(self, cluster_id, Codes, CodesRaw, clusters_to_ignore):
        nextended_signals = []
        
        nsig = 0
        nelim = 0
        indices1 = np.where(Codes==cluster_id)[0]
        
        dist_theshold = np.sqrt(2*self.T*(1-self.corr_threshold))
        epsilon_theta = dist_theshold*np.sqrt(self.Kfull/(2*(self.Kfull+1)))
        
        for ii in range(len(indices1)):
            nextended_signals.append(indices1[ii])
            nsig += 1
            
        neighbour_indices = np.argwhere(self.cluster_distances[:,cluster_id] <= epsilon_theta)[:,0]
        neighbours = np.array(range(len(self.cluster_distances[:,0])))
        neighbours = neighbours[neighbour_indices.tolist()]
        
        for j in range(len(neighbours)):
            if neighbours[j] == cluster_id:
                continue
            
            if neighbours[j] in clusters_to_ignore:
                continue
            
            indices2 = np.where(Codes==neighbours[j])[0]
            distsi1_i2 = np.zeros((len(indices2),1))
            for ii in range(len(indices2)):
                i2 = indices2[ii]
                cp = np.matlib.repmat(CodesRaw[i2, :self.K], 2,1)
                binar1 = self.cluster_thresholds[cluster_id][0,:] < CodesRaw[i2, :self.K]
                binar2 = CodesRaw[i2, :self.K] < self.cluster_thresholds[cluster_id][1,:]
                binar = np.multiply(binar1, binar2)
                binar = np.array([not x for x in binar])
                
                dist = np.multiply(binar, np.min(abs(self.cluster_thresholds[cluster_id]-cp), axis=0))
                distsi1_i2[ii] = np.linalg.norm(dist)
                
            for ii in range(len(indices2)):
                if distsi1_i2[ii] <= epsilon_theta:
                    
                    nextended_signals.append(indices2[ii])
                    nsig += 1
                else:
                    nelim += 1
        
        return np.array(nextended_signals)
    
    
    
    
        
    def find_max_set_inliers(self, X, xm=None, Xall=None, n_indices=None, plot_em=False, min_inliers=0,inlier_eps=None):
        
        distance_threshold = np.sqrt(2*self.T*(1-self.corr_threshold))
        
        if inlier_eps is None:
            inlier_eps = distance_threshold*1.25
        
        N = X.shape[0]
        inliers = list(range(N))
        
        radius = np.inf
        
        keep_ptg_large_step = max(int(len(inliers)*0.3), 1)
        keep_ptg_medium_step = max(int(len(inliers)*0.16), 1)
        keep_ptg_small_step = max(int(len(inliers)*0.05), 1)
        
        if self.super_intense_search:
            keep_ptg_large_step = max(int(len(inliers)*0.15*0.5), 1)
            keep_ptg_medium_step = max(int(len(inliers)*0.08*0.5), 1)
            keep_ptg_small_step = max(int(len(inliers)*0.025*0.5), 1)
            
        if xm is not None:
            inliers = []
            for i in range(N):
                if np.linalg.norm(xm-X[i,:]) <= inlier_eps:
                    inliers.append(i)

        if inliers is None or len(inliers) == 0:
            inliers = list(range(N))
        
        mb = miniball.Miniball(X[inliers,:])
        radius = math.sqrt(mb.squared_radius())
        
        tightest_radius = distance_threshold*np.sqrt(self.T/(2*(self.T+1)))
        while radius > tightest_radius:
            if xm is None:
                xm = np.mean(X[inliers,:], axis=0)
            
            Xh = np.matlib.repmat(xm, len(inliers),1)
            XD = Xh-X[inliers,:]
            XD2 = np.power(XD,2)
            D = np.sqrt(np.sum(XD2, axis=1))
            
            mb = miniball.Miniball(X[inliers,:])
            
            if radius/tightest_radius > 1.20:
                keep_ptg = keep_ptg_large_step
            elif radius/tightest_radius > 1.15:
                keep_ptg = keep_ptg_medium_step
            else:
                keep_ptg = keep_ptg_small_step
                
            outliers = np.argsort(-D)[0:keep_ptg]
            outliers = np.sort(outliers)[::-1]
            
            for outlier in outliers:
                inliers.pop(outlier)

            if len(inliers) <= 1:
                break
            
            if len(inliers) < min_inliers:
                inliers = []
                break
            
            mb = miniball.Miniball(X[inliers,:])
            radius = math.sqrt(mb.squared_radius())
            
    
        radd_diff = tightest_radius-distance_threshold*0.5
        
        num_iter = 0
        for i in range(len(inliers)):
            
            num_iter += 1
            Xh = np.matlib.repmat(mb.center(), len(inliers),1)
            XD = Xh-X[inliers,:]
            XD2 = np.power(XD,2)
            dists = np.sqrt(np.sum(XD2, axis=1))
            
            rem_idx = np.where(np.abs(dists-radius)<radd_diff)[0]
            Xred = X[np.array(inliers)[rem_idx]]
            d2 = scipy.spatial.distance.pdist(Xred,'Euclidean')
            D2 = scipy.spatial.distance.squareform(d2)
            ### C2 = np.corrcoef(Xred)
            
            if np.max(D2)<= distance_threshold:
                break
            d2mean = np.mean(D2,axis=0)
            tmp = np.where(D2 >= distance_threshold)
            to_remove = []
            
            elems = []
            vals = []

            for ii in range(len(tmp[0])):
                ptx = tmp[0][ii]
                pty = tmp[1][ii]
                
                if d2mean[ptx] > d2mean[pty]:
                    elem = rem_idx[ptx]
                else:
                    elem = rem_idx[pty]
                
                if elem in elems:
                    continue
                elems.append(elem)
                vals.append(D2[ptx,pty])
                
            vals = np.array(vals)
            sidx = np.argsort(vals)[::-1]

            Nto_remove = max([1,int(2*len(sidx)/3)])
            
            if Nto_remove <= 2:
                Nto_remove = 1
            to_remove = np.array(elems)[sidx[0:Nto_remove]]
            to_remove = np.sort(np.array(to_remove))[::-1]

            for remove_idx in to_remove:
                inliers.pop(remove_idx)
            mb = miniball.Miniball(X[inliers,:])
            radius = math.sqrt(mb.squared_radius())

        
        # This is for debugging purposes
        # Need to uncomment C2 up above before runnning this
        print_output = False
        
        if print_output and len(inliers)>0 and len(C2.shape) > 0:
            if C2.shape[0] > 0:
                print('DIST (ALLOWED): %.2f (%.2f)'  % (np.max(D2), distance_threshold))
                print('Corr: %.2f'  % (np.min(C2)))
                
        return (inliers, radius, xm)   




    def find_max_set_inliers_randomized(self, CodesRaw, ecluster_members,  rand_point_ratio=0.01, Signals=None):
        
        Signals_subset = None
        
        if Signals is not None:
            Signals_subset = Signals[ecluster_members]
        
        (inliers, radius, xm) = self.find_max_set_inliers(CodesRaw[ecluster_members,:], Xall=Signals_subset)

        inliers_set = [inliers]
        num_inliers = [len(inliers)]
                    
        np.random.seed(1907)
        sidx = np.arange(len(ecluster_members))
        
        np.random.shuffle(sidx)
        
        Nrand_points = max([min(len(ecluster_members),2), int(rand_point_ratio*len(ecluster_members))])

        for ii in range(Nrand_points):
            xm = CodesRaw[ecluster_members[sidx[ii]], :]                
            (inliers, radius, xm) = self.find_max_set_inliers(CodesRaw[ecluster_members,:],  xm=xm, Xall=Signals_subset,min_inliers=max(num_inliers))
            num_inliers.append(len(inliers))
            inliers_set.append(inliers)
            
        maxid = np.argmax([len(x) for x in inliers_set])
        num_max_inliers = len(inliers_set[maxid])
        return (num_max_inliers, inliers_set[maxid])
    
    
    
    
    def get_eclusters(self, Codes, CodesRaw, clusters_to_ignore):
        
        ucodes = np.unique(Codes)
        cluster_pop = np.zeros((self.Ncodes,1), dtype=int)
        
        for ci in range(self.Ncodes):
            if ci not in ucodes:
                continue
            indices1 = np.where(Codes==ci)[0]
            if ci not in clusters_to_ignore:
                cluster_pop[ci,0] = len(indices1)
        
        clusters_to_search = np.argsort(cluster_pop.flatten())[::-1][0:self.num_clusters_to_search].tolist()
        
        # Compute expanded clusters
        exp_clusters = {}
        for ci in range(self.Ncodes): 
            if ci not in clusters_to_search:
                continue
            
            tmp = self.get_expanded_cluster(ci, Codes, CodesRaw, clusters_to_ignore)
            if len(tmp) < 2:
                clusters_to_search.remove(ci)
                continue
            exp_clusters[ci] = tmp
            
        return (exp_clusters, list(exp_clusters.keys()))
    
    
        
    
    def provide_solution_certificate(self, SignalsRaw, inliers):
        M = len(inliers)
        D = np.zeros((M,M))
        
        for m1 in range(M):
            for m2 in range(M):
                D[m1,m2] = np.linalg.norm(SignalsRaw[inliers[m1]]-SignalsRaw[inliers[m2]])
        
        distance_threshold = np.sqrt(2*self.T*(1-self.corr_threshold))
        if np.max(D) <= distance_threshold:
            print('Solution is valid: %.2f <= %.2f' % (np.max(D), distance_threshold))
        else:
            print('SOLUTION IS INVALID!', np.max(D), ' vs ', distance_threshold)

        
