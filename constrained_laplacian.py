import matplotlib.pyplot as plt
from scipy.linalg import norm, eig, svd, pinv
from scipy.sparse.linalg import svds, eigs
from numpy import *
import time
import networkx as nx
#import scipy.sparse.linalg.eigen.arpack.speigs as speig
from scipy.sparse.linalg import dsolve
from operator import itemgetter
from matplotlib.colors import ListedColormap

from scipy.cluster.vq import *
from scipy.sparse import bsr_matrix,spdiags,csc_matrix,lil_matrix
from scipy.io import *
import numpy.random as r
import numpy as np
import scipy.optimize as o
from networkx import *

class constrained_laplacian():
    
    def __init__(self,A):
        d = sum(A.getA(), 0)
        self.N = len(A)
        D = diag(d) 
        self.D_norm = mat(diag(d ** -0.5))
        
        self.Q = zeros((self.N,self.N))
        self.vol = sum(diag(D))
        self.L = eye(self.N) - self.D_norm * A * self.D_norm
        
    def train(self,i,j,y):

        y = float(y*2 -1)
        
        self.Q[i,j] = y
        self.Q[j,i] = y

        Q_norm = self.D_norm * self.Q * self.D_norm
        lam = svd(Q_norm, compute_uv=False)[0]
        Q1 = Q_norm - (lam * 0.5) * eye(self.N)
        [val, vec] = eig(self.L, Q1)
        
        vs = []
        
        for i in range(len(val)):
            if val[i] > 0.000001:
                
                v = vec[:, i] / norm(vec[:, i]) * self.vol ** 0.5
                
                cost = v * self.L * mat(v).H
                vs.append((cost, v))
                
        vs = sorted(vs, key=itemgetter(0))
        
        _, best_v = vs[0]
    
        z = self.D_norm * mat(best_v).H
        
        return  (sign(z * z.H)+1)/2
        
     
           