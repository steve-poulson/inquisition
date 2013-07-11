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

class gausian_harmonic():
    def __init__(self,A):
        D = diag(d)
        self.L = D - A
        
        self.ls = zeros(len(L))
    
    def train(self,i,y):
        
        self.ls[i] = y
        
        n = len(ls)
        i = 0
        k = 0
    
        fl = zeros(n)
    
        ix = zeros(n)
    
        ret = array(ls)
        for j in range(n):
            L1 = zeros((n, n))
            L2 = zeros((n, n))
            if ls[j] != 0:
                L1[:i, :] = L[:i, :]
                L1[i:n - 1, :] = L[i + 1:, :]
                L1[n - 1, :] = L[i, :]
    
                L2[:, :i] = L1[:, :i]
                L2[:, i:n - 1] = L1[:, i + 1:]
                L2[:, n - 1] = L1[:, i]
    
                L = L2
                fl[k] = ls[j]#(ls[j]+1) /2
                k += 1
                 
            else:
                ix[i] = j
                i += 1
                
        L = mat(diag(sum(L, 0)) - L)
        
        sz = n - i
    
        fu = -L[:i, :i].I * L[:i, i:] * mat(fl[:sz]).H
    
        for j in range(i):
            ret[ix[j]] = sign(fu[j])
        
        return (mat(ret).H*ret +1)/2 