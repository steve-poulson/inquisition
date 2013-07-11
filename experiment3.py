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

from generators import *

X,y = random_data(n=10,m=10)

A,y = RBF_Kernel(X,y)

A,y = simple()

Y = (mat(y).H * y + 1) / 2 


from supervised import *
from constrained_laplacian import *
from lfl import *

cl = LFL(A)
p = supervised(A)

N = len(A)

t = [(x, y) for x in range(N) for y in range(N) if x != y]

r.shuffle(t)

for i,j in t:
    
    f = cl.train( i,j,Y[i,j])
    plt.title('constrained') 
    plt.imshow(f,interpolation='nearest');
    plt.colorbar()
    plt.show();
    
    f = p.train( i,j,Y[i,j])
    plt.title('percept') 
    plt.imshow(f,interpolation='nearest')
    plt.show();
        
     
           