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

A,y = karate()

print y

import nmf2


n_samples = 1000
n_features = 1000
n_topics = 10
n_top_words = 20

nmf = nmf2.NMF(n_components=2)

W = nmf.fit_transform(A)

Z =  dot(W,nmf.components_)

plt.imshow(A,interpolation='nearest')
plt.show();

plt.imshow(Z,interpolation='nearest')
plt.show();

plt.imshow([y,-(nmf.components_[0] - nmf.components_[1])],interpolation='nearest')
plt.show();

print sign(nmf.components_[0] - nmf.components_[1])

for topic_idx, topic in enumerate(nmf.components_):
    print "Topic #%d:" % topic_idx
    print topic
    print " ".join([str(i) for i in topic.argsort()[:-n_top_words - 1:-1]])
    print


        
     
           