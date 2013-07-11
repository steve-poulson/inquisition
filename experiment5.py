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
from scipy.sparse import *
from scipy.io import *
import numpy.random as r
from numpy import *
import scipy.optimize as o
from networkx import *

from generators import *
import csv


N = 1000999999

with open('/Users/spoulson/data/dataRev2/Train.csv', 'rb') as f:
    cs = csv.reader(f)
    cs.next()
    cnt = 0
    
    papers = []
    
    for row in cs:
        cnt +=1
        if cnt > N: break
        print cnt,row[0]
        papers += row[1].split(' ')
        papers += row[2].split(' ')

    papers = set(papers)
    print "1"
    
with open('/Users/spoulson/data/dataRev2/PaperAuthor.csv', 'rb') as f:
    cs = csv.reader(f)
    cs.next()
    cnt = 0
    
    authors = []
    
    for row in cs:
        if row[0] in papers:
            authors += [row[1]]

    authors = set(authors)
    
    print "2"
"""   
with open('/Users/spoulson/data/dataRev2/PaperAuthor.csv', 'rb') as f:

    cs = csv.reader(f)
    cs.next()
    
    cnt = 0
        
    paper_map = {}#author
    author_map = {}#auth
    for row in cs:
            if row[1] in authors:
                paper_map.setdefault(row[0], len(paper_map))
                author_map.setdefault(row[1], len(author_map))
    f.seek(0)
    cs.next()
        
    print "size", len(paper_map),len(author_map)
    M = lil_matrix((len(paper_map),len(author_map)))

    cnt = 0
    for row in cs:
        if row[1] in authors:
            author = paper_map[row[0]]
            y = author_map[row[1]]
            M[author,y] = 2.
        
#    M = lil_matrix(mat([[3,0,1,0,0],[0,0,1,1,3],[0,2,2,0,0],[0,2,2,0,0]]), dtype=float)


with open('/Users/spoulson/data/dataRev2/Train.csv', 'rb') as f:
    cs = csv.reader(f)
    cs.next()
    cnt = 0
    
    papers = []
    
    for row in cs:
        cnt +=1
        if cnt > N: break
        author = row[0]
        for y in row[1].split(' '):
            M[paper_map[y],author_map[author] ] = 3
        for y in row[2].split(' '):
            M[paper_map[y],author_map[author] ] = 1
    print "calculating"
    
    U,D,V = svds(M.tocsc(), k=1)
    
    M1 = mat(U)*diag(D)*mat(V)
    
    print M1
    """
with open('/Users/spoulson/data/dataRev2/Valid.csv', 'rb') as f:
    cs = csv.reader(f)
    cs.next()
    cnt = 0
    
    papers = []
    
    for row in cs:
        
        print "author_map=", row[0] in authors
        
        cnt +=1
        for y in row[1].split(' '):
            pass
        

""" 
    plt.title('percept') 
    plt.imshow(M1)
    plt.colorbar()
    plt.show();
"""
     
           