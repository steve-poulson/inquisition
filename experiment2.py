import networkx as nx
import matplotlib.pyplot as plt

import numpy
import logging
from apgl.util.Parameter import Parameter
from apgl.util.Util import Util
import matplotlib.pyplot as plt
from numpy import *
from numpy.linalg import *
from random import *
from numpy.random import *
from sklearn.ensemble import RandomForestClassifier
from supervised import *
from matrix_perceptron import *

from sklearn.linear_model import SGDClassifier

import scipy

def grow(n, m, G):
    
    G = G.copy()
    
    c = 0
    while c < m:
        seq = []
        for i in range(len(G)):
            seq += [i] * G.degree(i)
        
        e = (randint(0,len(G)), choice(seq))
        if (e not in G.edges() ):
            c +=1
            G.add_edge(e[0],e[1])
    
    #add new nodes
    for j in range(n):
        seq = []
        for i in range(len(G)):
            seq += [i] * G.degree(i)
            
        G.add_edge(len(G), choice(seq))
    return G

def grow1(G,edges,n,m):
    
    G1 =  nx.Graph()
    
    G1.add_node(edges[randint(0,len(edges))][0])
    print G.edges()

    while len(G1.nodes()) < m or len(G1.edges()) < n:
        
        print G1.edges()
        
        es = [(x,y) for x,y in edges if (x in G1.nodes() and  (x,y)  not in G1.edges()) or (y in G1.nodes() and  (y,x)  not in G1.edges())]
        
        if len(es) == 0: continue
        
        shuffle(es)
             
        x,y = es.pop()
        
        G1.add_edge(x,y)
        G1.add_node(y)
        
    return G1

def grow2(G,edges,n):
    
    es = [(x,y) for x,y in edges if (x in G1.nodes() and  (x,y)  not in G1.edges()) or (y in G1.nodes() and  (y,x)  not in G1.edges())]
    shuffle(es)
    
    for x,y in es[:n]:
        print x,y

        G1.add_edge(x,y)
        G1.add_node(y)
        
    return G1

def paretoMLE(G):
    
   X = nx.degree(G).values()
   n = len(X)
   m = min(X)
   a = n/sum(log(X)-log(m))
   return m,a 

def features(G,G1):
    A = nx.adj_matrix(G)
    n = len(A)
    A1 = nx.adj_matrix(G1)
    
    D = A1[:n,:n]-A
    
    pos = 0
    neg = 0
    
    iz = range(n)
    jz = range(n)
    
    shuffle(iz)
    shuffle(jz)
    
    for i in iz:
        for j in jz:
            
            if D[i,j] == 1:
                pos +=1
                train += [ [dot(A.A[i] , A.A[j]) / norm(A.A[i])* norm(A.A[j]),M[i,j],FF[i,j]]]
                target += [D[i,j]]
            elif neg < c:
                neg +=1
                train += [[dot(A.A[i] , A.A[j]) / norm(A.A[i])* norm(A.A[j]),M[i,j],FF[i,j]]]
                target += [D[i,j]]
    
    return train, target


def process(G1, G2):
    
    print nx.adj_matrix(G1)
    print nx.adj_matrix(G2)
    
    a = G1.nodes()
    b = [x for x in G2.nodes() if x not in set(G1.nodes())]
    pos = nx.random_layout(G2)
    nx.draw_networkx_nodes(G2, pos, nodelist=a, node_color='b')
    nx.draw_networkx_nodes(G2, pos, nodelist=b, node_color='r')
    nx.draw_networkx_edges(G2, pos, edgelist=G1.edges())
    nx.draw_networkx_edges(G2, pos, edge_color='r', edgelist=[x for x in G2.edges() if x not in set(G1.edges())])
    nx.draw_networkx_labels(G2, pos, font_size=8, font_family='sans-serif')
    plt.axis('off')
    plt.show()
    N = len(G1)
    A = nx.adj_matrix(G1)
    t = [x for x in G2.edges() if x not in G1.edges()]
    print t
    random.shuffle(t)
    return A, t

def run():
    """
    G = nx.barabasi_albert_graph(50,4)
    G1 = grow(10,50,G)
    G2 = grow(19,50,G1)
    A, t = process(G1, G2)
    """
    
    from generators import *

    A,y = simple()
    G = nx.from_numpy_matrix(A)
    G1 = grow1(nx.Graph(),G.edges(),3,4)
    
    A, t = process(G1, G)
    
    G1 = grow1(G1,G.edges(),5,6)
    
    A, t = process(G1, G)
    
    cl = supervised(A)
    
    for (i,j) in t:
        
        f = cl.train( i,j,A[i,j])
        plt.title('constrained') 
        plt.imshow(f,interpolation='nearest');
        plt.show();
run()

"""
nx.draw(G1)
plt.show()
"""
