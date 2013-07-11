
from numpy import *
import time
import networkx as nx

from random import *

from operator import itemgetter
from matplotlib.colors import ListedColormap

def simple():
    A = mat([[0, 1, 1, 0, 0, 0,],
              [1, 0, 1, 0, 0, 0],
              [1, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1],
              [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0]])
    
    y = [1,1,1,-1,-1,-1]
    
    return A, y

def karate():
    G=nx.karate_club_graph()
    club1 = 1
    club2 = -1
    G.node[0]['club'] = club1
    G.node[1]['club'] = club1
    G.node[2]['club'] = club1
    G.node[3]['club'] = club1
    G.node[4]['club'] = club1
    G.node[5]['club'] = club1
    G.node[6]['club'] = club1
    G.node[7]['club'] = club1
    G.node[8]['club'] = club1
    G.node[9]['club'] = club2
    G.node[10]['club'] = club1
    G.node[11]['club'] = club1
    G.node[12]['club'] = club1
    G.node[13]['club'] = club1
    G.node[14]['club'] = club2
    G.node[15]['club'] = club2
    G.node[16]['club'] = club1
    G.node[17]['club'] = club1
    G.node[18]['club'] = club2
    G.node[19]['club'] = club1
    G.node[20]['club'] = club2
    G.node[21]['club'] = club1
    G.node[22]['club'] = club2
    G.node[23]['club'] = club2
    G.node[24]['club'] = club2
    G.node[25]['club'] = club2
    G.node[26]['club'] = club2
    G.node[27]['club'] = club2
    G.node[28]['club'] = club2
    G.node[29]['club'] = club2
    G.node[30]['club'] = club2
    G.node[31]['club'] = club2
    G.node[32]['club'] = club2
    G.node[33]['club'] = club2 

    y =[ G.node[x]['club']  for x in G.nodes()]
    
    A = nx.to_numpy_matrix(G,weight='weight')
    
    return A, y

def random_delete(A):
    G = nx.from_numpy_matrix(A)
    
    x,y = G.edges()[random.randint(G.number_of_edges())]
    
    G.remove_edge(x,y)
    
    return (x,y),nx.to_numpy_matrix(G,weight='weight')

def find(A, x):
    A1 = (A*A*A*A*A)/(len(A) * len(A)* len(A)* len(A)* len(A))
    
    for y in range(len(A)):
        if A[x,y] != 1: print x,y, A1[x,y]
        


def random_data(n=10, m=10):
    data = []
    label = []
    d = {}
        
    data.extend(random.multivariate_normal([2, 2], [[0.2, 0.4], [0.3, 0.5]], n))
    data.extend(random.multivariate_normal([0, 0], [[0.5, 2], [0, 1]], m))
    label.extend([0]*n)
    label.extend([1]*m)
    #print label
    x, y = mat(data).T
   
    data = mat(data)
    
    return data, label
    
def RBF_Kernel(data, label):
    
    N = len(label)
    M = len(data.H)

    for i in xrange(len(data.H)):
        data[:, i] = data[:, i] - mean(data[:, i])
    
    my_var = var(data, 0, ddof=1)

    #prevents div errors
    #for i in xrange(len(my_var)):
    my_var += 0.00000001

    # Construct the graph using standard RBF kernel

    A = zeros((N, N));
    for i in range(0, N):
        for j in range(0, N):
            k = exp(-1 * sum((power(data[i, :] - data[j, :], 2) / (2 * my_var))))
            if (k > 0.3):
                k = 1
            else:
                k = 0
            A[i, j] = k

    for i in range(0, N):
        A[i, i] = 0;
        
    #print A

    G = nx.from_numpy_matrix(A)
    G = nx.connected_component_subgraphs(G)[0] #largest connected graph

    label1 = []
    p = {}
    
    for i in G.nodes():
        label1.append(label[i])
        if label[i] < 1:
            p[i] = 0
        else:
            p[i] = 1

    A = nx.adj_matrix(G)
    
    return A, label1

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

def grow2(G1,edges,n):
    
    if not G1:
        G1.add_node(y)
    
    es = [(x,y) for x,y in edges if x in G1.nodes() or y in G1.nodes()]
    
    print "G1", len(G1)
    print "??",len(es)
    
    shuffle(es)
    
    es = [(x,y) for x,y in es if not (x,y) in G1.edges()]
    
    for x,y in es[:n]:
        G1.add_edge(x,y)
        G1.add_node(y)
        
    return es[:n],G1

def paretoMLE(G):
    
   X = nx.degree(G).values()
   n = len(X)
   m = min(X)
   a = n/sum(log(X)-log(m))
   return m,a 

def save():
    label, A, x,y = randomGraph()
    savemat('/Users/spoulson/PhD/rnd1.mat',{'A':A, 'label':label, 'x':x,'y':y})
    
def jaccard (G,a,b):
    na = set(G.neighbors(a))
    nb = set(G.neighbors(b))
    
    return len(na.intersection(nb)) / float(len(na.union(nb)))