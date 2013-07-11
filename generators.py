
from numpy import *
import time
import networkx as nx

from random import *

from operator import itemgetter
from matplotlib.colors import ListedColormap
from scipy.linalg import norm, eig, svd

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

def grow2(G1,G,n_edges):
    
    if not G1:
        G1.add_node(G.nodes()[0])
    
    edges = set()
    
    for n in G1.nodes():
        eds = G.edges(n)
        edges = edges.union(eds)
        
    edges = edges.difference(G1.edges())
    
    es = []
    
    while len(es) < n_edges: #and set(G.edges()).difference(G1.edges())""":
        
        #print len(es)
        
        x,y = edges.pop()
        
        if x not in G1.nodes():
            G1.add_node(x)
            edges = edges.union(G.edges(x))
            
        elif y not in G1.nodes():
            G1.add_node(y)
            edges = edges.union(G.edges(y))
        
        if not G1.has_edge(x,y):
            es += [(x,y)]
            G1.add_edge(x,y)
        
    return es

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

#----------
def create(G,t_num, inc=10,initial=10):
    
    G2 = nx.Graph()
    
    es = grow2(G2,G,inc)
    
    y1 = []
    y0 = []
    
    data = []
    
    for t in range(t_num):
        print t
        G1 = G2.copy()
        
        es = grow2(G2,G,initial)
        
        if es:
            C = nx.current_flow_closeness_centrality(G1)
            pr = nx.pagerank_numpy(G1)
            M = mod(G1)
            K = katz(G1)
            cl = nx.closeness_centrality(G1)
            
            nodes = G2.nodes()
            
            link = []
            no_link = []
            
            for i in range(0,len(nodes)):
                for j in range(i+1,len(nodes)):
                    
                    #print i,j
                    
                    a = nodes[i]
                    b = nodes[j] 
                    
                    if a in G1 and b in G1 and nx.has_path(G1,a,b) and nx.shortest_path_length(G1,a,b) < 7:
                        
                        features = zeros(11)
                        
                        if nx.has_path(G1,a,b): features[0] = nx.shortest_path_length(G1,a,b)
                        features[1] = min(C[a],C[b])
                        features[2] = max(C[a],C[b])
                        features[3] = jaccard(G1,a,b)
                        features[4] = M[G1.nodes().index(a),G1.nodes().index(b)]
                        features[5] = K[G1.nodes().index(a),G1.nodes().index(b)]
                        features[6] = min(pr[a],pr[b])
                        features[7] = max(pr[a],pr[b])
                        features[8] = max(cl[a],cl[b])
                        features[9] = min(cl[a],cl[b])
                        features[10] = adamic_adar(G1,a,b)
                        
                        if b in G2.neighbors(a):
                            if b not in G1.neighbors(a):             
                                link += [[a,b,features]]
                        else:
                            no_link += [[a,b,features]]
            
            data.append((link,no_link)) 

    return data

#---------

"""
Adamic/Adar: A popular link prediction heuristic. In practice, the radius of a node is analogous to its de- gree,
 and hence it is natural to weight a node more if it has lower degree. The Adamic/Adar measure (Adamic & Adar, 2003) 
 was introduced to measure how related two home-pages are. The authors computed this by looking at 
 common features of the webpages, and instead of computing just the number of such features, 
 they weighted the rarer features more heavily. In our social networks context, 
 this is equivalent to computing similarity between two nodes by computing the number of common neighbors, 
 where each is weighted inversely by the logarithm of its degree.
"""

def adamic_adar(G,a,b):
    na = set(G.neighbors(a))
    nb = set(G.neighbors(b))
    
    ret =0 
    
    for n in G.degree(na.intersection(nb)).values():
        ret += 1 / math.log(n)
         
    return ret

def katz(G):
    
    A = nx.to_numpy_matrix(G)
    
    [val, vec] = eig(A)
    vals, vecs = val.real, vec.real

    vals = sorted(vals)
    
    beta = 1/vals[-1] - 0.1
    
    I = eye(len(A))
    
    K = (I - beta * A).I - I

    return K

def mod(G):
    
    k = -20
    
    A = nx.to_numpy_matrix(G)
    
    d = sum(A, 0)
    m = d.sum()
    P = (mat(d).H * mat(d)) / float(m)
    
    A = mat(A)
    M = A - P

    [val, vec] = eig(M)

    vals, vecs = val.real, vec.real

    idx = argsort(vals)
    
    S =mat( vecs[:,idx[k:]])
    
    M_ = S * diag(vals[idx[k:]]) * S.H
    return M_