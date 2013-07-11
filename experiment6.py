import matplotlib.pyplot as plt
import networkx as nx
import numpy
from matplotlib import pyplot

from generators import *

from sklearn import metrics
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn import metrics
from sklearn.decomposition import *
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import *
from scipy.spatial import distance
import hierarchical_clustering

def evaluate(l,nl):
    fp = []
    
    c = 0.
    
    for a,b,f in nl:
        c += clf.predict(f)
        
        if not G.has_edge(a,b):
            fp +=[ (clf.predict_proba(f)[0,1], f)]
    #print"------------------"
    for a,b,f in l:
        pass
        #print clf.predict_proba(f),G.has_edge(a,b)
    
    print "error=",c / len(nl)

    return fp

data = create(nx.read_edgelist("/Users/spoulson/data/com-amazon.ungraph.txt"),3,initial=100, inc=100)

#data = create(nx.karate_club_graph(), 5)

l,nl = data[2]

ll= len(l)

l = [c for a,b,c in l]
nl = [c for a,b,c in nl]

XD = l+nl
labels = [1]*ll+[0]*ll

D = distance.squareform(distance.pdist(XD, metric = 'jaccard'))
S = 1 - (D / numpy.max(D))

names = ['*'] * len(S)

plt.imshow(S,interpolation='nearest')
plt.show()

hierarchical_clustering.heatmap(S, names, names, 'average', 'average', 'cityblock', 'cityblock', 'red_white_blue', "/tmp/a.png")


pca = PCA(n_components=3)
pca.fit(XD)
P = pca.transform(XD)

print "pca",pca.explained_variance_ratio_

ln = ""
for i in range(len(P)):
    ln += "%.1f|" % P[i,0]

labels = array(labels)

plt.scatter(P[:, 0], P[:, 1], c=labels,s=10,alpha=0.5, lw = 0)
plt.show()

clf = SGDClassifier(loss="log", penalty="l2", alpha=0.001)

clf = RandomForestClassifier()
clf.fit(XD, labels)

l,nl = data[4]
  
fp = evaluate(l,nl)
  
ll = len(l)

fp = sorted(fp, reverse = True,key=lambda tup: tup[0])

fp = [c for a,c in fp[:ll]]
tp = [c for a,b,c in l[:ll]]

XD = tp+fp
labels = [1]*ll+[0]*ll

clf = RandomForestClassifier()
clf.fit(XD, labels)

evaluate(l,nl)
           