import matplotlib.pyplot as plt
import networkx as nx
import numpy
from matplotlib import pyplot

from generators import *

def plot(y, title, t):
    
        a = y[0]
        b = y[1]
        
        print a
        
        if a and b:
        
            bins = numpy.linspace(min(a+b), max(a+b), 20)
            pyplot.clf()
            
            if a:
                w0 = numpy.ones_like(a)/float(len(a))
                pyplot.hist(a, bins, weights=w0,alpha=0.5, color='r', histtype='stepfilled', label='link')
            
            if b:
                w1 = numpy.ones_like(b)/float(len(b))
                pyplot.hist(b, bins,weights=w1, alpha=0.5, color='b', histtype='stepfilled', label='no link')
            
            pyplot.title(title)
            pyplot.ylabel("Fraction over population")
            pyplot.xlabel("Similarity")
            pyplot.legend();
            #plt.savefig("/Users/spoulson/Dropbox/my_papers/figs/"+title.replace(' ','_')+'_'+ str(t) +'.png')
            pyplot.show()

A,y = karate()

name = "Amazon"

G = nx.from_numpy_matrix(A)

G = nx.read_edgelist("/Users/spoulson/data/com-amazon.ungraph.txt")

print "read in", len(G) 

G2 = nx.Graph()

es = grow2(G2,G,100)

print len(es)

y1 = []
y0 = []

for t in range(20):
    
    y2 = [[],[]]
    y3 = [[],[]]
    y4 = [[],[]]
    pry = [[],[]]
    
    G1 = G2.copy()
    
    es = grow2(G2,G,100)
    print "es",len(es)
    
    if es:
        print "flow"
        C = nx.current_flow_closeness_centrality(G1)
        print "pr"
        #pr = nx.pagerank_numpy(G1)
        M = katz(G1)
        print "pr done", M.shape
    
        a1 =0
        b1 =0
        
        nodes = G2.nodes()
        
        for i in range(0,len(nodes)):
            for j in range(i+1,len(nodes)):
                
                print i,j
                
                a = nodes[i]
                b = nodes[j] 
                
                if a in G1 and b in G1: 
                    
                    if nx.has_path(G1,a,b):
                        spl = nx.shortest_path_length(G1,a,b)
    
                    cen = max(C[a],C[b])
                    jac = jaccard(G1,a,b)
#                     pgr = (pr[a]+pr[b]) / 2
                    pgr = M[G1.nodes().index(a),G1.nodes().index(b)]
                        
                    if b in G2.neighbors(a):
                        if b not in G1.neighbors(a):
                            y2[0] += [spl]
                            y3[0] += [cen]
                            y4[0] += [jac]
                            pry[0] += [pgr]
                    else:
                        y2[1] += [spl]
                        y3[1] += [cen]
                        y4[1] += [jac]
                        pry[1] += [pgr]
                
        y0 += [a1]
        y1 += [b1]
        
        if len(G2) < 50:
            pos = nx.fruchterman_reingold_layout(G2)
            a = G1.nodes()
            b = [x for x in G2.nodes() if x not in set(G1.nodes())]
            
            pyplot.clf() 
            nx.draw_networkx_nodes(G2, pos, nodelist=a, node_color='b')
            nx.draw_networkx_nodes(G2, pos, nodelist=b, node_color='r')
            nx.draw_networkx_edges(G2, pos, edgelist=G1.edges())
            nx.draw_networkx_edges(G2, pos, edge_color='r', edgelist=es)
            nx.draw_networkx_labels(G2, pos, font_size=8, font_family='sans-serif')
            plt.axis('off')
            plt.savefig("/Users/spoulson/Dropbox/my_papers/figs/"+name+'_'+ str(t) +'.png')
            plt.show()
    #       
         
        plot(y2, name+" shortest path",t)
        plot(y3, name+" Centrality",t)
        plot(y4, name+" Jaccard",t)
        plot(pry, name+" Mod",t)
     
pyplot.clf()  

fig = pyplot.figure()
ax = fig.add_subplot(111)
 
ax.bar(numpy.arange(len(y0)),y0,color='r')
ax.bar(numpy.arange(len(y1))+0.35,y1)
        
pyplot.title("Internal links ")
pyplot.ylabel("Fraction over population")
pyplot.xlabel("Similarity")
pyplot.legend();

pyplot.show()

     
           