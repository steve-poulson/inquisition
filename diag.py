import matplotlib.pyplot as plt
import networkx as nx
import numpy
from matplotlib import pyplot

from generators import *

A = [[1,1],[1,1]]

a = nx.DiGraph()
a.add_edge(1,2)
pos = nx.spring_layout(a)
nx.draw_networkx_edges(a, pos)
plt.show()

G = nx.from_numpy_matrix(A)

pyplot.clf() 
nx.draw(G)
plt.axis('off')
plt.show()

     
           