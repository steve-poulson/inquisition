
import matplotlib.pyplot as pylab
from matplotlib import mpl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import numpy
import string
import time
import sys, os
import getopt
from sklearn.cluster import *

################# Perform the hierarchical clustering #################

def heatmap(x, row_header, column_header, row_method,
            column_method, row_metric, column_metric,
            color_gradient, filename):
    
    n = len(x[0]); m = len(x)
    if color_gradient == 'red_white_blue':
        cmap=pylab.cm.bwr
    if color_gradient == 'red_black_sky':
        cmap=RedBlackSkyBlue()
    if color_gradient == 'red_black_blue':
        cmap=RedBlackBlue()
    if color_gradient == 'red_black_green':
        cmap=RedBlackGreen()
    if color_gradient == 'yellow_black_blue':
        cmap=YellowBlackBlue()
    if color_gradient == 'seismic':
        cmap=pylab.cm.seismic
    if color_gradient == 'green_white_purple':
        cmap=pylab.cm.PiYG_r
    if color_gradient == 'coolwarm':
        cmap=pylab.cm.coolwarm

    ### Scale the max and min colors so that 0 is white/black
    vmin=x.min()
    vmax=x.max()
    vmax = max([vmax,abs(vmin)])
#     vmin = vmax*-1
    norm = mpl.colors.Normalize(vmin, vmax) ### adjust the max and min to scale these colors

    ### Scale the Matplotlib window size
    default_window_hight = 8.5
    default_window_width = 12
    fig = pylab.figure(figsize=(default_window_width,default_window_hight)) ### could use m,n to scale here
    color_bar_w = 0.015 ### Sufficient size to show
        
    ## calculate positions for all elements
    # ax1, placement of dendrogram 1, on the left of the heatmap
    #if row_method != None: w1 = 
    [ax1_x, ax1_y, ax1_w, ax1_h] = [0.05,0.22,0.2,0.6]   ### The second value controls the position of the matrix relative to the bottom of the view
    width_between_ax1_axr = 0.004
    height_between_ax1_axc = 0.004 ### distance between the top color bar axis and the matrix
    
    # axr, placement of row side colorbar
    [axr_x, axr_y, axr_w, axr_h] = [0.31,0.1,color_bar_w,0.6] ### second to last controls the width of the side color bar - 0.015 when showing
    axr_x = ax1_x + ax1_w + width_between_ax1_axr
    axr_y = ax1_y; axr_h = ax1_h
    width_between_axr_axm = 0.004

    # axc, placement of column side colorbar
    [axc_x, axc_y, axc_w, axc_h] = [0.4,0.63,0.5,color_bar_w] ### last one controls the hight of the top color bar - 0.015 when showing
    axc_x = axr_x + axr_w + width_between_axr_axm
    axc_y = ax1_y + ax1_h + height_between_ax1_axc
    height_between_axc_ax2 = 0.004

    # axm, placement of heatmap for the data matrix
    [axm_x, axm_y, axm_w, axm_h] = [0.4,0.9,2.5,0.5]
    axm_x = axr_x + axr_w + width_between_axr_axm
    axm_y = ax1_y; axm_h = ax1_h
    axm_w = axc_w

    # ax2, placement of dendrogram 2, on the top of the heatmap
    [ax2_x, ax2_y, ax2_w, ax2_h] = [0.3,0.72,0.6,0.15] ### last one controls hight of the dendrogram
    ax2_x = axr_x + axr_w + width_between_axr_axm
    ax2_y = ax1_y + ax1_h + height_between_ax1_axc + axc_h + height_between_axc_ax2
    ax2_w = axc_w

    # axcb - placement of the color legend
    [axcb_x, axcb_y, axcb_w, axcb_h] = [0.07,0.88,0.18,0.09]

    # Compute and plot top dendrogram
    if column_method != None:
        start_time = time.time()
        d2 = dist.pdist(x.T)
        D2 = dist.squareform(d2)
        ax2 = fig.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
        Y2 = sch.linkage(D2, method=column_method, metric=column_metric) ### array-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z2 = sch.dendrogram(Y2)
        ind2 = sch.fcluster(Y2,0.7*max(Y2[:,2]),'distance') ### This is the default behavior of dendrogram
        ax2.set_xticks([]) ### Hides ticks
        ax2.set_yticks([])
        time_diff = str(round(time.time()-start_time,1))

    else:
        ind2 = ['NA']*len(column_header) ### Used for exporting the flat cluster data
        
    # Compute and plot left dendrogram.
    if row_method != None:
        start_time = time.time()
        d1 = dist.pdist(x)
        D1 = dist.squareform(d1)  # full matrix
        ax1 = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True) # frame_on may be False
        Y1 = sch.linkage(D1, method=row_method, metric=row_metric) ### gene-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z1 = sch.dendrogram(Y1, orientation='right')
        ind1 = sch.fcluster(Y1,0.7*max(Y1[:,2]),'distance') ### This is the default behavior of dendrogram
        ax1.set_xticks([]) ### Hides ticks
        ax1.set_yticks([])
        time_diff = str(round(time.time()-start_time,1))

    else:
        ind1 = ['NA']*len(row_header) ### Used for exporting the flat cluster data
        
    # Plot distance matrix.
    axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h])  # axes for the data matrix
    xt = x
    if column_method != None:
        idx2 = Z2['leaves'] ### apply the clustering for the array-dendrograms to the actual matrix data
        xt = xt[:,idx2]
        ind2 = ind2[:,idx2] ### reorder the flat cluster to match the order of the leaves the dendrogram
    if row_method != None:
        idx1 = Z1['leaves'] ### apply the clustering for the gene-dendrograms to the actual matrix data
        xt = xt[idx1,:]   # xt is transformed x
        ind1 = ind1[idx1,:] ### reorder the flat cluster to match the order of the leaves the dendrogram
    ### taken from http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python/3011894#3011894
    im = axm.matshow(xt, aspect='auto', origin='lower', cmap=cmap, norm=norm) ### norm=norm added to scale coloring of expression with zero = white or black
    axm.set_xticks([]) ### Hides x-ticks
    axm.set_yticks([])

    # Add text
    new_row_header=[]
    new_column_header=[]
    for i in range(x.shape[0]):
        if row_method != None:
            if len(row_header)<100: ### Don't visualize gene associations when more than 100 rows
                axm.text(x.shape[1]-0.5, i, '  '+row_header[idx1[i]])
            new_row_header.append(row_header[idx1[i]])
        else:
            if len(row_header)<100: ### Don't visualize gene associations when more than 100 rows
                axm.text(x.shape[1]-0.5, i, '  '+row_header[i]) ### When not clustering rows
            new_row_header.append(row_header[i])
    for i in range(x.shape[1]):
        if column_method != None:
            if len(column_header)<500:
                axm.text(i, -0.9, ' '+column_header[idx2[i]], rotation=270, verticalalignment="top") # rotation could also be degrees
                new_column_header.append(column_header[idx2[i]])
        else: ### When not clustering columns
            axmtext(i, -0.9, ' '+column_header[i], rotation=270, verticalalignment="top")
            new_column_header.append(column_header[i])

    # Plot colside colors
    # axc --> axes for column side colorbar
    if column_method != None:
        axc = fig.add_axes([axc_x, axc_y, axc_w, axc_h])  # axes for column side colorbar
        cmap_c = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        dc = numpy.array(ind2, dtype=int)
        dc.shape = (1,len(ind2)) 
        im_c = axc.matshow(dc, aspect='auto', origin='lower', cmap=cmap_c)
        axc.set_xticks([]) ### Hides ticks
        axc.set_yticks([])
    
    # Plot rowside colors
    # axr --> axes for row side colorbar
    if row_method != None:
        axr = fig.add_axes([axr_x, axr_y, axr_w, axr_h])  # axes for column side colorbar
        dr = numpy.array(ind1, dtype=int)
        dr.shape = (len(ind1),1)
        #print ind1, len(ind1)
        cmap_r = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        im_r = axr.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
        axr.set_xticks([]) ### Hides ticks
        axr.set_yticks([])

    # Plot color legend
    axcb = fig.add_axes([axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)  # axes for colorbar
    cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=norm, orientation='horizontal')
    axcb.set_title("colorkey")

    ### Render the graphic
    if len(row_header)>50 or len(column_header)>50:
        pylab.rcParams['font.size'] = 5
    else:
        pylab.rcParams['font.size'] = 8

#     pylab.savefig(filename, dpi=100) #,dpi=200
    pylab.show()

def getColorRange(x):
    """ Determines the range of colors, centered at zero, for normalizing cmap """
    vmax=x.max()
    vmin=x.min()
    if vmax<0 and vmin<0: direction = 'negative'
    elif vmax>0 and vmin>0: direction = 'positive'
    else: direction = 'both'
    if direction == 'both':
        vmax = max([vmax,abs(vmin)])
        vmin = -1*vmax
        return vmax,vmin
    else:
        return vmax,vmin


################# Create Custom Color Gradients #################
#http://matplotlib.sourceforge.net/examples/pylab_examples/custom_cmap.html

def RedBlackSkyBlue():
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),
    
             'green': ((0.0, 0.0, 0.9),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return my_cmap

def RedBlackBlue():
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return my_cmap

def RedBlackGreen():
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),
    
             'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'green':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }
    
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return my_cmap

def YellowBlackBlue():
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),
    
             'green': ((0.0, 0.0, 0.8),
                       (0.5, 0.1, 0.0),
                       (1.0, 1.0, 1.0)),
    
             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }
    ### yellow is created by adding y = 1 to RedBlackSkyBlue green last tuple
    ### modulate between blue and cyan using the last y var in the first green tuple
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return my_cmap
  
def dump(filenames,labels):
    groups = {}
    
    for i in range(len(labels)):
        groups.setdefault(int(labels[i]),[])
        groups[int(labels[i])] += [i]
    
    fs =[]
    print groups
    sm =0
    for i in range(0,len(groups)-1):
        f = [filenames[j] for j in groups[i]]
        sm += len(f)
        fs += [f]
    
    print sm, len(groups[-1])
    
    return fs

def load(st):
    
    ids =[]
    md5s = []
    names = []
    
    with open("/Users/spoulson/data/andromeda/details.txt") as f:
        for l in f:
            s = l.split('\t')
            
            #if s[1] == 'LMN' and s[2] == 'BITS':
            if s[2] == 'BITS' and (st == None or s[3].strip() in st):
                ids += [int(s[0])]
                
                names += [s[1]]
                md5s += [s[3].strip()]
    
    print names
    print md5s
    
    matrix = 1- numpy.zeros(( len(ids), len(ids) ))
    
    with open("/Users/spoulson/data/andromeda/distance_mat2.txt") as f:
        for l in f:
            
            (x,y),a =  eval(l.replace('\t',','))
            
            if x in ids and y in ids:
                x = ids.index(x)
                y = ids.index(y)
                matrix[x,y] = 1-a
                matrix[y,x] = 1-a
                #matrix[x,x] = 1
    return md5s, names, matrix
################  Default Methods ################
if __name__ == '__main__':
    
    row_method = 'average'
    column_method = 'average'
    row_metric = 'cityblock' #cosine
    column_metric = 'cityblock'
    color_gradient = 'red_white_blue'
    
    family = 'ZAccess'
    
    fs = {}
    families = set()
    flist = {}
    
    md5s, names,matrix = load(None)
    
    filename = "output"
    
    db = DBSCAN(eps=0.25, metric='precomputed').fit(1-matrix)
    st = dump(md5s,db.labels_)
    
    print len(st)
    
    for c in range(len(st)):
        print "------------" ,c,"-----------------"
        md5s, names,matrix = load(st[c])
        print "names",names
        print "md5s",md5s
        heatmap(matrix, names, names, row_method, column_method, row_metric, column_metric, color_gradient, "/Users/spoulson/data/andromeda/"+str(c)+".png")
 