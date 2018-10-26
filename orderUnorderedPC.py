# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:52:00 2018

@author: ivan
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
#import networkx as nx
#
#import pandas as pd


def rotateBlade(bladeArray, angle):
    
    angle = -angle
    qx = 0 + np.cos(np.radians(angle)) * (bladeArray[:,0] - 0) - np.sin(np.radians(angle)) * (bladeArray[:,1] - 0)
    qy = 0 + np.sin(np.radians(angle)) * (bladeArray[:,0] - 0) + np.cos(np.radians(angle)) * (bladeArray[:,1] - 0)
    
    

    
    newCoord = np.array([qx,qy]).T

    return newCoord


pointCloud = np.loadtxt("testfile2.txt", delimiter=",")
##
pointCloud = pointCloud[:2,1:]
pointCloud = pointCloud.T


pointCloud = rotateBlade(pointCloud, -70)
#
#fig, ax = plt.subplots(1,1, subplot_kw={'aspect':'equal'})
##
#ax.plot(pointCloud[:,0], pointCloud[:,1], 'r.')

x = pointCloud[:,0]
y = pointCloud[:,1]
#
#total_bins = 1000
#bins = np.linspace(x.min(),x.max(), total_bins)
#
#delta = bins[1]-bins[0]
#idx  = np.digitize(x,bins)
#running_median = [np.median(y[idx==k]) for k in range(total_bins)]
#
#
##plt.scatter(x,y,color='k',alpha=.2,s=2)
#
#newX = bins-delta/2
#newY = running_median
#plt.plot(newX[:],newY[:])
#plt.axis('equal')
#plt.show()

#bins = 1000
#df = pd.DataFrame({'X' : x, 'Y' : y})
#data_cutY = pd.cut(df.Y,bins) 
#data_cutX = pd.cut(df.X,bins) 
#
#
#grp = df.groupby(by = data_cutX)
#ret = grp.aggregate(np.median)  
#
#
#
#ret = ret.dropna(axis = 0, thresh  = 2)
#
#
#
#
#plt.scatter(df.X,df.Y,color='k',alpha=.2,s=2)
#plt.plot(ret.X[:100],ret.Y[:100],lw=2,alpha=.8)
#plt.show()
#
#plt.axis("equal")


clf = NearestNeighbors(10).fit(pointCloud)

distances, indices = clf.kneighbors(pointCloud)


thinnedPC = np.zeros_like(pointCloud)

pointCloud_copy =pointCloud.copy()

for i in range(0, 1):
    temp_indices = indices[0,:]
    
    
    temp_subsetPC = pointCloud_copy[temp_indices,:]
    temp_meanPoint = temp_subsetPC.mean(axis=0)
    thinnedPC[i,:] = temp_meanPoint
    


#plt.plot(pointCloud[:,0],pointCloud[:,1],'r.')
#
#subsetPC = pointCloud[test,:]
#
#plt.plot(thinnedPC[:,0],thinnedPC[:,1],'bo')





#
#G = clf.kneighbors_graph()
#
#T = nx.from_scipy_sparse_matrix(G)
#
#
#order = list(nx.dfs_preorder_nodes(T))
##
#x = pointCloud[:,0]
#y = pointCloud[:,1]
#
#xx = x[order]
#yy = y[order]
#
#
#fig2, ax2 = plt.subplots(1,1, subplot_kw={'aspect':'equal'})
#
#ax2.plot(xx, yy)
#
#paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(pointCloud))]
#
#
#mindist = np.inf
#minidx = 0
#
#for i in range(len(pointCloud)):
#    p = paths[i]           # order of nodes
#    ordered = pointCloud[p]    # ordered nodes
#    # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
#    cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
#    if cost < mindist:
#        mindist = cost
#        minidx = i
#        
#x = pointCloud[:,0]
#y = pointCloud[:,1]
#        
#opt_order = paths[minidx]        
#xx2 = x[opt_order]
#yy2 = y[opt_order]
#
#
#fig3, ax3 = plt.subplots(1,1, subplot_kw={'aspect':'equal'})
#
#ax3.plot(xx2, yy2)





#x = np.linspace(0, 2 * np.pi, 100)
#y = np.sin(x)
#
##plt.plot(x, y)
##plt.show()
#
#idx = np.random.permutation(x.size)
#x = x[idx]
#y = y[idx]
#
#plt.plot(x, y)
#plt.show()
#
#points = np.c_[x, y]
#
#clf = NearestNeighbors(2).fit(points)
#G = clf.kneighbors_graph()
#
#T = nx.from_scipy_sparse_matrix(G)
#
#order = list(nx.dfs_preorder_nodes(T, 0))
#
#
#xx = x[order]
#yy = y[order]
##
#plt.plot(xx, yy)
#plt.show()