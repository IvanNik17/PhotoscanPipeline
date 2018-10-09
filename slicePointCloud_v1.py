# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:55:09 2018

@author: ivan
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

from plot2D_animate import *
from plot3D_animate import *




#    Initial variables
subsamplingPC = 500


if 'pointCloud' not in  locals():
    pointCloud = np.loadtxt("outputCloud.txt", delimiter = ",")

minHeight = pointCloud[:,2].min()


#get point cloud to zero

pointCloud[:,2] -= minHeight

recHeight = pointCloud[:,2].max()

sampling_height = recHeight * 0.005


pointCloud_smaller = pointCloud.copy()
pointCloud_smaller = pointCloud[::subsamplingPC,:]

pointCloud_smaller_points = pointCloud_smaller[:,:3]
pointCloud_smaller_color = pointCloud_smaller[:,3:]



ax2D, title2D, slice2D = initializePlot()

#title3D, band3D = initializePlot3D(pointCloud_smaller_points, pointCloud_smaller_color)

allSlices = []
counter = 0

file = open("testfile.txt","w") 

for currHeight in np.arange(0, recHeight, sampling_height):
    mask_big = np.logical_and(pointCloud[:,2]>currHeight - sampling_height, pointCloud[:,2]<currHeight + sampling_height)
    mask_small = np.logical_and(pointCloud_smaller_points[:,2]>currHeight - sampling_height, pointCloud_smaller_points[:,2]<currHeight + sampling_height)
    pointCloud_oneHeight = pointCloud[mask_big ,:3]
    pointCloud_oneHeight_small = pointCloud_smaller[mask_small ,:3]
    
    numbering = np.ones([3,1])*counter 

    plot2D_animate(ax2D, title2D, slice2D, pointCloud_oneHeight, currHeight, recHeight)
    
    #    plot3D_animate(title3D, band3D, pointCloud_oneHeight_small,  currHeight, recHeight)
    
    pointCloud_oneHeight = np.concatenate([numbering,pointCloud_oneHeight.T], axis=1)
    
    file.write("%s\n" % pointCloud_oneHeight[0,:].tolist())
    file.write("%s\n" % pointCloud_oneHeight[1,:].tolist())
    file.write("%s\n" % pointCloud_oneHeight[2,:].tolist())
    
    counter+=1

    
    
    
    
file.close()

    
    
    

