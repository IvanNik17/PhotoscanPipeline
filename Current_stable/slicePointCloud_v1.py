# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:55:09 2018

@author: ivan
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# The two animate scripts used for 2D and 3D visualization.
from plot2D_animate import *
from plot3D_animate import *


# Main slicing function with some default parameters
def slicer_v1(pointCloud, dirSave, visualization = "2D", samplingConst = 0.005, patchHeight = 0, subsampling3DVis = 500):
    
    # Get minimum height of point cloud. This expects that the point cloud is oriented with its height on the Z axis        
    minHeight = pointCloud[:,2].min()
    
    
    # Get point cloud to zero
    pointCloud[:,2] -= minHeight

    # Calculate the sampling height, which depends on the size of the blade.    
    recHeight = pointCloud[:,2].max()
    sampling_height = recHeight * samplingConst
    
    #  Make a subsampled version of the point cloud for visualization   
    pointCloud_smaller = pointCloud.copy()
    pointCloud_smaller = pointCloud[::subsampling3DVis,:]
    
    pointCloud_smaller_points = pointCloud_smaller[:,:3]
    pointCloud_smaller_color = pointCloud_smaller[:,3:]
    
    # Depending on the chosen visualization - 2D, 3D or none, the specific visualizer is initialized.    
    if visualization == "2D":
        
        ax2D, title2D, slice2D = initializePlot()
    elif visualization == "3D":
        
        title3D, band3D = initializePlot3D(pointCloud_smaller_points, pointCloud_smaller_color)
    elif visualization == "noVis":
        print("Do not visualize")

    counter = 0
    
    
    
    # For each height slice of the blade, create a new file for saving, then mask everything of the point cloud, except the height, making use of the sampling height.    
    # After that if visualization is selected, these masked points are shown
    # Finally the slice is saved in the prepared file    
    for currHeight in np.arange(0, recHeight, sampling_height):
        
        stringHeight = "{:.3f}".format(currHeight + patchHeight)
        currFile = dirSave + r"\slice_" + stringHeight + ".ler"
        
        file = open(currFile,"w") 
        
        mask_big = np.logical_and(pointCloud[:,2]>currHeight - sampling_height, pointCloud[:,2]<currHeight + sampling_height)
        
        pointCloud_oneHeight = pointCloud[mask_big ,:3]
        
        if visualization == "2D":
            plot2D_animate(ax2D, title2D, slice2D, pointCloud_oneHeight, stringHeight, recHeight)
        elif visualization == "3D":
            mask_small = np.logical_and(pointCloud_smaller_points[:,2]>currHeight - sampling_height, pointCloud_smaller_points[:,2]<currHeight + sampling_height)
            pointCloud_oneHeight_small = pointCloud_smaller[mask_small ,:3]
            plot3D_animate(title3D, band3D, pointCloud_oneHeight_small,  currHeight, recHeight)
        elif visualization == "noVis":
            print("Cutting slice " + stringHeight)
        
        numbering = np.ones([3,1])*counter 
    
        
        
        
        pointCloud_oneHeight = np.concatenate([numbering,pointCloud_oneHeight.T], axis=1)
        
        file.write("%s\n" % pointCloud_oneHeight[0,:].tolist())
        file.write("%s\n" % pointCloud_oneHeight[1,:].tolist())
        file.write("%s\n" % pointCloud_oneHeight[2,:].tolist())
        
        counter+=1
        
        file.close()

    
    
    
if __name__ == '__main__':     
    if 'pointCloud' not in  locals():
        pointCloud = np.loadtxt("outputCloud.txt", delimiter = ",")
    
    dirSave=r"C:\Users\ivan\TestDir"
    slicer_v1(pointCloud,dirSave)
    
    
    

