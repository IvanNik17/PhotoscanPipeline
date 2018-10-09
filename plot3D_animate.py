# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:56:54 2018

@author: ivan
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

    

def initializePlot3D(mainBlade_points, mainBlade_colors):
#    fig3d_anim = plt.figure()
#    
#    ax = fig3d_anim.add_subplot(1,1,1)
#    
#    ax = Axes3D(fig3d_anim)
    
    
    fig, ax = plt.subplots(1,1, subplot_kw={'projection':'3d', 'aspect':'equal'})

    ax.view_init(elev=10., azim=40)

    
    graphMain = ax.scatter(mainBlade_points[:,0], mainBlade_points[:,1], mainBlade_points[:,2], facecolors=mainBlade_colors/255, marker=".", edgecolors='none')
    graphBand, = ax.plot([0], [0], [0], 'ro')
    

    
    
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    
    
    
#    ax.axis([-axisSize, axisSize, -axisSize, axisSize])

#    ax.set_xlim(-axisSize, axisSize)
#    ax.set_ylim(-axisSize, axisSize)
#    ax.set_zlim(-axisSize, axisSize)
    
    title = ax.set_title('Blade segment')
    
    
    return title, graphBand, 
    
    
def plot3D_animate(title, graphBand, currPoints, currHeight, maxHeight):
    
    graphBand.set_data (currPoints[:,0], currPoints[:,1])
    
    graphBand.set_3d_properties(currPoints[:,2])
    
    
    
    title.set_text('Blade segment at height ' + str(currHeight) + ' from max ' +str(maxHeight))
    
    plt.pause(0.0001)  
    
def closeFigures():
    plt.close()
    

#if __name__ == '__main__': 
#    
#    import numpy as np
#    
#    title, graphDrone, graphEnv, = initializePlot(1000)
#    
#    dronePos =np.array( [345, 566]) 
#    bladePointsPos = np.array( [[456, 334]])
#
#    height = 200
#    
#    lidarRotAngle = 30
#    
#    try:
#        
#        while True:
#            plot3D_animate(title, graphDrone, graphEnv ,dronePos, bladePointsPos, height, lidarRotAngle)
#            height +=1
#            
#    except KeyboardInterrupt:
#        print("end")
    
    