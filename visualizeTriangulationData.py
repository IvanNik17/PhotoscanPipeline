# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:31:22 2018

@author: ivan
"""

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def isOnOneLine(p0,norm, dist ,p):
    dx = dist*norm[0]
    dy = dist*norm[1]
    dz = dist*norm[2]

    ex = p[0] - p0[0]
    ey = p[1] - p0[1]
    ez = p[2] - p0[2]

    q = dx*ex
    q += dy*ey
    q += dz*ez
    
    q*=q
    q/=(dx*dx + dy*dy + dz*dz)
    q/=(ex*ex + ey*ey + ez*ez)
    
    return q    
    
    
    

fig, ax = plt.subplots(1,1, subplot_kw={'projection':'3d', 'aspect':'equal'})

cam_pos = np.loadtxt("telemetrydata_pos.txt", delimiter=",")
cam_norm = np.loadtxt("telemetrydata_norm.txt", delimiter=",")

cam_perp = np.loadtxt("telemetrydata_tang1.txt", delimiter=",")

sparsePC = np.loadtxt("sparsePoints.txt", delimiter=',')

checkDist = 999

raycastDistances =[]

for k in range(0,len(cam_pos)):
        
    
    whichCam = k
    
    
    sparsePC_curr = sparsePC[sparsePC[:,0].astype(int) == whichCam,1:]
    
    cam_pos_curr = cam_pos[whichCam,:]
    cam_norm_curr = cam_norm[whichCam,:]
    cam_perp_curr = cam_perp[whichCam,:]
    
    
    isOnNormal =[]
    
    
    for i in range(0, len(sparsePC_curr)):
        
        currTest = isOnOneLine(cam_pos_curr,cam_norm_curr, checkDist ,sparsePC_curr[i,:])
    #    currTest = isOnOneLineV2(cam_pos_curr,endPoint,p)
        isOnNormal.append(currTest)
    
    isOnNormal = np.array(isOnNormal)
    
    n = 2
    nBiggest = isOnNormal.argsort()[-n:][::-1]
    
    hitPoints = sparsePC_curr[nBiggest,:]
    camPos_forDist = np.array([cam_pos_curr])
    distToCam = np.sum((hitPoints - camPos_forDist)**2, axis=1)
    
    raycastDistances.append(distToCam.mean())
    
    ax.plot(sparsePC_curr[:,0], sparsePC_curr[:,1], sparsePC_curr[:,2], 'r.')
    ax.scatter(cam_pos_curr[0], cam_pos_curr[1], cam_pos_curr[2], c='red')
    ax.quiver(cam_pos_curr[0], cam_pos_curr[1], cam_pos_curr[2], cam_norm_curr[0], cam_norm_curr[1], cam_norm_curr[2],length=4, normalize = True)
    ax.quiver(cam_pos_curr[0], cam_pos_curr[1], cam_pos_curr[2], cam_perp_curr[0], cam_perp_curr[1], cam_perp_curr[2],length=1, normalize = True, color='red')
    
    ax.plot(hitPoints[:,0], hitPoints[:,1], hitPoints[:,2], 'bo')


raycastDistances= np.array(raycastDistances)


#ax.plot(sparsePC[:,1], sparsePC[:,2], sparsePC[:,3], 'r.')
#ax.scatter(cam_pos[:,0], cam_pos[:,1], cam_pos[:,2], c='red')
#ax.quiver(cam_pos[:,0], cam_pos[:,1], cam_pos[:,2], cam_norm[:,0], cam_norm[:,1], cam_norm[:,2],length=1, normalize = True)
#ax.quiver(cam_pos[:,0], cam_pos[:,1], cam_pos[:,2], cam_perp[:,0], cam_perp[:,1], cam_perp[:,2],length=1, normalize = True, color='red')

set_axes_equal(ax)


