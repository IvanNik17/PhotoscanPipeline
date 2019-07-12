# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:44:54 2019

@author: ivan
"""

import numpy as np


import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial import ConvexHull, convex_hull_plot_2d

import math


import time


from open3d import *

import vispy.scene
from vispy.scene import visuals


def visualize3D(titleV,pointCloud, rgbMap):
    
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title = titleV)
    view = canvas.central_widget.add_view()
    
    
    scatter = visuals.Markers()
    scatter.set_data(pointCloud, edge_width=0.1, edge_color=rgbMap, face_color=rgbMap, size=5, scaling=False)
    
    view.add(scatter)
    
    view.camera = 'arcball'


def circ_resvec(vals, axis=None):
    """Calculate the mean resultant vector length for circular data.
    Parameters
    ----------
    vals : array-like
        The array of angles
    axis : int (default=None)
        The axis along which to take the mean
    Returns
    -------
    out : np.ndarray
        The resulting numpy array of resultant vector lengths
    References
    ----------
    http://www.jstatsoft.org/v31/i10
    """
    
    alpha = np.array(vals, dtype='f8')
    # sum of cos & sin angles
    t = np.exp(1j * alpha)
    r = np.sum(t, axis=axis)
    # obtain length 
    r = np.abs(r) / alpha.shape[axis]
    return r    

def circ_var(vals, axis=None):
    """Computes circular variance for circular data.
    Parameters
    ----------
    vals : array-like
        The array of angles
    axis : int (default=None)
        The axis along which to take the variance
    Returns
    -------
    out : np.ndarray
        The calculated circular variance
    References
    ----------
    http://www.jstatsoft.org/v31/i10
    """

    alpha = np.array(vals, dtype='f8')
    var = 1. - circ_resvec(alpha, axis=axis)
    return var

def circ_std(vals, axis=None):
    """Computes circular standard deviation for circular data.
    Parameters
    ----------
    vals : array-like
        The array of angles
    axis : int (default=None)
        The axis along which to take the standard deviation
    Returns
    -------
    out : np.ndarray
        The calculated circular standard deviation
    References
    ----------
    http://www.jstatsoft.org/v31/i10
    http://en.wikipedia.org/wiki/Directional_statistics#Measures_of_location_and_spread
    """

    var = circ_var(vals, axis=axis)
    if var<0.0000001:
        var =0
    std = np.sqrt(abs(-2*np.log(1 - var+0.000000001)))
    return std


def circ_stdNew(samples,low, high):
    
    samples = (samples - low)*2.*np.pi / (high - low)
    
    S = np.sin(samples).mean(axis=0)
    C = np.cos(samples).mean(axis=0)
    R = np.hypot(S, C)
#    var = ((high - low)/2/np.pi) * np.sqrt(np.abs(-2*np.log(R)) )
    
    var = ((high - low)/2/np.pi) * np.sqrt(2*(1-R + 0.0000001) )
    std = var
    return std

def set_axes_equal(ax):


    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    theta = math.radians(theta)
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    
    
    
def makeSpherePiece(theta,phi,r):
    X = r * np.cos(theta) * np.sin(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.ones_like(theta) * np.cos(phi)     
   
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    
    return np.array([x,y,z])



def ClosestPointOnLine(a, b, p):

    ap = p-a
    ab = b-a
    result = a + np.multiply(np.expand_dims(ab, axis=0),np.expand_dims(np.dot(ap,ab)/np.dot(ab,ab), axis=0).T)
    
    return result

def meanAngle(angleArr):
    theta_cos_mean = np.cos(angleArr).mean()
    theta_sin_mean = np.sin(angleArr).mean()
    
    meanAngle = 0
    if theta_cos_mean>0 and theta_sin_mean>0:
        meanAngle= np.rad2deg(np.arctan(theta_sin_mean/theta_cos_mean))
        
    elif theta_cos_mean<0:
        meanAngle= np.rad2deg(np.arctan(theta_sin_mean/theta_cos_mean)) + 180
        
    elif theta_sin_mean <0 and theta_cos_mean > 0:
        meanAngle= np.rad2deg(np.arctan(theta_sin_mean/theta_cos_mean)) + 360
        
    return meanAngle

# Initialize the hemisphere and the containers

quadrantCounter = 0

allPoints=[]


theta_delta = np.pi/4
phi_delta = np.pi/8

allMeanPoints = []

allMeanAngles = []

for i in np.arange(0,2*np.pi,theta_delta ):
    for j in np.arange(0,np.pi/2,phi_delta):
        THETA, PHI = np.ogrid[i:i+theta_delta:10j, j:j +phi_delta:10j]
        
        r=1
        newPoints = makeSpherePiece(THETA,PHI,r)
        
        currQuadrant = np.ones([1,len(newPoints[0,:])])*quadrantCounter
        
#        ax.scatter(newPoints[0,:],newPoints[1,:],newPoints[2,:])
        
        allMeanPoints.append(newPoints.mean(axis=1))
        
        newPoints = np.concatenate([newPoints, currQuadrant],axis=0)
        meanTheta = np.rad2deg(THETA.mean())
        meanPhi = np.rad2deg(PHI.mean())
#        meanTheta = meanAngle(THETA)
#        meanPhi = meanAngle(PHI)
        
        allMeanAngles.append([meanTheta, meanPhi])
        
        quadrantCounter +=1
        
        allPoints.append(newPoints.T)

allPoints_arr = np.concatenate( allPoints, axis=0 )

allMeanPoints_arr = np.array( allMeanPoints)

allMeanAngles_arr = np.array(allMeanAngles)

normal_starting = np.array([0,0,1])

scale =162.993


print('Import data', flush=True) 

if 'pointCameraVis' not in  locals():
    pointCameraVis = np.loadtxt(r"F:\Ivan\Paper_findNoiseOnMesh\Results\Bunny\vertsAngleOrient.txt")

cameraPos = np.loadtxt(r"F:\Ivan\Paper_findNoiseOnMesh\Reconstructions\OLD_OBJECTS\Bunny\ForEnvMethods\cameraPos.txt", delimiter = ',')

if 'vertexNormals' not in  locals():
    vertexNormals = np.loadtxt(r"F:\Ivan\Paper_findNoiseOnMesh\Reconstructions\OLD_OBJECTS\Bunny\object_norms.txt", delimiter = ',')

if 'bladeVerts' not in  locals():
    bladeVerts = np.loadtxt(r"F:\Ivan\Paper_findNoiseOnMesh\Reconstructions\OLD_OBJECTS\Bunny\object.txt", delimiter = ',')
        
        
vertices = bladeVerts[:,:3]  

vertices*=scale
cameraPos*=scale     

print('Start calculations', flush=True) 

#allPointsSTD = np.zeros([ len(vertices), 2])

allPointsHemisphereArea = np.zeros([ len(vertices), 1])   
fullArea = np.array([[0,360,0,360],[0,0,90,90]]).T
hull_fullarea = ConvexHull(fullArea)
      
#len(vertices)
start = time.time()     
#489
#fig, ax = plt.subplots(1,1, subplot_kw={'projection':'3d', 'aspect':'equal'})
for i in range(0,len(vertices)):

    
    
    center = vertices[i,:]
    normal = vertexNormals[i,:]
    
    
    
    tangent = np.array([normal[2],0,-normal[0]])/np.sqrt(normal[0]*normal[0] + normal[2]*normal[2])
    angle = np.arccos(np.dot(normal_starting, normal) / (np.linalg.norm(normal_starting) * np.linalg.norm(normal) ));
    bidirect= np.cross(normal,tangent)
    rotMat = rotation_matrix(bidirect, np.rad2deg(angle))
    
    
    curr_allMeanPoints = allMeanPoints_arr.copy()
    
    
    curr_allMeanPoints = np.dot(rotMat,curr_allMeanPoints[:,:3].T) 
    curr_allMeanPoints =curr_allMeanPoints.T


    curr_allMeanPoints = curr_allMeanPoints + center
    
    
#    ax.scatter(curr_allMeanPoints[:,0],curr_allMeanPoints[:,1],curr_allMeanPoints[:,2],c='red',s=20)
    
    
#    ax.scatter(vertices[::100,0],vertices[::100,1],vertices[::100,2],c='green',s=5)
    
    currSeenCams_all = pointCameraVis[i,:]
    
    currSeenCams = currSeenCams_all[currSeenCams_all!=-1]
    
    if len(currSeenCams) <=1 :
#        allPointsSTD[i,:] =  [0,0]
        
        allPointsHemisphereArea[i] = 0
        
    else:
    
        allHitP_onHemisphere = np.zeros([len(currSeenCams),2])
        
    
        
        for j in range(0, len(currSeenCams)):
            
            currCamera = cameraPos[j,:]
            
            result = ClosestPointOnLine(center, currCamera, curr_allMeanPoints)
    
            a_min_b = currCamera - result
            distances = np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))
            
            minInd = np.argmin(distances)
            
    #        ax.scatter(curr_allMeanPoints[minInd,0],curr_allMeanPoints[minInd,1],curr_allMeanPoints[minInd,2],c='green',s=30)
            
    #        ax.scatter(currCamera[0],currCamera[1],currCamera[2],c='blue',s=20)
    
    #        ax.plot([center[0], currCamera[0]], [center[1], currCamera[1]],zs=[center[2], currCamera[2]], c= 'r')
            
            allHitP_onHemisphere[j,:] = allMeanAngles_arr[minInd,:]
        
            
    
    #    testingPoint = np.array([0.3,0.5,2])
#        allPointsSTD[i,:] = [circ_stdNew(np.deg2rad(allHitP_onHemisphere[:,0]),0, 6.28), circ_stdNew(np.deg2rad(allHitP_onHemisphere[:,1]),0,1.57)]
        try:
            
            hull = ConvexHull(allHitP_onHemisphere)
            allPointsHemisphereArea[i] = hull.area/hull_fullarea.area
        except:
            allPointsHemisphereArea[i] = 0
    #    set_axes_equal(ax)    

print("Finished in " + str(time.time()-start) )




#plt.scatter([0,360,0,360],[0,0,90,90],c='red')
#plt.scatter(allHitP_onHemisphere[:,0],allHitP_onHemisphere[:,1])
#for simplex in hull.simplices:
#    plt.plot(allHitP_onHemisphere[simplex, 0], allHitP_onHemisphere[simplex, 1], 'k-')
#    
#
#for simplex in hull_fullarea.simplices:
#    plt.plot(fullArea[simplex, 0], fullArea[simplex, 1], 'k-')
    


#isThereParallelView = np.zeros([len(pointCameraVis),1])
#for k in range(0, len(pointCameraVis)):
#    currPoint = pointCameraVis[k,:]
#    currPoint[currPoint == -1]=np.nan
#    if ( np.abs(0 - np.nanmin(currPoint))<30 or np.abs(180 - np.nanmax(currPoint))<30  ):
#        isThereParallelView[k] = 1

#colorMap = np.ones([len(allPointsSTD),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#
#image_rgb[np.logical_and(allPointsSTD[:,0]<np.sqrt(2)*0.5, allPointsSTD[:,1]<np.sqrt(2)*0.5),:] = [1,0,0,1]
#visualize3D("Color Entropy", vertices, image_rgb)



#colorMap = np.ones([len(isThereParallelView),1])
#image_rgb = np.concatenate([isThereParallelView,isThereParallelView,isThereParallelView,colorMap], axis=1)
#
##image_rgb[np.logical_and(allPointsSTD[:,0]<np.sqrt(2)*0.5, allPointsSTD[:,1]<np.sqrt(2)*0.5),:] = [1,0,0,1]
#visualize3D("Color Entropy", vertices, image_rgb)