# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:52:55 2019

@author: ivan
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

import vispy.scene
from vispy.scene import visuals






def visualize3D(titleV,pointCloud, rgbMap):
    
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title = titleV)
    view = canvas.central_widget.add_view()
    
    
    scatter = visuals.Markers()
    scatter.set_data(pointCloud, edge_width=0.1, edge_color=rgbMap, face_color=rgbMap, size=5, scaling=False)
    
    view.add(scatter)
    
    view.camera = 'arcball'

                    
                       
pathResults = r"F:\Ivan\Paper_findNoiseOnMesh\Results\Angel"

pathVertices =r"F:\Ivan\Paper_findNoiseOnMesh\Reconstructions\OLD_OBJECTS\Angel"



labeledGT = np.loadtxt(pathResults + "\labeledPointCloud.txt") 


verticesColor = np.loadtxt(pathVertices + "\object.txt",delimiter = ',')

vertices = verticesColor[:,:3]
colors = verticesColor[:,3:]

#visualize3D("Texture Color", vertices, colors/255)  


## Pure mesh analysis
map_entropy = np.loadtxt(pathResults + "\entropy.txt")
map_entropy_invert = 1 - map_entropy
map_saliency = np.loadtxt(pathResults + "\saliency.txt")
map_don = np.loadtxt(pathResults + "\don.txt")
map_pdcu = np.loadtxt(pathResults + "\pdcu.txt")

map_entropy_invert_mean = map_entropy_invert.mean()
map_entropy_invert_std = map_entropy_invert.std()





## Mesh and capturing environment analysis
map_avncn = np.loadtxt(pathResults + r"\vertsAngleOrient.txt")
map_avncn[map_avncn ==-1] = np.nan
allVerts_maxAngle =np.nanmax(map_avncn,axis=1)
allVerts_maxAngle = np.nan_to_num(allVerts_maxAngle)
closenessToParallel = 180 - allVerts_maxAngle
closenessToParallel_norm = (closenessToParallel - np.min(closenessToParallel))/(np.max(closenessToParallel) - np.min(closenessToParallel) + 0.00001)


#map_hemisphere = np.loadtxt(pathResults + r"\hemisphereSTD.txt",delimiter = ",")
#map_hemisphere_norm = np.zeros_like(map_hemisphere)
#map_hemisphere_norm[:,0] = (map_hemisphere[:,0] - np.min(map_hemisphere[:,0]))/(np.max(map_hemisphere[:,0]) - np.min(map_hemisphere[:,0]) + 0.00001)
#map_hemisphere_norm[:,1] = (map_hemisphere[:,1] - np.min(map_hemisphere[:,1]))/(np.max(map_hemisphere[:,1]) - np.min(map_hemisphere[:,1]) + 0.00001)
#
#map_hemisphere_norm = 1- map_hemisphere_norm
#
#map_hemisphere_norm_theta = map_hemisphere_norm[:,0]
#map_hemisphere_norm_phi = map_hemisphere_norm[:,1]

map_hemisphereArea = np.loadtxt(pathResults + r"\seenAreaOnHemisphere.txt")
map_hemisphereArea = (map_hemisphereArea - np.min(map_hemisphereArea))/(np.max(map_hemisphereArea) - np.min(map_hemisphereArea) + 0.00001)
map_hemisphereArea_inverse = 1-map_hemisphereArea

map_vertsInFocus = np.loadtxt(pathResults + r"\vertsInFocus.txt")
numCams = len(map_vertsInFocus[1,:])
sumFocusForEachVert = np.sum(map_vertsInFocus,axis=1)
sumFocusForEachVert_norm= (sumFocusForEachVert - np.min(sumFocusForEachVert))/(np.max(sumFocusForEachVert) - np.min(sumFocusForEachVert) + 0.00001)
#sumFocusForEachVert_inverted = 1-sumFocusForEachVert_norm


map_vertsSeen = np.loadtxt(pathResults + r"\vertsSeen.txt")
map_vertsSeen_norm = (map_vertsSeen - np.min(map_vertsSeen))/(np.max(map_vertsSeen) - np.min(map_vertsSeen) + 0.00001)
map_vertsSeen_inverted = 1 - map_vertsSeen_norm


map_projKeypoints = np.loadtxt(pathResults + r"\projKeypoints.txt")
map_projKeypoints_norm = (map_projKeypoints - np.min(map_projKeypoints))/(np.max(map_projKeypoints) - np.min(map_projKeypoints) + 0.00001)
map_projKeypoints_inverted = 1 - map_projKeypoints_norm



#arr_color = cm.ScalarMappable(cmap = 'bwr').to_rgba(sumFocusForEachVert_inverted[:,0], bytes = True)
#visualize3D("normal", vertices, arr_color/255)
#
#
###Results
#
### no view close to parallel + low circle std.dev. in theta and phi + not in focus in most cameras + not seen by most cameras + not part of keypoints or their area
#noiseCertaintyMap_env = (closenessToParallel_norm + map_hemisphere_norm_theta + map_hemisphere_norm_phi + sumFocusForEachVert_inverted + map_vertsSeen_inverted + map_projKeypoints_inverted)/6
#
#arr_color = cm.ScalarMappable(cmap = 'bwr').to_rgba(noiseCertaintyMap_env, bytes = True)
#visualize3D("Noise Certainty Environment", vertices, arr_color/255)
#
### Low entropy + high saliency + high don + high pdcu
#noiseCertaintyMap_mesh = (map_entropy_invert + map_saliency + map_don + map_pdcu)/4
##
#arr_color = cm.ScalarMappable(cmap = 'bwr').to_rgba(noiseCertaintyMap_mesh, bytes = True)
#visualize3D("Noise Certainty Mesh", vertices, arr_color/255)
#noiseCertaintyMap = (closenessToParallel_norm + map_hemisphere_norm_theta + map_hemisphere_norm_phi + sumFocusForEachVert_inverted + map_vertsSeen_inverted + map_projKeypoints_inverted + map_entropy_invert + map_saliency + map_don + map_pdcu)/10
#
#
#arr_color = cm.ScalarMappable(cmap = 'seismic').to_rgba(noiseCertaintyMap, bytes = True)
#visualize3D("Noise Certainty Combined", vertices, arr_color/255)
#
#
#colorMap = colors.copy()
#colorMap[noiseCertaintyMap>0.3,:] = [255,0,0]
#visualize3D("Noise Certainty Combined", vertices, colorMap/255)         




# learn stuff



labeledGT = np.expand_dims(labeledGT,axis=1)
closenessToParallel_norm = np.expand_dims(closenessToParallel_norm,axis=1)
#map_hemisphere_norm_theta = np.expand_dims(map_hemisphere_norm_theta,axis=1)
#map_hemisphere_norm_phi = np.expand_dims(map_hemisphere_norm_phi,axis=1)
sumFocusForEachVert_norm = np.expand_dims(sumFocusForEachVert_norm,axis=1)
map_vertsSeen_inverted = np.expand_dims(map_vertsSeen_inverted,axis=1)
map_projKeypoints_inverted = np.expand_dims(map_projKeypoints_inverted,axis=1)
map_entropy_invert = np.expand_dims(map_entropy_invert,axis=1)
map_saliency = np.expand_dims(map_saliency,axis=1)
map_don = np.expand_dims(map_don,axis=1)
map_pdcu = np.expand_dims(map_pdcu,axis=1)
map_hemisphereArea_inverse = np.expand_dims(map_hemisphereArea_inverse,axis=1)

allData = np.concatenate([labeledGT,closenessToParallel_norm,map_hemisphereArea_inverse,sumFocusForEachVert_norm,map_vertsSeen_inverted,map_projKeypoints_inverted,map_entropy_invert,map_saliency,map_don,map_pdcu],axis=1)

