# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:09:54 2018

@author: ivan
"""

import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from roughnessFunctions import *

from open3d import *

import vispy.scene
from vispy.scene import visuals

import time

def visualize3D(titleV,pointCloud, rgbMap):
    
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, title = titleV)
    view = canvas.central_widget.add_view()
    
    
    scatter = visuals.Markers()
    scatter.set_data(pointCloud, edge_width=0.1, edge_color=rgbMap, face_color=rgbMap, size=5, scaling=False)
    
    view.add(scatter)
    
    view.camera = 'arcball'
    
    
meshPath = r"F:\Ivan\Paper_findNoiseOnMesh\Reconstructions\cups\whiteFlower_newer"    




if 'objVerts' not in  locals():
    bladeVerts = np.loadtxt(meshPath + r"\object.txt", delimiter = ',')
    
    
if 'objFaces' not in  locals():
    bladeFaces = np.loadtxt(meshPath + r"\object_faces.txt", delimiter = ',')
    
    
if 'objNormals' not in  locals():
    objNormals = np.loadtxt(meshPath + r"\object_norms.txt", delimiter = ',')


#scaleFactor = 9964.275

scaleFactor = 317.172


blade_verts = bladeVerts[:,:3]*scaleFactor
blade_norms = objNormals
blade_colors = bladeVerts[:,3:6]

blade_faces = bladeFaces
blade_faces = blade_faces.astype(int)


#Input variables

#Saliency:
saliency_a= 1;

#Entropy:
entropy_knn = 80

#Difference of Normals:

don_radPercent = 0.02


#Point Density and Color Uniformity:
pdcu_percentChange = [0.1,0.15,0.15]
pdcu_num_percentChange = 0.5
pdcu_percentFromMaxNeighbour = 0.3




start = time.time()
saliencyMap = meshSaliency(blade_verts, blade_faces,saliency_a)

saliencyNonNan = np.isnan(saliencyMap)

saliencyMap[saliencyNonNan[:,0],:] = 0

end = time.time()
print("Saliency time: " + str( (end - start)/60) )


start = time.time()
entropyMap = meshEntropy(blade_verts, blade_colors,entropy_knn)

end = time.time()
print("Entropy time: " + str( (end - start)/60) )

start = time.time()


donMap = meshDifferenceOfNormals(blade_verts, blade_norms,don_radPercent)
donMapNonNan = np.isnan(donMap)

donMap[donMapNonNan[:,0],:] = 0
end = time.time()
print("DON time: " + str( (end - start)/60) )


start = time.time()
pdcuMap = meshPointDensityAndColorUniformity(blade_verts, blade_colors,pdcu_percentChange,pdcu_num_percentChange,pdcu_percentFromMaxNeighbour)
end = time.time()
print("Pdcu time: " + str( (end - start)/60) )

#
entropyMap_norm = (entropyMap - np.min(entropyMap))/(np.max(entropyMap) - np.min(entropyMap))
saliencyMap_norm = (saliencyMap - np.min(saliencyMap))/(np.max(saliencyMap) - np.min(saliencyMap))
donMap_norm = (donMap - np.min(donMap))/(np.max(donMap) - np.min(donMap))
pdcuMap_norm = (pdcuMap - np.min(pdcuMap))/(np.max(pdcuMap) - np.min(pdcuMap))
#
#
#entropyMap_norm_centered= entropyMap_norm - entropyMap_norm.mean()
#saliencyMap_norm_centered= saliencyMap_norm - saliencyMap_norm.mean()
#donMap_norm_centered= donMap_norm - donMap_norm.mean()
#pdcuMap_norm_centered= pdcuMap_norm - pdcuMap_norm.mean()


#-----------------------------


#test = np.concatenate([entropyMap_norm_centered, saliencyMap_norm_centered, donMap_norm_centered, pdcuMap_norm_centered], axis=1)
#
#cov_mat = np.cov(test.T)
#eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
#eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
#eig_pairs.sort()
#eig_pairs.reverse()
#
#matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
#                      eig_pairs[1][1].reshape(4,1)))
#
#test_proj = test.dot(matrix_w)


#visualize3D("Colored Model",blade_verts, blade_colors/255)
#
#
#colorMap = np.ones([len(entropyMap_norm),1])
#image_rgb = np.concatenate([entropyMap_norm,entropyMap_norm,entropyMap_norm,colorMap], axis=1)
#visualize3D("Color Entropy", blade_verts, image_rgb)
####
####
#colorMap = np.ones([len(saliencyMap_norm),1])
#image_rgb = np.concatenate([saliencyMap_norm,saliencyMap_norm,saliencyMap_norm,colorMap], axis=1)
#visualize3D("Color Entropy", blade_verts, image_rgb)
####
#colorMap = np.ones([len(donMap_norm),1])
#image_rgb = np.concatenate([donMap_norm,donMap_norm,donMap_norm,colorMap], axis=1)
#visualize3D("Color Entropy", blade_verts, image_rgb)
####
#colorMap = np.ones([len(pdcuMap_norm),1])
#image_rgb = np.concatenate([pdcuMap_norm,pdcuMap_norm,pdcuMap_norm,colorMap], axis=1)
#visualize3D("Color Entropy", blade_verts, image_rgb)
##
##
#colorMap = np.ones([len(entropyMap_norm),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#image_rgb[entropyMap_norm[:,0] < 0.8,:] = [1,0,0,1]
#visualize3D("Color Entropy", blade_verts, image_rgb)

#
#colorMap = np.ones([len(saliencyMap_norm),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#image_rgb[saliencyMap_norm[:,0] > 0.5,:] = [1,0,0,1]
#visualize3D("Color Entropy", blade_verts, image_rgb)


#colorMap = np.ones([len(donMap_norm),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#
#image_rgb[donMap_norm[:,0] >0.15,:] = [1,0,0,1]
#visualize3D("Difference of Normals",blade_verts, image_rgb)

#------------------------------

#visualize3D("Colored Model",blade_verts, blade_colors/255)
#
#
#colorMap = np.ones([len(saliencyMap),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#
#image_rgb[saliencyMap[:,0] > saliencyMap.max()*0.5,:] = [1,0,0,1]
#visualize3D("Local Curvature",blade_verts, image_rgb)
#
#
#colorMap = np.ones([len(entropyMap),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#
#image_rgb[entropyMap[:,0] > entropyMap.max()*0.8,:] = [1,0,0,1]
#visualize3D("Color Entropy", blade_verts, image_rgb)
#
#
#colorMap = np.ones([len(donMap),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#
#image_rgb[donMap[:,0] >0.25,:] = [1,0,0,1]
#visualize3D("Difference of Normals",blade_verts, image_rgb)
#
#
#
#saliencyNoise = saliencyMap[:,0] > saliencyMap.max()*0.5
#entropyNoise = entropyMap[:,0] < entropyMap.max()*0.8
#donNoise = donMap[:,0] > donMap.max()*0.15
#finalNoise = np.logical_or(np.logical_or(saliencyNoise, entropyNoise),donNoise)
#image_rgb[finalNoise,:] = [1,0,0,1]
#visualize3D("Final Combination",blade_verts, image_rgb)






#saliencyNoise = saliencyMap[:,0] > saliencyMap.max()*0.5
#entropyNoise = entropyMap[:,0] < entropyMap.max()*0.8
#donNoise = donMap[:,0] > donMap.max()*0.15
#
#votingArr = np.zeros([len(saliencyMap),1])
#votingArr[saliencyNoise,:] +=1
#votingArr[entropyNoise,:] +=1
#votingArr[donNoise,:] +=1
#
#colorMap = np.ones([len(votingArr),1])
#image_rgb = np.concatenate([colorMap,colorMap,colorMap,colorMap], axis=1)
#image_rgb[votingArr[:,0] == 3,:] = [0,1,0,1]
#image_rgb[votingArr[:,0] == 2,:] = [0,0,1,1]
#image_rgb[votingArr[:,0] == 1,:] = [1,0,0,1]
#
#visualize3D(blade_verts, image_rgb)



#image_rgb[saliencyMap[:,0] > saliencyMap.max()*0.5,:] = [1,0,0,1]

#image_rgb[entropyMap[:,0] > entropyMap.max()*0.8,:] = [1,0,0,1]

#image_rgb[donMap[:,0] > donMap.max()*0.15,:] = [1,0,0,1]

#saliencyNoise = saliencyMap[:,0] > saliencyMap.max()*0.5
#entropyNoise = entropyMap[:,0] < entropyMap.max()*0.8
#donNoise = donMap[:,0] > donMap.max()*0.15

#votingArr = np.zeros([len(saliencyMap),1])
#
#votingArr[saliencyNoise,:] =+1
#votingArr[entropyNoise,:] =+1
#votingArr[donNoise,:] =+1
#
#votingArr[votingArr[:,1] < 3] = False
#votingArr[votingArr[:,1] == 3] = True


#finalNoise = np.logical_or(np.logical_or(saliencyNoise, entropyNoise),donNoise)
##
##
#image_rgb[finalNoise,:] = [1,0,0,1]



#canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
#view = canvas.central_widget.add_view()
#
#
#scatter = visuals.Markers()
#scatter.set_data(blade_verts, edge_width=0.1, edge_color=image_rgb, face_color=image_rgb, size=5, scaling=False)
#
#view.add(scatter)
#
#view.camera = 'arcball'




#image_rgb = np.concatenate([colorMap,colorMap,colorMap], axis=1)
##image_rgb[image[:,0] > allEntropy.max()*0.7,:] = [1,0,0]
#
#
#pcd = PointCloud()
#pcd.points = Vector3dVector(blade_verts)           
#pcd.colors = Vector3dVector(image_rgb)
#downpcd = voxel_down_sample(pcd, voxel_size = 8)
#downpcd_points = np.asarray(downpcd.points)
#downpcd_colors = np.asarray(downpcd.colors)
##
##
#fig, ax = plt.subplots(1,1, subplot_kw={'projection':'3d', 'aspect':'equal'})
#ax.scatter(downpcd_points[:,0], downpcd_points[:,1], downpcd_points[:,2], c=downpcd_colors[:,0], marker=".", edgecolors='none')