#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:08:37 2021

@author: alessandro
"""

#%% Introduction
#Each run has a UUID code ==> Outer dictionary
#We have, as of today 1000 events generated in a single run ==> Numbered dictionaries
#For each event we have the amplitudes and position of the pixels
#%% Importing the libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
#%%Opening of the pickled file
dictionary=pickle.load(open("Pickles/3dreco0.pkl","rb"))
#We separate the UUID run key from the voxel dictionary
key_name, dictionary = next(iter(dictionary.items()))
#take one of the events (this will have to remain )
event= dictionary[2]['amplitude']
#convert the event dic. to numpy array
array=np.array(event)
#setting the threshold for drawing
cut=int(0.6*np.max(array))
#Plot the voxel array with a threshold
cmap = matplotlib.cm.get_cmap("viridis").copy()
cmap.set_under('none')
#plot a basic discrete heatmap of the data
#define cocrdinate arrays
x= np.arange(0.,array.shape[0],1)
y=np.arange(0.,array.shape[1],1)
z=np.arange(0.,array.shape[2],1)
#plotting the cut data
cut_array=array[array>=cut]
x_cut=np.nonzero(array>=cut)[0]
y_cut=np.nonzero(array>=cut)[1]
z_cut=np.nonzero(array>=cut)[2]
# X,Y,Z=np.meshgrid(x_cut,y_cut,z_cut)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img=ax.scatter(x_cut, y_cut, z_cut, c=cut_array.ravel(),s=2,marker='s')
plt.colorbar(img,fraction=0.025, pad=0.06)
plt.show()
