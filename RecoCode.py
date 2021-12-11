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
import plot3d
import pickle
#%%Opening of the pickled file
dictionary=pickle.load(open("Pickles/3dreco0.pkl","rb"))
#We separate the UUID run key from the voxel dictionary
key_name, dictionary = next(iter(dictionary.items()))
#take one of the events (this will have to remain )
event= dictionary[6]['amplitude']
#convert the event dic. to numpy array
array=np.array(event)
#setting the threshold for drawing
cut=int(0.7*np.max(array))
cmap = matplotlib.cm.get_cmap("viridis").copy()
cmap.set_under('none')
#plot a basic discrete heatmap of the data
#define the discrete coordinates
X, Y, Z = np.mgrid[:30, :30, :136]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img=ax.scatter(X+0.5, Y+0.5, Z+0.5, c=array.ravel(),s=1.5,vmin=cut)
plt.colorbar(img)
plt.show()