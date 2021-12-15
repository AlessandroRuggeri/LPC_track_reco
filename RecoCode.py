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
import pickle
import numpy as np
import math as mt
import random
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
#%%Functions for frequently used operations
#computing the local mean
#Currently the hits are chosen for the computation based on the X index!!!
def loc_mean(u,closest,weight):
#compute the average
    m=np.average(closest,axis=0,weights=weight)
    return m

#Computing the local covariant matrix
def loc_cov(m,closest,weight):
#compute the point-m differences
    closest=closest-m
#define the covariance matrix array
    sigma = np.zeros_like(np.outer(closest[0],closest[0]),dtype=float)
#iterate to get the sum at the numerator
    for i in range(closest.shape[0]):
        sigma+=np.outer(closest[i],closest[i])*weight[i]
#divide by the sum of the weights
    sigma = sigma/np.sum(weight)
    return sigma

#Computing the mean shift (see if it can be incorporated in another function)
def mean_shift(u,closest,weight):
    return loc_mean(u,closest,weight)-u
#computing the weight vector
#We take the starting point and the N closest for which to compute the weights
def weight(u,hits):
    #bandwidth parameter to be chosen by input
    h=0.05
    w = np.zeros(hits.shape[0])
    for i in range(hits.shape[0]):
    #placeholder weight formula
        w[i] = (cut_array[i]/((2*mt.pi)**(3/2)*h**3))*mt.exp((-1/(2*h**3))*np.dot((hits[i]-u),(hits[i]-u)))
    return w

#Computing the set of N closest points
def N_closest(u,N):
    dist = np.zeros(X.shape[0])
#computing the distance between u and each X[i]
    for i in range(X.shape[0]):
        dist[i]=mt.sqrt(np.dot(X[i]-u,X[i]-u))
#sorting the distance array 
    dist = np.argsort(dist,axis=0)
#selecting the closest N (except for u itself)
    dist = dist[0:N]
#slicing the X matrix over the closest indices
    return np.take(X,dist,axis=0)
#Plotting of the hits

#Perform the LPC cycle
def lpc_cycle(u,N):
#for each starting point u
    #finding the closest N points to u
    closest = N_closest(u,N)
    #computing the weight vector
    w=weight(u,closest)
    m=loc_mean(u,closest,w)
    sigma = loc_cov(m,closest,w)
    print("{0} ==> {1}".format(u,m))
    
    
    
    
    
def plot_heatmap():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img=ax.scatter(x_cut, y_cut, z_cut, c=cut_array,s=5,marker='s')
    plt.colorbar(img,fraction=0.025, pad=0.07)
    plt.title("Event {0}".format(j))
    plt.show()

#%%Main function
if  __name__ == "__main__":
#Opening of the pickled file
    dictionary=pickle.load(open("Pickles/3dreco0.pkl","rb"))
#We separate the UUID run key from the voxel dictionary
    key_name, dictionary = next(iter(dictionary.items()))
#take one of the events (this will have to remain )
    j=15
    event= dictionary[j]['amplitude']
#convert the event dic. to numpy array
    array=np.array(event)
#cutting the 8 voxels in x-y closest to the masks
#this allows to eliminate artefacts
    array=array[:,:,7:(array.shape[2]-7)]
#define the complete coordinate arrays
    x= np.arange(0.,array.shape[0],1)
    y=np.arange(0.,array.shape[1],1)
    z=np.arange(0.,array.shape[2],1)
#rescaling the amplitudes
    diff=np.max(array)-np.min(array)
    array = array/diff
#setting the amplitude threshold to be considered
    cut=0.9*np.max(array)
#performing the cut over the amplitude
    cut_array=array[array>=cut]
    x_cut=np.nonzero(array>=cut)[0]
    y_cut=np.nonzero(array>=cut)[1]
    z_cut=np.nonzero(array>=cut)[2]
 #plotting the cut data as a heatmap
    plot_heatmap()   
#The "algorithm"
#defining the matrix of cut coordinates
#reshaping the index vectors to get the right shape for X
    x_cut=x_cut.transpose()
    y_cut=y_cut.transpose()
    z_cut=z_cut.transpose()
#rows for the events, columns for the coordinates
    X=np.column_stack((x_cut,y_cut,z_cut))
#rescale the array of coordinates
    d = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d[i]=mt.sqrt(np.dot(X[i],X[i]))
    X = X/(d.max()-d.min())
#placeholder loc_mean computation
#the starting point is chosen at random among the points above cut
    start=X[random.randint(0,X.shape[0]-1)]
    lpc_cycle(start,20)
    
    
    
    #Ampiezze dsi probabilità come luminosità
    #taglio sulle ampiezze, abbastanza alto


