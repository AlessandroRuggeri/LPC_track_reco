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
import matplotlib.pyplot as plt
#%%Functions for frequently used operations
#computing the local mean
#Currently the hits are chosen for the computation based on the X index!!!
def loc_mean(u,closest):
    #find the weights
    weights=weight(u,closest)
    #compute the average
    m=np.average(closest,axis=0,weights=weights)
    return m

#Computing the local covariant matrix
def loc_cov(u,closest,m):
    #find the weights
    weights=weight(u,closest)
    #compute the point-m differences
    closest=closest-m
    #define the covariance matrix array
    sigma = np.zeros_like(np.outer(closest[0],closest[0]),dtype=float)
    #iterate to get the sum at the numerator
    for i in range(closest.shape[0]):
        sigma+=np.outer(closest[i],closest[i])*weights[i]
    #divide by the sum of the weights
    sigma = sigma/np.sum(weights)
    return sigma

#Computing the mean shift (see if it can be incorporated in another function)
def mean_shift(u,closest):
    lm=loc_mean(u,closest)-u
    return lm
#computing the weight vector
#We take the starting point and the N closest for which to compute the weights
def weight(u,hits):
    w = np.zeros(hits.shape[0])
    for i in range(hits.shape[0]):
    #placeholder weight formula
        w[i] = (1/((2*mt.pi)**(3/2)*h**3))*mt.exp((-1/(2*h**2))*np.dot((hits[i]-u),(hits[i]-u)))
    return w

#Computing the set of N closest points
def N_closest(u,N):
    dist = np.zeros(X.shape[0])
#computing the distance between u and each X[i]
    for i in range(X.shape[0]):
        dist[i]=np.sqrt(np.dot(X[i]-u,X[i]-u))
#sorting the distance array 
    dist = np.argsort(dist,axis=0)
#selecting the closest N (except for u itself)
    dist = dist[1:N+1]
#slicing the X matrix over the closest indices
    return np.take(X,dist,axis=0)
#Plotting of the hits

def remove_old(points,points_old):
    #print("points shape: ",points.shape,", old points shape: ",points_old.shape)
    mask=np.isin(points,points_old,invert=True)
    points[mask]
    return points

#bandwidth parameter
h=0.05
#Perform the LPC cycle
def lpc_cycle(m0,N):
    pathl=0.0
    m_old=m0
    gamma_old=np.zeros(3,)
    pathl_old=0.
    f_b=+1
    t=0.5
    b=0.005
    c=1
    N_p=200
    #array in which to store the lpc points
    lpc_points=np.zeros((N_p,3))
    lpc_points[0]=m0
    count=0
    closest_old=np.zeros_like(X)
#start the cycle
    for l in range(N_p):
        print("Cycle {0},lpc point = {1}".format(l,lpc_points[l]))
        count+=1
        #find the N closest points to lpc_points[l]
        closest=N_closest(lpc_points[l],N)
        closest=remove_old(closest,closest_old)
        closest_old=closest
        #compute the local mean
        m=loc_mean(lpc_points[l],closest)
        #compute the path length
        if l>0:
            m_old=loc_mean(lpc_points[l-1],closest_old)
            pathl+=np.sqrt(np.dot(m-m_old,m-m_old))
        sigma = loc_cov(lpc_points[l],closest,m,)
        val,gamma=np.linalg.eigh(sigma)
        gamma=gamma[:,val.size-1]
        gamma=gamma/np.sqrt(np.dot(gamma,gamma))
        #reverse the vector if the cos of the angle with the previous is negative
        if np.dot(gamma,gamma_old)<0:
            gamma =-1*gamma
        #apply penalization if l>=1
        if l>0:
            a=(np.dot(gamma,gamma_old))**2
            gamma=a*gamma+(1-a)*gamma_old
        #find the next local neighborhood point
        lpc_points[l+1]=m+f_b*t*gamma
        #save the old closest points
        closest_old=closest
        if l==0:
            R=1
        else:
            R=(pathl-pathl_old)/(pathl+pathl_old)
            print("R = ",R)
        #update the "old" variables
        gamma_old=gamma
        pathl_old=pathl
        if R<10**(-6):
            if f_b ==-1:
                print("Curve has converged: exiting")
                break
            else:
                print("Inverting")
                f_b=-1
                lpc_points[l+1]=m0
                c=1
                count=0
        else:
            if R<b:
                c=(1-b)*c
            else:
                c=min(1.01*c,1.0)
                #print("Branch 3")
            if count>=(0.5*N_p):
                if f_b==-1:
                    print("Reached max. number of points: exiting")
                    break
                else:
                    print("Inverting")
                    f_b=-1
                    lpc_points[l+1]=m0
                    c=1
                    count=0
    lpc_points=lpc_points[~np.all(lpc_points==0, axis=1)]
    lpc_points=lpc_points*norm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    img=ax.scatter(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],c=np.arange(0,lpc_points.shape[0]),s=20,marker='s')
    plt.colorbar(img,fraction=0.025, pad=0.07)
    plt.show()
    #draw the vector plot
    lpc_arrows=np.zeros_like(lpc_points)
    for i in range(1,lpc_points.shape[0]):
        lpc_arrows[i]=lpc_points[i]-lpc_points[l-1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.quiver(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],lpc_arrows[:,0],lpc_arrows[:,1],lpc_arrows[:,2])
    plt.title("Curve {0}".format(j))
    plt.show()
            
    #print("{0} ==> {1}".format(u,m))
    
    
    
    
    
def plot_heatmap():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    img=ax.scatter(x_cut, y_cut, z_cut, c=cut_array,s=20,marker='s')
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
#event 13 of config0 seems good
    j=273
    event= dictionary[j]['amplitude']
#convert the event dic. to numpy array
    array=np.array(event)
#cutting the 8 voxels in x-y closest to the masks
#this allows to eliminate artefacts
    array=array[:,:,7:(array.shape[2]-7)]
#Test the algorithm with a made up track
    np.random.seed(24)
    array[:,:,:]=0.
    for a in range(array.shape[0]):
            y_cor=int(np.abs(np.random.normal(13.,1)))
            z_cor=int(np.abs(np.random.normal(60.,2)))
            amp=np.abs(np.random.normal(1.,0.05))
            array[a][y_cor][z_cor]=amp
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
    d = np.zeros([X.shape[0],X.shape[0]])
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            d[i][j]=np.sqrt(np.dot(X[i],X[j]))
    norm=(d.max()-d.min())
    X = X/norm
#placeholder loc_mean computation
#the starting point is chosen at random among the points above cut
    start=X[np.random.randint(0,X.shape[0]-1)]
    #start=[0,10,70]/norm
    lpc_cycle(start,4)
    
    
    
    #Ampiezze dsi probabilità come luminosità
    #taglio sulle ampiezze, abbastanza alto


