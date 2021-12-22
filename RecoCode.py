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
import os
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#%%Functions for frequently used operations

#computing the weight vector
#We take the starting point and the N closest for which to compute the weights
def weight(u,hits,amplitudes,h):
    w = np.zeros(hits.shape[0])
    for i in range(hits.shape[0]):
    #placeholder weight formula
        w[i] = (amplitudes[i]/((h**3)*((2*mt.pi)**(3/2))))*mt.exp((-1/(2*(h**2)))*np.dot((hits[i]-u),(hits[i]-u)))
        #w[i]=amplitudes[i]
        #print("weight {0} = {1}".format(i,w[i]))
    return w

#computing the local mean
#Currently the hits are chosen for the computation based on the X index!!!
def loc_mean(u,closest,amplitudes,h):
    #find the weights
    weights=weight(u,closest,amplitudes,h)
    #compute the average
    try:
        m=np.average(closest,axis=0,weights=weights)
    except ZeroDivisionError:
        print("Zero Division Error!!!")
        print("The weights:")
        print(weights)
        m=[0.,0.,0.]
    return m

#Computing the local covariant matrix
def loc_cov(u,closest,m,amplitudes,h):
    #find the weights
    weights=weight(u,closest,amplitudes,h)
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

# #Computing the mean shift (see if it can be incorporated in another function)
# def mean_shift(u,closest):
#     lm=loc_mean(u,closest)-u
#     return lm

#Computing the set of N closest points
# def N_closest(u,N):
#     dist = np.zeros(X.shape[0])
# #computing the distance between u and each X[i]
#     for i in range(X.shape[0]):
#         dist[i]=np.linalg.norm(X[i]-u)
# #sorting the distance array 
#     dist = np.argsort(dist,axis=0)
# #selecting the closest N (except for u itself)
#     dist = dist[1:N+1]
# #slicing the X matrix over the closest indices
#     return np.take(X,dist,axis=0), np.take(cut_array,dist,axis=0);


#alternative: get points within N voxels
def N_closest(u,N):
    dist = np.zeros(X.shape[0])
#computing the distance between u and each X[i]
    for i in range(X.shape[0]):
        dist[i]=np.sqrt(np.dot(X[i]-u,X[i]-u))
#selecting the points closer than N
    dist = np.where(np.logical_and(dist>0, dist<=N/norm))[0]
#slicing the X matrix over the closest indices
    return np.take(X,dist,axis=0),np.take(cut_array,dist,axis=0)


def remove_old(points,points_old,amplitudes):
    #print("points shape: ",points.shape,", old points shape: ",points_old.shape)
    mask=(points[:,None]==points_old).all(-1).any(-1)
    mask=np.invert(mask)
    return points[mask], amplitudes[mask];


#Perform the LPC cycle
def lpc_cycle(m0,N):
    pathl=0.0
    gamma_old=np.zeros(3,)
    pathl_old=0.
    f_b=+1
    t=0.05
    b=0.005
    c=1
    #bandwidth parameter
    h=0.05
    N_p=200
    #array in which to store the lpc points
    lpc_points=np.zeros((N_p,3))
    m_vec=np.zeros_like(lpc_points)
    lpc_points[0]=m0
    count=0
    closest_old=np.zeros_like(X)
    angles=np.zeros(lpc_points.shape[0])
#start the cycle
    for l in range(N_p):
        print("+++++++++++++++++++++")
        print("Cycle {0}".format(l))
        print("LPC point = ",lpc_points[l])
        count+=1
        #find the N closest points to lpc_points[l]
        closest, amplitudes=N_closest(lpc_points[l],N)
        closest, amplitudes=remove_old(closest,closest_old,amplitudes)
        #compute the local mean
        m_vec[l]=loc_mean(lpc_points[l],closest,amplitudes,h)
        #print("local mean = ",m_vec[l])
        #compute the path length
        if l>0:
            pathl+=np.linalg.norm(m_vec[l]-m_vec[l-1])
            #print("old mean = ",m_vec[l-1])
        #print("path length = ",pathl)
        sigma = loc_cov(lpc_points[l],closest,m_vec[l],amplitudes,h)
        try:
            val,gamma=np.linalg.eigh(sigma)
        except np.linalg.LinAlgError:
            print("Eigenvalues did not converge: exiting")
            break
        gamma=gamma[:,val.size-1]
        gamma=gamma/np.sqrt(np.dot(gamma,gamma))
        #reverse the vector if the cos of the angle with the previous is negative
        if np.dot(gamma,gamma_old)<0:
            gamma =-1*gamma
        #apply penalization if l>=1
        if l>0:
            a=(np.dot(gamma,gamma_old))**2
            gamma=a*gamma+(1-a)*gamma_old
            angles[l]=1- abs(np.dot(gamma,gamma_old))
        #print("gamma = ",gamma)
        #find the next local neighborhood point
        try:
            lpc_points[l+1]=m_vec[l]+f_b*t*gamma
        except IndexError:
            print("Could not reach convergence")
            break;
        #save the "old" variables
        closest_old=closest
        #amplitudes_old=amplitudes
        if l==0:
            R=1
        else:
            R=(pathl-pathl_old)/(pathl+pathl_old)
            print("R = ",R)
        #update the "old" variables
        gamma_old=gamma
        pathl_old=pathl
        if R<7*10**(-4):
            if f_b ==-1:
                print("Curve has converged: exiting")
                break
            else:
                print("----------")
                print("Inverting")
                print("----------")
                f_b=-1
                lpc_points[l+1]=m0
                c=1
                count=0
        elif l>0:
            if R<b:
                c=(1-b)*c
                h=h*c
            else:
                c=min(1.01*c,1.0)
                h=0.5
                #print("Branch 3")
            if count>=(0.5*N_p):
                if f_b==-1:
                    print("Reached max. number of points: exiting")
                    break
                else:
                    print("+++++++++++")
                    print("+Inverting+")
                    print("+++++++++++")
                    f_b=-1
                    lpc_points[l+1]=m0
                    c=1
                    count=0
    #Draw the LPC points plot
    save_results(N_p,h,t,lpc_points,m_vec,angles)
    
def save_results(N_p,h,t,lpc_points,m_vec,angles):
    #save the event parameters
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    with open('{0}/parameters.txt'.format(folder), 'w') as s:
        print("Track type: {0}".format(ttype),file=s)
        print("Track seed: {0}".format(track_seed),file=s)
        print("Cut fraction: {0}".format(cut_frac),file=s)
        print("Start seed: {0}".format(start_seed),file=s)
        print("Start: {0}".format(start*norm),file=s)
        print("h: {0}".format(h),file=s)
        print("t: {0}".format(t),file=s)
        print("Number of points: {0}".format(N_p),file=s)
        #plot the data
        #save the lpc_points array firts
        lpc_cache=lpc_points
        #eliminate the unused points
        lpc_points=lpc_points[~np.all(lpc_points==0, axis=1)]
        lpc_points=lpc_points*norm
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim((x[0],x[x.shape[0]-1]))
        ax.set_ylim((y[0],y[y.shape[0]-1]))
        ax.set_zlim((z[0],z[z.shape[0]-1]))
        img=ax.scatter(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],c=np.arange(0,lpc_points.shape[0]),s=20,marker='s')
        plt.colorbar(img,fraction=0.025, pad=0.07)
        plt.title("LPC points plot")
        plt.savefig('{0}/LPC_points.png'.format(folder))
        plt.show()
        #Draw the LPC means plot
        m_vec=m_vec[~np.all(m_vec==0, axis=1)]
        m_vec=m_vec*norm
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim((x[0],x[x.shape[0]-1]))
        ax.set_ylim((y[0],y[y.shape[0]-1]))
        ax.set_zlim((z[0],z[z.shape[0]-1]))
        img=ax.scatter(m_vec[:,0],m_vec[:,1],m_vec[:,2],c=np.arange(0,m_vec.shape[0]),s=20,marker='s')
        plt.colorbar(img,fraction=0.025, pad=0.07)
        plt.title("Mean points plot")
        plt.savefig('{0}/LPC_means.png'.format(folder))
        plt.show()
        #Draw the plot of eigenvector angles
        fig = plt.figure(figsize=(5, 5))
        lpc_range= np.arange(angles.shape[0])
        plt.plot(lpc_range, angles,marker='o')
        plt.title("Feature points plot")
        plt.savefig('{0}/Feature_points.png'.format(folder))
        plt.show()
        #Draw the heatmap of eigenvector angles
        angles=angles[~np.all(lpc_cache==0,axis=1)]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim((x[0],x[x.shape[0]-1]))
        ax.set_ylim((y[0],y[y.shape[0]-1]))
        ax.set_zlim((z[0],z[z.shape[0]-1]))
        img=ax.scatter(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],c=angles,s=20,marker='s')
        plt.colorbar(img,fraction=0.025, pad=0.07)
        plt.title("Eigenvector angles plot")
        plt.savefig('{0}/Angles_plot.png'.format(folder))
        plt.show()
        # #draw the vector plot
        # lpc_arrows=np.zeros_like(lpc_points)
        # for i in range(1,lpc_points.shape[0]):
        #     lpc_arrows[i]=lpc_points[i]-lpc_points[l-1]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim((x[0],x[x.shape[0]-1]))
        # ax.set_ylim((y[0],y[y.shape[0]-1]))
        # ax.set_zlim((z[0],z[z.shape[0]-1]))
        # ax.quiver(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],lpc_arrows[:,0],lpc_arrows[:,1],lpc_arrows[:,2])
        # plt.title("Curve {0}".format(j))
        # plt.show()
    
    
def plot_heatmap():
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    img=ax.scatter(x_cut, y_cut, z_cut, c=cut_array,s=20,marker='s')
    plt.colorbar(img,fraction=0.025, pad=0.07)
    plt.title("Event heatmap")
    plt.savefig('{0}/Ev_heatmap.png'.format(folder))
    plt.show()

#%% Test track generation
#test the algorithm with a made-up gaussian-distributed track
#127 "sort-of" converged with N=8
def gaussian_test():
    seed=1392
    np.random.seed(int(seed))
    array[:,:,:]=0.
    for a in range(array.shape[0]):
        for j in range(array.shape[0]):
            y_cor=int(np.abs(np.random.normal(14.,0.5)))
            for k in range(array.shape[0]):
                z_cor=int(np.abs(np.random.normal(77.,1)))
                amp=np.abs(np.random.normal(1.,0.05))
                array[a][y_cor][z_cor]=amp
    return "Gaussian track", seed
def vert_gauss_test():
    array[:,:,:]=0.
    seed=np.zeros(2)
    seed[0]=1392
    np.random.seed(int(seed[0]))
    for a in range(array.shape[0]):
        for j in range(array.shape[0]):
            y_cor=int(np.abs(np.random.normal(a*0.5,0.5)))
            for k in range(array.shape[0]):
                z_cor=int(np.abs(np.random.normal(77.,0.5)))
                amp=np.abs(np.random.normal(1.,0.05))
                array[a][y_cor][z_cor]=amp
    seed[1]=4592
    np.random.seed(int(seed[1]))
    for a in range(array.shape[0]):
        for j in range(array.shape[0]):
            if 2*a <array.shape[0]:
                y_cor=int(np.abs(np.random.normal(a*2,0.5)))
                for k in range(array.shape[0]):
                    z_cor=int(np.abs(np.random.normal(77.,0.5)))
                    amp=np.abs(np.random.normal(1.,0.05))
                    array[a][y_cor][z_cor]=amp
    return "Gaussian tracks vertex",seed
#Test the algorithm with a made up straight line track
#28-127-460-461-480-481 converge
def linear_test():
    seed=430
    np.random.seed(int(seed))
    array[:,:,:]=0.
    for a in range(array.shape[0]):
        amp=np.abs(np.random.normal(1.,0.05))
        array[a][14][77]=amp
    return "Linear track",seed
#Vertex test
def vertex_test():
    seed=550
    np.random.seed(int(seed))
    array[:,:,:]=0.
    for a in range(array.shape[0]):
        amp=np.abs(np.random.normal(1.,0.05))
        if 2*a <array.shape[0]:
            array[a][2*a][77]=amp
    for a in range(array.shape[0]):
        amp=np.abs(np.random.normal(1.,0.05))
        array[a][a//2][77]=amp
    return "Linear track vertex",seed
#Test the algorithm with a circular sector track
def circular_test():
    seed=127
    np.random.seed(int(seed))
    array[:,:,:]=0.
    for a in range(array.shape[0]):
        y_cor= int(np.sqrt((array.shape[0]-1)**2-a**2))
        amp=np.abs(np.random.normal(1.,0.05))
        array[a][y_cor][77]=amp
    return "Crircular arch track", seed
#%%Main function
if  __name__ == "__main__":
#define the folder in which to save the results
    folder = "./Plots/Plots_1"
#Opening of the pickled file
    dictionary=pickle.load(open("Pickles/3dreco0.pkl","rb"))
#We separate the UUID run key from the voxel dictionary
    key_name, dictionary = next(iter(dictionary.items()))
#take one of the events (this will have to remain )
#event 13 of config0 seems good
    j=2
    event= dictionary[j]['amplitude']
#convert the event dic. to numpy array
    array=np.array(event)
#cutting the 8 voxels in x-y closest to the masks
#this allows to eliminate artefacts
    array=array[:,:,7:(array.shape[2]-7)]
#Test the algorithm with made-up tracks
    ttype,track_seed = vert_gauss_test()
#define the complete coordinate arrays
    x= np.arange(0.,array.shape[0],1)
    y=np.arange(0.,array.shape[1],1)
    z=np.arange(0.,array.shape[2],1)
#rescaling the amplitudes
    diff=np.max(array)-np.min(array)
    array = array/diff
#setting the amplitude threshold to be considered
    cut_frac=0.6
    cut=cut_frac*np.max(array)
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
    cut_array=cut_array.transpose()
#rows for the events, columns for the coordinates
    X=np.column_stack((x_cut,y_cut,z_cut))
#rescale the array of coordinates
    d = np.zeros([X.shape[0],X.shape[0]])
    for i in range(X.shape[0]):
        d[i]=np.linalg.norm(X[i])
    norm=(d.max()-d.min())
    X = X/norm
#placeholder loc_mean computation
#the starting point is chosen at random among the points above cut
    start_seed=25
    np.random.seed(int(start_seed))
    #start=X[np.random.randint(0,X.shape[0]-1)]
    start=np.array([7,20,78])/norm
    print("Start = ",start)
    lpc_cycle(start,3)
    print("Start = ",start*norm)
    #array delle medie come risultato
    #Ampiezze dsi probabilità come luminosità
    #taglio sulle ampiezze, abbastanza alto
#cut value and number of points are relevant parameters:
#for more complex curves 