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
import pytest
#import  tkinter as tk
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import cdist form scipy to efficiently compute distances between all pairs of points
from scipy.spatial.distance import cdist
import scipy.optimize as optimize
#%%Functions for frequently used operations

#computing the weight vector
#We take the starting point and the closest points for which to compute the weights
def weight(u,hits,amplitudes,h):
    w = np.zeros(hits.shape[0])
    for i in range(hits.shape[0]):
    #computing the weights
        w[i] = (amplitudes[i]/((h**3)*((2*mt.pi)**(3/2))))*mt.exp((-1/(2*(h**2)))*np.dot((hits[i]-u),(hits[i]-u)))
    return w

#computing the local mean
#Currently the hits are chosen for the computation based on the X index!!!
def loc_mean(u,closest,amplitudes,h):
    #find the weights
    weights=weight(u,closest,amplitudes,h)
    #flag to call-off execution if no weights are found
    zero_flag= False
    #compute the average
    try:
        m=np.average(closest,axis=0,weights=weights)
    except ZeroDivisionError:
        print("Zero Division Error!!!")
        print("The weights:")
        print(weights)
        m=[0.,0.,0.]
        zero_flag=True
    return m, zero_flag

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



#alternative: get points within N voxels
def N_closest(u,hits,amps,n_width,norm):
    dist = np.zeros(hits.shape[0])
#computing the distance between u and each hits[i]
    for i in range(hits.shape[0]):
        dist[i]=np.sqrt(np.dot(hits[i]-u,hits[i]-u))
#selecting the points closer than N
    dist = np.where(np.logical_and(dist>0, dist<=n_width/norm))[0]
#slicing the hits matrix over the closest indices
    return np.take(hits,dist,axis=0),np.take(amps,dist,axis=0)


def remove_old(points,points_old,amplitudes):
    #print("points shape: ",points.shape,", old points shape: ",points_old.shape)
    mask=(points[:,None]==points_old).all(-1).any(-1)
    mask=np.invert(mask)
    return points[mask], amplitudes[mask];

#%% LPC cycle functions
#cycle over the hit clusters to find possible tracks
def track_cycle(hits, amps,norm,N_p,n_cyc,n_width,n_neg,x,y,z,folder):
    #keep the complete set of voxels to compute the lpc on
    og_hits=hits
    og_amps=amps
    #define the arrays that will contain the points and parameters
    #for the fits
    par_ar_xy=np.zeros((n_cyc,2))
    par_ar_xz=np.zeros((n_cyc,2))
    m_array=np.zeros((n_cyc,N_p,3))
    for i in range(n_cyc):
        #plotting the cut data as a heatmap
        plot_heatmap(hits,amps,i,norm,x,y,z,folder)
        # print("Hits shape: ",hits.shape)
        # print("Amps shape: ",amps.shape)
        try:
            #compute the c.o.m. of the remaining hits
            c_mass=np.average(hits,axis=0,weights=amps)
            c_mass=np.reshape(c_mass,(1,3))
            #find the index of the hit closest to the c.o.m.
            index=(cdist(hits,c_mass)).argmin()
            #use this hit as the starting point
            start=hits[index]
        except ZeroDivisionError:
            # print("Zero Division Error!!!")
            start=[0.,0.,0.]
        # print("-- Track {0} --".format(i+1))
        # print("Start = ",start*norm)
        #perform an lpc cycle and save the resulting points (and graphs)
        lpc_points,m_array[i]=lpc_cycle(start,og_hits,og_amps,n_width,i,N_p,norm,x,y,z)
        # print("lpc shape: ",lpc_points.shape)
        #remove from X the hits in the vicinity of the lpc curve
        amps=amps[np.all(cdist(hits,lpc_points)>=n_neg/norm,axis=1)]
        hits= hits[np.all(cdist(hits,lpc_points)>=n_neg/norm,axis=1)]
        #return to the original scale
        m_array[i]=m_array[i]*norm
        par_ar_xy[i],par_ar_xz[i]=fit_projections(m_array[i],x,y,z)
    #plot the combined fitted data using a cycle
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    plt.title("Mean points fit",size=20)
    for i in range(n_cyc):
        m_arrayc=m_array[i][~np.all(m_array[i]==0, axis=1)]
        img=ax.scatter(m_arrayc[:,0],m_arrayc[:,1],
                       m_arrayc[:,2],c=np.arange(0,m_arrayc.shape[0]),s=20,marker='s')
        plt.plot(m_arrayc[:,0],m_arrayc[:,0]*par_ar_xy[i][0]+par_ar_xy[i][1],
                 m_arrayc[:,0]*par_ar_xz[i][0]+par_ar_xz[i][1],'r',linewidth=3)
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.2)
    cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
    plt.show()
    find_vert(m_array,par_ar_xy,par_ar_xz,n_cyc,n_width)

#function that determines the vertex with a closeness criterion
def find_vert(m_array,par_ar_xy,par_ar_xz,n_cyc,n_width):
    for i  in range(n_cyc):
        #cycle in range(i,2) to count each pair only once
        for j in range(i+1,n_cyc):
            m_array1=m_array[i][~np.all(m_array[i]==0, axis=1)]
            m_array2=m_array[j][~np.all(m_array[j]==0, axis=1)]
            fit_points1=np.array([m_array1[:,0],m_array1[:,0]*par_ar_xy[i][0]+par_ar_xy[i][1],
                                  m_array1[:,0]*par_ar_xz[i][0]+par_ar_xz[i][1]])
            fit_points2=np.array([m_array2[:,0],m_array2[:,0]*par_ar_xy[j][0]+par_ar_xy[j][1],
                                  m_array2[:,0]*par_ar_xz[j][0]+par_ar_xz[j][1]])
            #transpose to get (n,3) arrays
            fit_points1=fit_points1.transpose()
            fit_points2=fit_points2.transpose()
            #sort by the "x" coordinate
            fit_points1=fit_points1[np.argsort(fit_points1[:, 0])]
            fit_points2=fit_points2[np.argsort(fit_points2[:, 0])]
            #get the array of distances between all the points
            t_dist=cdist(fit_points1,fit_points2)
            # print(t_dist.shape)
            ind=np.unravel_index(np.argmin(t_dist, axis=None), t_dist.shape)
            # if t_dist.min() <=n_width:
            #     print("Vertex between tracks {0} and {1} found".format(i,j))
            #     print("Distance: ",t_dist.min())
            #     print("Track 1 coordinate: ",fit_points1[ind[0]])
            #     print("Track 2 coordinate: ",fit_points2[ind[1]])
            # else:
            #     print("No vertex found")
                    

#Perform the LPC cycle
def lpc_cycle(m0,hits,amps,n_width,cycle,N_p,norm,x,y,z):
    pathl=0.0
    gamma_old=np.zeros(3,)
    pathl_old=0.
    f_b=+1
    t=0.05
    b=0.005
    c=1
    #bandwidth parameter
    h=0.05
    #array in which to store the lpc points
    lpc_points=np.zeros((N_p,3))
    m_vec=np.zeros_like(lpc_points)
    lpc_points[0]=m0
    count=0
    closest_old=np.zeros_like(hits)
    angles=np.zeros(lpc_points.shape[0])
#start the cycle
    for l in range(N_p):
        # print("+++++++++++++++++++++")
        # print("Cycle {0}".format(l))
        # print("LPC point = ",lpc_points[l])
        count+=1
        #find the N closest points to lpc_points[l]
        closest, amplitudes=N_closest(lpc_points[l],hits,amps,n_width,norm)
        closest, amplitudes=remove_old(closest,closest_old,amplitudes)
        #compute the local mean
        m_vec[l],zero_flag=loc_mean(lpc_points[l],closest,amplitudes,h)
        if zero_flag:
                # print("Zero division error: exiting cycle")
                break;
        #compute the path length
        if l>0:
            pathl+=np.linalg.norm(m_vec[l]-m_vec[l-1])
        sigma = loc_cov(lpc_points[l],closest,m_vec[l],amplitudes,h)
        try:
            val,gamma=np.linalg.eigh(sigma)
        except np.linalg.LinAlgError:
            # print("Eigenvalues did not converge: exiting")
            break
        gamma=gamma[:,val.size-1]
        gamma=gamma/np.sqrt(np.dot(gamma,gamma))
        #reverse the vector if the cos of the angle with the previous is negative
        if np.dot(gamma,gamma_old)<0:
            gamma =-1*gamma
        #apply penalization if l>=1
        if l>0:
            a=abs(np.dot(gamma,gamma_old))**2
            gamma=a*gamma+(1-a)*gamma_old
            angles[l]=1- abs(np.dot(gamma,gamma_old))
        #find the next local neighborhood point
        try:
            lpc_points[l+1]=m_vec[l]+f_b*t*gamma
        except IndexError:
            # print("Could not reach convergence")
            break;
        #save the "old" variables
        closest_old=closest
        #amplitudes_old=amplitudes
        if l==0:
            R=1
        else:
            R=(pathl-pathl_old)/(pathl+pathl_old)
            # print("R = ",R)
        #update the "old" variables
        gamma_old=gamma
        pathl_old=pathl
        if R<7*10**(-4):
            if f_b ==-1:
                # print("Curve has converged: exiting")
                break
            else:
                # print("----------")
                # print("Inverting")
                # print("----------")
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
                    # print("Reached max. number of points: exiting")
                    break
                else:
                    # print("+++++++++++")
                    # print("+Inverting+")
                    # print("+++++++++++")
                    f_b=-1
                    lpc_points[l+1]=m0
                    c=1
                    count=0
    #Draw the LPC points plot
    draw_plots(lpc_points,m_vec,angles,cycle,norm,x,y,z)
    return lpc_points, m_vec


def line(x,m,b):
    return x*m+b

def fit_projections(m_vec,x,y,z):
    #m_vec is already rescaled
    #cut the non-assigned points
    m_vec=m_vec[~np.all(m_vec==0, axis=1)]
    #get the projections with slicings
    x_y_pro=m_vec[:,[0,1]]
    x_z_pro=m_vec[:,[0,2]]
    #perform the linear fit to the xy projection
    par_xy,cov_xy=optimize.curve_fit(line,x_y_pro[:,0],x_y_pro[:,1])
    #perform the linear fit to the xy projection
    par_xz,cov_xz=optimize.curve_fit(line,x_z_pro[:,0],x_z_pro[:,1])
    #2D projection plots are in the RecoCode
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    img=ax.scatter(m_vec[:,0],m_vec[:,1],m_vec[:,2],c=np.arange(0,m_vec.shape[0]),s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
    plt.title("Mean points fit",size=20)
    plt.plot(m_vec[:,0],m_vec[:,0]*par_xy[0]+par_xy[1],
             m_vec[:,0]*par_xz[0]+par_xz[1],'r',linewidth=3)
    plt.show()
    print(par_xy)
    print(par_xz)
    return par_xy,par_xz


# def fit_3d(m_vec):
#     #take a guess for the fit parameters
#     guess=(1,1,1)
#     #cut the non-assigned points
#     m_vec=m_vec[~np.all(m_vec==0, axis=1)]
#     #return to the original scale
#     m_vec=m_vec*norm
#     #sort the array by x (z) index
#     m_vec=m_vec[np.argsort(m_vec[:, 0])]
#     params, pcov = optimize.curve_fit(line, m_vec[:,:2], m_vec[:,2], guess)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim((x[0],x[x.shape[0]-1]))
#     ax.set_ylim((y[0],y[y.shape[0]-1]))
#     ax.set_zlim((z[0],z[z.shape[0]-1]))
#     img=ax.scatter(m_vec[:,0],m_vec[:,1],m_vec[:,2],c=np.arange(0,m_vec.shape[0]),s=20,marker='s')
#     cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
#     cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
#     plt.plot(m_vec[:,0],m_vec[:,1],(m_vec[:,0]*params[0]+m_vec[:,1]*params[1]+params[2]),'g')
#     print(params)

   
def draw_plots(lpc_points,m_vec,angles,cycle,norm,x,y,z):
    #plot the data
    #if points are to be saved check if the folder already exists
    #lpc_cache=lpc_points
    #eliminate the unused points
    lpc_points=lpc_points[~np.all(lpc_points==0, axis=1)]
    lpc_points=lpc_points*norm
    fig1 = plt.figure(figsize=(7, 7))
    ax = fig1.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    img=ax.scatter(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],c=np.arange(0,lpc_points.shape[0]),s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
    plt.title("LPC points plot",size=20)
    plt.pause(0.05)
    #plt.savefig('{0}/LPC_points_{1}.png'.format(folder,cycle))
    #plt.show()
    #Draw the LPC means plot
    m_vec=m_vec[~np.all(m_vec==0, axis=1)]
    m_vec=m_vec*norm
    fig2 = plt.figure(figsize=(7,7))
    ax = fig2.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    img=ax.scatter(m_vec[:,0],m_vec[:,1],m_vec[:,2],c=np.arange(0,m_vec.shape[0]),s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
    plt.title("Mean points plot",size=20)
    plt.pause(0.05)
    #plt.show()
    # #Draw the plot of eigenvector angles
    # fig3 = plt.figure(figsize=(7, 7))
    # lpc_range= np.arange(angles.shape[0])
    # plt.plot(lpc_range, angles,marker='o')
    # plt.title("Feature points plot")
    # #plt.savefig('{0}/Feature_points.png'.format(folder,cycle))
    # #plt.show()
    # #Draw the heatmap of eigenvector angles
    # angles=angles[~np.all(lpc_cache==0,axis=1)]
    # fig4 = plt.figure(figsize=(7, 7))
    # ax = fig4.add_subplot(111, projection='3d')
    # ax.set_xlim((x[0],x[x.shape[0]-1]))
    # ax.set_ylim((y[0],y[y.shape[0]-1]))
    # ax.set_zlim((z[0],z[z.shape[0]-1]))
    # ax.set_xlabel('x',fontsize=14,weight='bold')
    # ax.set_ylabel('y',fontsize=14,weight='bold')
    # ax.set_zlabel('z',fontsize=14,weight='bold')
    # img=ax.scatter(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],c=angles,s=20,marker='s')
    # plt.colorbar(img,fraction=0.025, pad=0.07,label="Angle")
    # plt.title("Eigenvector angles plot")
    #plt.savefig('{0}/Angles_plot.png'.format(folder,cycle))
    plt.show()



# def save_results():
#     #save the event parameters
#     try:
#         os.mkdir(folder)
#     except FileExistsError:
#         pass
#     with open('{0}/parameters.txt'.format(folder), 'w') as s:
#         print("Track type: {0}".format(ttype),file=s)
#         print("b_x: {0}".format(b_x),file=s)
#         print("b_y: {0}".format(b_y),file=s)
#         print("b_z: {0}".format(b_z),file=s)
#         print("Track seed: {0}".format(track_seed),file=s)
#         print("Cut fraction: {0}".format(cut_frac),file=s)

def show_plot(X,cut_array,norm,x,y,z):
    hits=X
    amps=cut_array
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=10,weight='bold')
    ax.set_ylabel('y',fontsize=10,weight='bold')
    ax.set_zlabel('z',fontsize=10,weight='bold')
    hits=hits*norm
    img=ax.scatter(hits[:,0], hits[:,1], hits[:,2], c=amps,s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Amplitude',size=10,weight='bold')
    plt.title("Event heatmap",size=20)
    plt.pause(0.05)
    plt.show(block=False)

def plot_heatmap(hits,amps,cycle,norm,x,y,z,folder):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    hits=hits*norm
    img=ax.scatter(hits[:,0], hits[:,1], hits[:,2], c=amps,s=20,marker='s')
    plt.colorbar(img,fraction=0.025, pad=0.07,label="Amplitude")
    plt.title("Event heatmap")
    # if question:
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    plt.savefig('{0}/Ev_heatmap{1}.png'.format(folder,cycle))
    plt.show()

#%% Test track generation
#test the algorithm with a made-up gaussian-distributed track
#127 "sort-of" converged with N=8
# def gaussian_test():
#     seed=1392
#     np.random.seed(int(seed))
#     array[:,:,:]=0.
#     for a in range(array.shape[0]):
#         for j in range(array.shape[0]):
#             y_cor=int(np.abs(np.random.normal(14.,0.5)))
#             for k in range(array.shape[0]):
#                 z_cor=int(np.abs(np.random.normal(77.,1)))
#                 amp=np.abs(np.random.normal(1.,0.05))
#                 array[a][y_cor][z_cor]=amp
#     return "Gaussian track", seed
def vert_gauss_test():
    array=np.zeros((50,50,130))
    seed=np.zeros(2)
    seed[0]=1392
    np.random.seed(int(seed[0]))
    for a in range(array.shape[0]):
        for j in range(100):
            y_cor=int(np.abs(np.random.normal(a*0.5,0.5)))
            for k in range(100):
                z_cor=int(np.abs(np.random.normal(77.,1.)))
                amp=np.abs(np.random.normal(1.,0.05))
                array[a][y_cor][z_cor]=amp
    seed[1]=4592
    np.random.seed(int(seed[1]))
    for a in range(array.shape[0]):
        for j in range(100):
            if 2*a <array.shape[0]:
                y_cor=int(np.abs(np.random.normal(2*a,0.5)))
                for k in range(100):
                    z_cor=int(np.abs(np.random.normal(77.,1.)))
                    amp=np.abs(np.random.normal(1.,0.05))
                    array[a][y_cor][z_cor]=amp
    return array,"Gaussian tracks vertex",seed
#Test the algorithm with a made up straight line track
#28-127-460-461-480-481 converge
# def linear_test():
#     seed=430
#     np.random.seed(int(seed))
#     array[:,:,:]=0.
#     for a in range(array.shape[0]):
#         amp=np.abs(np.random.normal(1.,0.05))
#         array[a][14][77]=amp
#     return "Linear track",seed
# #Vertex test
# def vertex_test():
#     seed=550
#     np.random.seed(int(seed))
#     array[:,:,:]=0.
#     for a in range(array.shape[0]):
#         amp=np.abs(np.random.normal(1.,0.05))
#         if 2*a <array.shape[0]:
#             array[a][2*a][77]=amp
#     for a in range(array.shape[0]):
#         amp=np.abs(np.random.normal(1.,0.05))
#         array[a][a//2][77]=amp
#     return "Linear track vertex",seed
# #Test the algorithm with a circular sector track
# def circular_test():
#     seed=127
#     np.random.seed(int(seed))
#     array[:,:,:]=0.
#     for a in range(array.shape[0]):
#         y_cor= int(np.sqrt((array.shape[0]-1)**2-a**2))
#         amp=np.abs(np.random.normal(1.,0.05))
#         array[a][y_cor][77]=amp
#     return "Circular arch track", seed
#%%Main function
def main(folder="textReco_def",file_sel = "./eventi_valentina/diagmu_1ev_reco.pkl",
         b_x=1,b_y=1,b_z=10,cut_frac=0.6,n_cyc=2,n_width=3,n_neg=5):
#define the folder in which to save the results
    # folder = input("Enter the save folder name: ") or "textReco_def"
    folder = "./Plots/tests/{0}".format(folder)
#Opening of the pickled file
    print("++++++++++++")
    print("Enter dataset parameters:")
    print("++++++++++++")
    # root = tk.Tk()
    # root.withdraw()
    # print("Select the .pkl")
    # file_sel = filedialog.askopenfilename()
    # file_sel = "./eventi_valentina/diagmu_1ev_reco.pkl"
    dictionary=pickle.load(open(file_sel,"rb"))
#We separate the UUID run key from the voxel dictionary
    key_name, dictionary = next(iter(dictionary.items()))
#take one of the events (this will have to remain )
    j=0
    event= dictionary[j]['amplitude']
#convert the event dic. to numpy array
    og_array=np.array(event)
    ttype="Simulated event"
    track_seed="None"
    #Test the algorithm with made-up tracks
    og_array,ttype,track_seed = vert_gauss_test()
    array=np.empty_like(og_array)
#default cuts the voxels closest to the masks to avoid artifacts
    # b_x=1
    # b_y=1
    # b_z=10
    # #default amplitude threshold to be considered
    # cut_frac=0.6
# while True:
    #initialize the array
    array=np.empty_like(og_array)
    # b_x=int(input("Enter x cut: ") or b_x)
    # b_y=int(input("Enter y cut: ") or b_y)
    # b_z=int(input("Enter z cut: ") or b_z)
    # cut_frac=float(input("Enter amplitude cut: ") or cut_frac)
    array=og_array[b_x:(og_array.shape[0]-b_x),b_y:(og_array.shape[1]-b_y),b_z:(og_array.shape[2]-b_z)]
#define the complete coordinate arrays
    x= np.arange(0.,array.shape[0],1)
    y=np.arange(0.,array.shape[1],1)
    z=np.arange(0.,array.shape[2],1)
#rescaling the amplitudes
    array=array-np.min(array)
    array = array/np.max(array)
#performing the cut over the amplitude
    cut_array=array[array>=cut_frac]
    x_cut=np.nonzero(array>=cut_frac)[0]+b_x
    y_cut=np.nonzero(array>=cut_frac)[1]+b_y
    z_cut=np.nonzero(array>=cut_frac)[2]+b_z
#defining the matrix of cut coordinates
#reshaping the index vectors to get the right shape for X
    x_cut=x_cut.transpose()+0.5
    y_cut=y_cut.transpose()+0.5
    z_cut=z_cut.transpose()+0.5
    cut_array=cut_array.transpose()
#rows for the events, columns for the coordinates
    X=np.column_stack((x_cut,y_cut,z_cut))
#rescale the array of coordinates
    d = np.zeros([X.shape[0],X.shape[0]])
    for i in range(X.shape[0]):
        d[i]=np.linalg.norm(X[i])
    norm=(d.max()-d.min())
    X = X/norm
    show_plot(X,cut_array,norm,x,y,z)
    # if input("Start the LPC cycle? [y/n] ") == "y":
    #     break            
    # question = True  
#starting the track finding cycle
    #set the number of cycles as a main variable
    # N_p=200
    # n_cyc=1
    # n_width=3
    # n_neg=5
    # n_cyc=int(input("Enter number of cycles: ") or n_cyc)
    # n_width=int(input("Enter neighborhood width: ") or n_width)
    # n_neg=int(input("Enter voxels to be neglected: ") or n_neg)
    N_p=200
    track_cycle(X,cut_array,norm,N_p,n_cyc,n_width,n_neg,x,y,z,folder)
    #proj1,proj2=fit_projections(test_array)



#%%Testing main() with pytest
#Using pytest.raises the test fails if no Exception is raised
def test_void_folder():
    with pytest.raises(Exception):
        main("")
def test_invalid_folder():
    with pytest.raises(Exception):
        main("/-$prova")
def test_valid_folder():
    with pytest.raises(Exception):
        main("")
#testing the file name
def test_valid_file():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl")
def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        main("textReco_def","./eventi_valentina/diagmu_reco.pkl")
def test_invalid_extension():
    with pytest.raises(FileNotFoundError):
        main("textReco_def","./eventi_valentina/diagmu_reco.pdf")
#testing the cuts
#x negative cut
def test_neg_cut():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",-10)
#cut above limit
def test_invalid_cut():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",500)
#cut equal to dimensions
def test_max_cut():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",50)
#testing the cut fraction
def test_un_frac():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,1.0)
#cut fraction above 1
def test_inv_frac():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,2.)
#negative cut fraction
def test_neg_frac():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6)
#testing the number of cycles
#0 cycles
def test_zero_cyc():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,0)
#negative number of cycles
def test_neg_cyc():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,-2)
#much more cycles than tracks
def test_many_cyc():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,5)
#testing the neighborhood parameter
#0 cycles
def test_0_width():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,0.)
#test small width
def test_small_width():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,1)
#test fractional width
def test_frac_width():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,3.5)
#test very large widths
def test_large_width():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,10)
#test negative widths
def test_neg_width():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,-3)
#testing the neglection width parameter
#0 cycles
def test_0_negl():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,3,0)
#test small width
def test_small_negl():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,3,1)
#test fractional width
def test_frac_negl():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,3,3.5)
#test very large widths
def test_large_negl():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,3,30)
def test_neg_negl():
    with pytest.raises(Exception):
        main("textReco_def","./eventi_valentina/diagmu_1ev_reco.pkl",1,1,10,0.6,2,3,-3)