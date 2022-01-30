#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:06:25 2022

@author: alessandro
"""

import PySimpleGUI as sg      
import numpy as np
import pickle
import math as mt
import os
#import  tkinter as tk
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import cdist form scipy to efficiently compute distances between all pairs of points
from scipy.spatial.distance import cdist
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
def N_closest(u,hits,amps,N):
    dist = np.zeros(hits.shape[0])
#computing the distance between u and each hits[i]
    for i in range(hits.shape[0]):
        dist[i]=np.sqrt(np.dot(hits[i]-u,hits[i]-u))
#selecting the points closer than N
    dist = np.where(np.logical_and(dist>0, dist<=N/norm))[0]
#slicing the hits matrix over the closest indices
    return np.take(hits,dist,axis=0),np.take(amps,dist,axis=0)


def remove_old(points,points_old,amplitudes):
    #print("points shape: ",points.shape,", old points shape: ",points_old.shape)
    mask=(points[:,None]==points_old).all(-1).any(-1)
    mask=np.invert(mask)
    return points[mask], amplitudes[mask];

#%% LPC cycle functions
#cycle over the hit clusters to find possible tracks
def track_cycle(hits, amps):
    #keep the complete set of voxels to compute the lpc on
    og_hits=hits
    og_amps=amps
    for i in range(1):
        #plotting the cut data as a heatmap
        plot_heatmap(hits,amps,i)   
        print("Hits shape: ",hits.shape)
        print("Amps shape: ",amps.shape)
        try:
            #compute the c.o.m. of the remaining hits
            cm=np.average(hits,axis=0,weights=amps)
            cm=np.reshape(cm,(1,3))
            #find the index of the hit closest to the c.o.m.
            index=(cdist(hits,cm)).argmin()
            #use this hit as the starting point
            start=hits[index]
        except ZeroDivisionError:
            print("Zero Division Error!!!")
            start=[0.,0.,0.]
        print("-- Track {0} --".format(i+1))
        print("Start = ",start*norm)
        #perform an lpc cycle and save the resulting points (and graphs)
        lpc_points=lpc_cycle(start,og_hits,og_amps,3,i)
        print("lpc shape: ",lpc_points.shape)
        #remove from X the hits in the vicinity of the lpc curve 
        amps=amps[np.all(cdist(hits,lpc_points)>=7/norm,axis=1)]
        hits= hits[np.all(cdist(hits,lpc_points)>=7/norm,axis=1)]



#Perform the LPC cycle
def lpc_cycle(m0,hits,amps,N,cycle):
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
    closest_old=np.zeros_like(hits)
    angles=np.zeros(lpc_points.shape[0])
#start the cycle
    for l in range(N_p):
        print("+++++++++++++++++++++")
        print("Cycle {0}".format(l))
        print("LPC point = ",lpc_points[l])
        count+=1
        #find the N closest points to lpc_points[l]
        closest, amplitudes=N_closest(lpc_points[l],hits,amps,N)
        closest, amplitudes=remove_old(closest,closest_old,amplitudes)
        #compute the local mean
        m_vec[l],zero_flag=loc_mean(lpc_points[l],closest,amplitudes,h)
        if zero_flag:
                print("Zero division error: exiting cycle")
                break;
        #compute the path length
        if l>0:
            pathl+=np.linalg.norm(m_vec[l]-m_vec[l-1])
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
            a=abs(np.dot(gamma,gamma_old))**2
            gamma=a*gamma+(1-a)*gamma_old
            angles[l]=1- abs(np.dot(gamma,gamma_old))
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
    draw_plots(lpc_points,m_vec,angles,cycle)
    
    return lpc_points
#%%Drawing functions 
def draw_plots(lpc_points,m_vec,angles,cycle):
    #plot the data
    #if points are to be saved check if the folder already exists
    lpc_cache=lpc_points
    #eliminate the unused points
    lpc_points=lpc_points[~np.all(lpc_points==0, axis=1)]
    lpc_points=lpc_points*norm
    fig1 = plt.figure(figsize=(7, 7))
    ax = fig1.add_subplot(111, projection='3d')
    fig1.tight_layout()
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
    fig2.tight_layout()
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
    plt.show(block=False)



def show_plot():
    hits=X
    amps=cut_array
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    hits=hits*norm
    img=ax.scatter(hits[:,0], hits[:,1], hits[:,2], c=amps,s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Amplitude',size=14,weight='bold',labelpad=10.)
    plt.title("Event heatmap",size=20)
    plt.pause(0.05)
    plt.show(block=False)


def plot_heatmap(hits,amps,cycle):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    hits=hits*norm
    img=ax.scatter(hits[:,0], hits[:,1], hits[:,2], c=amps,s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Amplitude',size=14,weight='bold',labelpad=10.)
    plt.title("Event heatmap",size=20)
    plt.pause(0.05)
    plt.show(block=False)



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
    return array




#%%GUI setup
sg.theme('DarkAmber')    # Keep things interesting for your users
font_title = ("Gill Sans", 20)
font_corpus= ("Gill Sans", 16)
def TextLabel(text): return sg.Text(text+': ', justification='r',font=font_corpus, size=(6,1))


browse_layout = [[sg.T("")], [sg.Text("Choose a file: ",font=font_corpus), 
                              sg.Input(font=font_corpus), 
                              sg.FileBrowse(key="-File-",font=font_corpus)],
                 [sg.Button("Open",font=font_corpus)]]

browse_window = sg.Window("Browse Panel",browse_layout)


#%% Main
if  __name__ == "__main__":
    folder = "./Plots/gui_test"
    #Opening of the pickled file
    print("++++++++++++")
    
    
    while True:                             # The Event Loop
    
        event,values=browse_window.Read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == "Open":
            file_sel = str(values.get('-File-'))
            print("Selected .pkl: "+file_sel)
            dictionary=pickle.load(open(file_sel,"rb"))
            key_name, dictionary = next(iter(dictionary.items()))
        #take one of the events (this will have to remain )
            #use og_array to get the ize of the volume
            key_list=list(dictionary[0])
            og_array=np.array(dictionary[0][key_list[0]]['amplitude'])
            layout = [[sg.T("")], [sg.Text("Choose a file: ",font=font_corpus), 
                                              sg.Input(file_sel,font=font_corpus), 
                                              sg.FileBrowse(key="-File-",font=font_corpus)],
                          [sg.Button("Submit",font=font_corpus)],
                          [sg.Text('Enter event',font=font_title)],      
                          [sg.Combo(np.arange(len(dictionary)),default_value='0',font=font_corpus,key='-IN0-')],
                          [sg.Text('Enter parameters',font=font_title)],
                          [TextLabel('b_x'),sg.Input('10',key='-IN1-',justification='l',font=font_corpus)],  
                          [TextLabel('b_y'),sg.Input('10',key='-IN2-',justification='l',font=font_corpus)],  
                          [TextLabel('b_z_d'),sg.Slider(range=(0,og_array.shape[2]),default_value ='0',orientation = 'horizontal',key='-IN3-',font=font_corpus)],
                          [TextLabel('b_z_u'),sg.Slider(range=(0,og_array.shape[2]),default_value ='130',orientation = 'horizontal',key='-IN4-',font=font_corpus)],
                          [TextLabel('c_frac'),sg.Input('0.90',key='-IN5-',justification='l',font=font_corpus)],
                          [sg.Button('Plot',font=font_corpus),sg.Button('Start LPC',font=font_corpus), sg.Exit(font=font_corpus)]]      
            window = sg.Window('Control Panel', layout)      
            browse_window.Close()
            while True:
                
                
                event,values = window.read() 
                j=int(values.get('-IN0-'))
                key_list=list(dictionary[j])
                ev=dictionary[j][key_list[0]]['amplitude']
                og_array=np.array(ev)
                #convert the event dic. to numpy array summing over the cameras
                for i in range(1,len(dictionary[j])):
                    ev=dictionary[j][key_list[i]]['amplitude']
                    og_array+=np.array(ev)
                array=np.empty_like(og_array)
                b_x=int(values.get('-IN1-'))
                b_y=int(values.get('-IN2-'))
                b_z_d=int(values.get('-IN3-'))
                b_z_u=int(values.get('-IN4-'))
                c_frac=float(values.get('-IN5-'))
                array=og_array[b_x:(array.shape[0]-b_x),b_y:(array.shape[1]-b_y),b_z_d:(b_z_u)]
                #define the complete coordinate arrays
                x= np.arange(0.,og_array.shape[0],1)
                y=np.arange(0.,og_array.shape[1],1)
                z=np.arange(0.,og_array.shape[2],1)
                #rescaling the amplitudes
                array=array-np.min(array)
                array = array/np.max(array)
                cut_array=array[array>=c_frac]
                x_cut=np.nonzero(array>=c_frac)[0]+b_x
                y_cut=np.nonzero(array>=c_frac)[1]+b_y
                z_cut=np.nonzero(array>=c_frac)[2]+b_z_d
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
                if event == sg.WIN_CLOSED or event == 'Exit':
                    break      
                elif event=='Plot':
                    plt.close()
                    show_plot()
                elif event=='Start LPC':
                    track_cycle(X,cut_array)
                    
                    
            window.close()
            
            
            
    browse_window.close()
    window.close()