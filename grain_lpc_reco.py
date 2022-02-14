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
        w[i] = (amplitudes[i]/
                ((h**3)*((2*mt.pi)**(3/2))))*mt.exp((-1/(2*(h**2)))*np.dot((hits[i]-u),(hits[i]-u)))
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
    return points[mask], amplitudes[mask]

#%% LPC cycle functions

#refined search of the center of mass 
def find_com(hits,amps):
    #start with the global c.o.m.
    try:
        #compute the c.o.m. of the remaining hits
        c_mass=np.average(hits,axis=0,weights=amps)
        c_mass=np.reshape(c_mass,(1,3))
        #find the index of the hit closest to the c.o.m.
        index=(cdist(hits,c_mass)).argmin()
        #use this hit as the starting point
        start=hits[index]
    except ZeroDivisionError:
        print("Zero Division Error!!!")
        start=[0.,0.,0.]
    #find the hits closer than CM_width and compute their c.o.m.
    #cycle until the c.o.m. position stabilizes (or enough times)
    #then use the closest point to the c.o.m. as the start of the LP
    for i in range(30):
        start_old=start
        close_hits, close_amps=N_closest(start,hits,amps,CM_width)
        try:
            c_mass=np.average(close_hits,axis=0,weights=close_amps)
            c_mass=np.reshape(c_mass,(1,3))
            index=(cdist(close_hits,c_mass)).argmin()
            start=close_hits[index]
        except ZeroDivisionError:
            print("Too few points left: exiting.")
            break
        if (np.linalg.norm(start-start_old)<=3/norm):
            break
    return start


#cycle over the hit clusters to find possible tracks
def track_cycle(hits, amps):
    #keep the complete set of voxels to compute the lpc on
    og_hits=hits
    og_amps=amps
    m_array=np.zeros((n_cyc,N_p,3))
    for i in range(n_cyc):
        #find the starting point 
        start=find_com(hits,amps)
        # perform an lpc cycle and save the resulting points (and graphs)
        lpc_points,m_array[i]=lpc_cycle(start,og_hits,og_amps,n_width,i)
        print("lpc shape: ",lpc_points.shape)
        #remove from X the hits in the vicinity of the lpc curve
        amps=amps[np.all(cdist(hits,lpc_points)>=n_neg/norm,axis=1)]
        hits= hits[np.all(cdist(hits,lpc_points)>=n_neg/norm,axis=1)]
        #return to the original scale
        m_array[i]=m_array[i]*norm
        try:
            parametric_fit(m_array[i],i)
        except TypeError:
            print("Empty mean points vector: exiting")
        

# #function that determines the vertex with a closeness criterion
# def find_vert(m_array,par_ar_xy,par_ar_xz):
#     for i  in range(n_cyc):
#         #cycle in range(i,2) to count each pair only once
#         for j in range(i+1,n_cyc):
#             m_array1=m_array[i][~np.all(m_array[i]==0, axis=1)]
#             m_array2=m_array[j][~np.all(m_array[j]==0, axis=1)]
#             fit_points1=np.array([m_array1[:,0],m_array1[:,0]*par_ar_xy[i][0]+par_ar_xy[i][1],
#                                   m_array1[:,0]*par_ar_xz[i][0]+par_ar_xz[i][1]])
#             fit_points2=np.array([m_array2[:,0],m_array2[:,0]*par_ar_xy[j][0]+par_ar_xy[j][1],
#                                   m_array2[:,0]*par_ar_xz[j][0]+par_ar_xz[j][1]])
#             #transpose to get (n,3) arrays
#             fit_points1=fit_points1.transpose()
#             fit_points2=fit_points2.transpose()
#             #sort by the "x" coordinate
#             fit_points1=fit_points1[np.argsort(fit_points1[:, 0])]
#             fit_points2=fit_points2[np.argsort(fit_points2[:, 0])]
#             #get the array of distances between all the points
#             t_dist=cdist(fit_points1,fit_points2)
#             print(t_dist.shape)
#             ind=np.unravel_index(np.argmin(t_dist, axis=None), t_dist.shape)
#             if t_dist.min() <=n_width:
#                 print("Vertex between tracks {0} and {1} found".format(i,j))
#                 print("Distance: ",t_dist.min())
#                 print("Track 1 coordinate: ",fit_points1[ind[0]])
#                 print("Track 2 coordinate: ",fit_points2[ind[1]])
#             else:
#                 print("No vertex found")

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
        #find the points of the N-wide neighborhood of lpc_points[l]
        closest, amplitudes=N_closest(lpc_points[l],hits,amps,N)
        #remove the points featured in the previous cycle
        closest, amplitudes=remove_old(closest,closest_old,amplitudes)
        #compute the local mean
        m_vec[l],zero_flag=loc_mean(lpc_points[l],closest,amplitudes,h)
        if zero_flag:
                print("Zero division error: exiting cycle")
                break
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
            break
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
    return lpc_points, m_vec

#3D fitting using the parametric equation of a straight line
def parametric_fit(m_vec,cycle):
    #m_vec is already rescaled
    #cut the non-assigned points
    m_vec=m_vec[~np.all(m_vec==0, axis=1)]
    t = np.linspace(0,1,m_vec.shape[0])
    x_par=np.polyfit(t,m_vec[:,0],1)
    y_par=np.polyfit(t,m_vec[:,1],1)
    z_par=np.polyfit(t,m_vec[:,2],1)
    #plot the fitted lines over the LPC mean curves
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
    plt.plot(x_par[1]+t[:]*x_par[0],y_par[1]+t[:]*y_par[0],z_par[1]+t[:]*z_par[0],'r',linewidth=3)
    plt.savefig('{0}/Event_fit_{1}.pdf'.format(save_fol,cycle))
    #histogram of the point-line distances
    #define the starting point for the line and the unit direction vector
    # a=np.column_stack((x_par[1],y_par[1],z_par[1]))
    # n=np.column_stack((x_par[0],y_par[0],z_par[0]))
    # dist =np.zeros(m_vec.shape[0])
    # for i in range(m_vec.shape[0]):
    #     dist[i]=np.linalg.norm(np.cross((m_vec[i]-a),n))/np.linalg.norm(n)
    # fig1 = plt.figure(figsize=(7, 7))
    # plt.hist(dist,20,edgecolor='white')
    # plt.title("Points-fit distance histogram",size=20)
    # plt.xlabel('Point-fit distance (cm)',fontsize=16)
    # plt.ylabel('Occurrences',fontsize=16)
    # plt.savefig('{0}/Fit_dist_{1}.pdf'.format(save_fol,cycle))
    # plt.show()
    #save the fit parameters in the log file
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    with open('{0}/fit_par.txt'.format(save_fol), 'a') as s:
        print('Cycle {0} parameters'.format(cycle),file=s)
        print("x0: {0}".format(x_par[1]),file=s)
        print("x1: {0}".format(x_par[0]),file=s)
        print("y0: {0}".format(y_par[1]),file=s)
        print("y1: {0}".format(y_par[0]),file=s)
        print("z0: {0}".format(z_par[1]),file=s)
        print("z1: {0}".format(z_par[0]),file=s)
    
#%%Drawing functions
def draw_plots(lpc_points,m_vec,angles,cycle):
    #plot the data
    # lpc_cache=lpc_points
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
    img=ax.scatter(lpc_points[:,0],lpc_points[:,1],lpc_points[:,2],
                   c=np.arange(0,lpc_points.shape[0]),s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
    plt.title("LPC points plot",size=20)
    plt.savefig('{0}/Ev_{1}_LPC_c{2}.pdf'.format(save_fol,j,cycle))
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
    plt.savefig('{0}/Ev_{1}_mean_c{2}.pdf'.format(save_fol,j,cycle))
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
    plt.savefig(('{0}/Ev_{1}.pdf').format(save_fol,j))
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
    plt.savefig('{0}/Ev_{1}_c{2}.pdf'.format(save_fol,j,cycle))
    plt.pause(0.05)
    plt.show(block=False)


def save_results():
    #save the event parameters
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    with open('{0}/parameters.txt'.format(save_fol), 'w') as s:
        print("File: {0}".format(file_sel),file=s)
        print("Event: {0}".format(j),file=s)
        # print("b_x_l: {0}".format(b_x_l),file=s)
        # print("b_x_u: {0}".format(b_x_u),file=s)
        # print("b_y_l: {0}".format(b_y_l),file=s)
        # print("b_y_u: {0}".format(b_y_u),file=s)
        # print("b_z_l: {0}".format(b_z_l),file=s)
        # print("b_z_u: {0}".format(b_z_u),file=s)
        print("Lower cut fraction: {0}".format(c_frac_l),file=s)
        print("Upper cut fraction: {0}".format(c_frac_u),file=s)
        print("C.O.M neighborhood width: {0}".format(CM_width),file=s)
        print("Cycles: {0}".format(n_cyc),file=s)
        print("LPC Neighborhood width: {0}".format(n_width),file=s)
        print("Neglected {0} closest when repeating".format(n_neg),file=s)


#%%GUI setup
sg.theme('DarkAmber')    # Keep things interesting for your users
font_title = ("Gill Sans", 20)
font_corpus= ("Gill Sans", 18)
def TextLabel(text): return sg.Text(text+': ',
                                    justification='l',font=font_corpus,pad=(5,2), size=(10))

browse_layout = [[sg.T("")], [sg.Text("Choose a file: ",font=font_corpus),
                              sg.Input(font=font_corpus),
                              sg.FileBrowse(key="-File-",font=font_corpus,initial_folder=
                                            '/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles')],
                 [sg.T("")],
                 [sg.Button("Open file",font=font_corpus),sg.Exit(font=font_corpus)]]

browse_window = sg.Window("Browse Panel",browse_layout)


#%% Main
if  __name__ == "__main__":
    #Opening of the pickled file
    print("++++++++++++")


    while True:                             # The Event Loop

        event,values=browse_window.Read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == "Open file":
            file_sel = str(values.get('-File-'))
            try:
                dictionary=pickle.load(open(file_sel,"rb"))
            except FileNotFoundError:
                print("Please select a valid file!")
                continue
            print("Selected .pkl: "+file_sel)
            key_name, dictionary = next(iter(dictionary.items()))
        #take one of the events (this will have to remain )
            #use og_array to get the ize of the volume
            key_list=list(dictionary[0])
            og_array=np.array(dictionary[0][key_list[0]]['amplitude'])
            lpc_layout = [[sg.T("")], #[sg.Text("Choose a file: ",font=font_corpus),
                                              #sg.Input(file_sel,font=font_corpus),
                                              #sg.FileBrowse(key="-File-",font=font_corpus),
                                              #sg.Button("Open file",font=font_corpus)],
                          [sg.Text("Save folder: ",font=font_corpus),
                           sg.Input(font=font_corpus),
                           sg.FolderBrowse(key="-Fol-",
                                           font=font_corpus,initial_folder='/Users/alessandro/TesiMag/MURA_code/TrackReco/GRAIN_plots')],
                          [sg.T("")],
                          [sg.Text('Enter event',font=font_title)],
                          [sg.Combo(np.arange(len(dictionary)),default_value='0',font=font_corpus,key='-IN0-')],
                          [[sg.T("")],sg.Text('Enter dataset parameters',font=font_title)],
                          # [TextLabel('b_x_l'),sg.Slider(range=(0,og_array.shape[2]),
                          #                               default_value ='0',orientation = 'horizontal',key='-IN1L-',font=font_corpus)],
                          # [TextLabel('b_x_u'),sg.Slider(range=(0,og_array.shape[2]),
                          #                               default_value=og_array.shape[2],orientation = 'horizontal',key='-IN1U-',font=font_corpus)],
                          # [TextLabel('b_y_l'),sg.Slider(range=(0,og_array.shape[1]),
                          #                               default_value ='0',orientation = 'horizontal',key='-IN2L-',font=font_corpus)],
                          # [TextLabel('b_y_u'),sg.Slider(range=(0,og_array.shape[1]),
                          #                               default_value =og_array.shape[1],orientation = 'horizontal',key='-IN2U-',font=font_corpus)],
                          # [TextLabel('b_z_l'),sg.Slider(range=(0,og_array.shape[0]),
                          #                               default_value ='0',orientation = 'horizontal',key='-IN3L-',font=font_corpus)],
                          # [TextLabel('b_z_u'),sg.Slider(range=(0,og_array.shape[0]),
                          #                               default_value =og_array.shape[0],orientation = 'horizontal',key='-IN3U-',font=font_corpus)],
                          [TextLabel('Lower amp. cut'),sg.Input('0.96',key='-IN5L-',justification='l',font=font_corpus,size=(4))],
                          [TextLabel('Upper amp. cut'),sg.Input('1.',key='-IN5U-',justification='l',font=font_corpus,size=(4))],
                          [[sg.T("")],sg.Button('Plot',font=font_corpus)],
                          [[sg.T("")],sg.Text('Enter LPC parameters',font=font_title)],
                          [TextLabel('CM width'),sg.Slider(range=(1,20),
                                                          default_value ='10',orientation = 'horizontal',key='-INCM-',font=font_corpus)],
                          [TextLabel('Cycles'),sg.Input('1',key='-IN6-',justification='l',font=font_corpus, size=(3))],
                          [TextLabel('N_width'),sg.Slider(range=(1,10),
                                                          default_value ='6',orientation = 'horizontal',key='-IN7-',font=font_corpus)],
                          [TextLabel('Closest'),sg.Slider(range=(1,10),
                                                          default_value ='3',orientation = 'horizontal',key='-IN8-',font=font_corpus)],
                          [[sg.T("")],[sg.Button('Start LPC',font=font_corpus),sg.Button("Save Parameters",font=font_corpus)],
                          [sg.T("")],sg.Exit(font=font_corpus)]]
            proc_window = sg.Window('Control Panel', lpc_layout)
            browse_window.Close()
            while True:

                event,values = proc_window.read()
                save_fol=str(values.get('-Fol-'))
                j=int(values.get('-IN0-'))
                try:
                    key_list=list(dictionary[j])
                except KeyError:
                    print("Invalid event number!")
                    continue
                ev=dictionary[j][key_list[0]]['amplitude']
                og_array=np.array(ev)
                #convert the event dic. to numpy array summing over the cameras
                for i in range(1,len(dictionary[j])):
                    ev=dictionary[j][key_list[i]]['amplitude']
                    og_array+=np.array(ev)
                #og_array=np.swapaxes(og_array, 0, 2)
                array=np.empty_like(og_array)
                # b_x_l=int(values.get('-IN1L-'))
                # b_x_u=int(values.get('-IN1U-'))
                # b_y_l=int(values.get('-IN2L-'))
                # b_y_u=int(values.get('-IN2U-'))
                # b_z_l=int(values.get('-IN3L-'))
                # b_z_u=int(values.get('-IN3U-'))
                c_frac_l=float(values.get('-IN5L-'))
                c_frac_u=float(values.get('-IN5U-'))
                array=og_array[5:(array.shape[0]-5),5:(array.shape[1]-5),5:(array.shape[2]-5)]
                #define the complete coordinate arrays
                x= np.arange(-og_array.shape[2]/2,og_array.shape[2]/2,1)
                y=np.arange(-og_array.shape[1]/2,og_array.shape[1]/2,1)
                z=np.arange(-og_array.shape[0]/2,og_array.shape[0]/2,1)
                #rescaling the amplitudes
                array=array-np.min(array)
                array = array/np.max(array)
                #skip cycle if the cut values are not valid
                if (c_frac_l <0) or (c_frac_l >1.) or (c_frac_l >= c_frac_u):
                    print("Invalid lower cut fraction!")
                    continue
                elif (c_frac_u <0) or (c_frac_u >1.):
                    print("Invalid upper cut fraction!")
                    continue
                #perform the amplitude cuts on amp. and coord. arrays
                cut_array=array[(array>=c_frac_l) & (array<=c_frac_u)]
                x_cut=np.nonzero((array>=c_frac_l) & (array<=c_frac_u))[2]
                y_cut=np.nonzero((array>=c_frac_l) & (array<=c_frac_u))[1]
                z_cut=np.nonzero((array>=c_frac_l) & (array<=c_frac_u))[0]
                #defining the matrix of cut coordinates
                #reshaping the index vectors to get the right shape for X
                x_cut=x_cut.transpose()+5.5-og_array.shape[2]/2
                y_cut=y_cut.transpose()+5.5-og_array.shape[1]/2
                #flip an axis to get the right cartesian triplet
                z_cut=-(z_cut.transpose()+5.5-og_array.shape[0]/2)
                cut_array=cut_array.transpose()
                #rows for the events, columns for the coordinates
                X=np.column_stack((x_cut,y_cut,z_cut))
                #rescale the array of coordinates
                d = np.zeros([X.shape[0],X.shape[0]])
                for i in range(X.shape[0]):
                    d[i]=np.linalg.norm(X[i])
                norm=(d.max()-d.min())
                X = X/norm
                #set the max number of LPC points as a main var.
                N_p=200
                if event == sg.WIN_CLOSED or event == 'Exit':
                    break
                elif event=='Plot':
                    plt.close()
                    try:
                        show_plot()
                    except OSError:
                        print("Select a valid folder!")
                        continue
                elif event=='Start LPC':
                    CM_width=int(values.get('-INCM-'))
                    n_cyc=int(values.get('-IN6-'))
                    n_width=int(values.get('-IN7-'))
                    n_neg=int(values.get('-IN8-'))
                    try:
                        if n_cyc<=0:
                            print("Invalid LPC cycle number!")
                            continue
                        else:
                            track_cycle(X,cut_array)
                    except OSError:
                        print("Select a valid folder!")
                        continue
                elif event=="Save Parameters":
                    try:
                        save_results()
                    except OSError:
                        print("Select a valid folder!")
                        continue

            proc_window.close()
            
            
        proc_window.close()
    browse_window.close()