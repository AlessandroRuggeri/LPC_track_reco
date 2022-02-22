#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:35:12 2022

@author: alessandro
"""

import numpy as np
import math as mt
import os
#import cdist form scipy to efficiently compute distances between all pairs of points
from scipy.spatial.distance import cdist

import plot_save

#%%Functions for frequently used operations

#computing the weight vector
#We take the starting point and the closest points for which to compute the weights
def weight(u,hits,amplitudes,h):
    #computing the weights
    w = (amplitudes/
            ((h**3)*((2*mt.pi)**(3/2))))*np.exp((-1/(2*(h**2)))*np.linalg.norm(hits-u,axis=1)**2)
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
        # print("Zero Division Error!!!")
        # print("The weights:")
        # print(weights)
        m=np.array([0.,0.,0.])
        zero_flag=True
    return m, zero_flag

#Computing the local covariance matrix
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

#Get points within n_width cm
def N_closest(u,hits,amps,n_width,norm):
    dist = np.zeros(hits.shape[0])
#computing the distance between u and each hits[i]
    dist=np.linalg.norm(hits-u,axis=1)
#selecting the points closer than N
    dist = np.where(np.logical_and(dist>0, dist<=n_width/norm))[0]
#slicing the hits matrix over the closest indices
    return np.take(hits,dist,axis=0),np.take(amps,dist,axis=0)

#Remove the old points when computing the LPC
def remove_old(points,points_old,amplitudes):
    #print("points shape: ",points.shape,", old points shape: ",points_old.shape)
    mask=(points[:,None]==points_old).all(-1).any(-1)
    mask=np.invert(mask)
    return points[mask], amplitudes[mask]
#%% LPC cycle functions

#refined search of the center of mass 
def find_com(hits,amps,norm,CM_width,log):
    print("++ Starting C.O.M computation ++")
    print("++ Starting C.O.M computation ++",file=log)
    print("++ With {0} cm local neighborhood width ++".format(CM_width),file=log)
    #compute the local c.o.m
    try:
        #compute the c.o.m. of the remaining hits
        c_mass=np.average(hits,axis=0,weights=amps)
        c_mass=np.reshape(c_mass,(1,3))
        #find the index of the hit closest to the c.o.m.
        index=(cdist(hits,c_mass)).argmin()
        #use this hit as the starting point
        start=hits[index]
        print("++ Initial local C.O.M: {0} (cm)".format(start*norm),file=log)
    except ZeroDivisionError:
        print("-- Too few points, starting from [0,0,0] --",file=log)
        start=np.array([0.,0.,0.])
    #find the hits closer than CM_width and compute their c.o.m.
    #cycle until the c.o.m. position stabilizes (or enough times)
    for i in range(30):
        start_old=start
        close_hits, close_amps=N_closest(start,hits,amps,CM_width,norm)
        try:
            c_mass=np.average(close_hits,axis=0,weights=close_amps)
            c_mass=np.reshape(c_mass,(1,3))
            index=(cdist(close_hits,c_mass)).argmin()
            start=close_hits[index]
            print("++ Cycle {0} local C.O.M: {1} (cm)".format(i,start*norm),file=log)
        except ZeroDivisionError:
            print("-- Too few points left: exiting --",file=log)
            break
        if (np.linalg.norm(start-start_old)<=3/norm):
            break
    return start


#cycle over the hit clusters to find possible tracks
def track_cycle(hits,amps,norm,N_p,n_cyc,n_width,n_neg,x,y,z,CM_width,
                event_num,save_fol):
    #keep the complete set of voxels to compute the lpc on
    og_hits=hits
    og_amps=amps
    m_array=np.zeros((n_cyc,N_p,3))
    #keep a log of the LPC cycles
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    with open('{0}/Ev{1}_lpc_log.txt'.format(save_fol,event_num), 'w') as log:
        print('++ Event {0}'.format(event_num),file=log)
        print("++ Starting track cycles ++")
        for i in range(n_cyc):
            print("++ Track {0} ++".format(i))
            print("++ Track {0} ++".format(i),file=log)
            #find the starting point 
            start=find_com(hits,amps,norm,CM_width,log)
            print("++ Starting point: {0} (cm)".format(start*norm),file=log)
            # perform an lpc cycle and save the resulting points (and graphs)
            print("++ Starting LPC cycles ++",file=log)
            lpc_points,m_array[i]=lpc_cycle(start,og_hits,og_amps,n_width,i,
                                            N_p,norm,x,y,z,event_num,save_fol,log)
            #print("lpc shape: ",lpc_points.shape)
            #remove from X the hits in the vicinity of the lpc curve
            amps=amps[np.all(cdist(hits,lpc_points)>=n_neg/norm,axis=1)]
            hits= hits[np.all(cdist(hits,lpc_points)>=n_neg/norm,axis=1)]
            #return to the original scale
            m_array[i]=m_array[i]*norm
            try:
                print("++ Fitting the LPC means ++")
                print("++ Fitting the LPC means ++",file=log)
                parametric_fit(m_array[i],i,x,y,z,event_num,save_fol)
            except TypeError:
                print("++ Fitting the LPC means ++",file=log)
                print("-- Empty mean points vector: exiting --",file=log)
        print("++ LPC computation complete ++ ")
        print("++ LPC computation complete ++ ",file=log)
        log.close()

#compute the center of mass of blobs
def COM_cycle(hits,amps,norm,CM_width,G_CM_width,event_num,save_fol):
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    with open('{0}/Ev{1}_COM_log.txt'.format(save_fol,event_num), 'w') as log:
        print('++ Event {0}'.format(event_num),file=log)
        #find the starting point 
        start=find_com(hits,amps,norm,CM_width,log)
        print("++ Final local C.O.M: {0} (cm)".format(start*norm),file=log)
        #once at the center of the cluster compute its c.o.m.
        close_hits, close_amps=N_closest(start,hits,amps,G_CM_width,norm)
        try:
            c_mass=np.average(close_hits,axis=0,weights=close_amps)
            c_mass=np.reshape(c_mass,(1,3))
            index=(cdist(close_hits,c_mass)).argmin()
            center_est=close_hits[index]
            center_est=center_est*norm
            print("++ C.O.M neighborhood width: {0} cm".format(G_CM_width),file=log)
            print('++ C.O.M= {0} (cm)'.format(center_est),file=log)
        except ZeroDivisionError:
            print("-- Too few points left: exiting.",file=log)
        print("++ C.O.M computation complete ++ ")
        print("++ C.O.M computation complete ++ ",file=log)
        log.close()

#Perform the LPC cycle
def lpc_cycle(m0,hits,amps,n_width,cycle,N_p,norm,x,y,z,event_num,
              save_fol,log):
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
        print("+++++++++++++++++++++",file=log)
        print("Cycle {0}".format(l),file=log)
        print("LPC point = ",lpc_points[l]*norm," (cm)",file=log)
        count+=1
        #find the points of the N-wide neighborhood of lpc_points[l]
        closest, amplitudes=N_closest(lpc_points[l],hits,amps,n_width,norm)
        #remove the points featured in the previous cycle
        closest, amplitudes=remove_old(closest,closest_old,amplitudes)
        #compute the local mean
        m_vec[l],zero_flag=loc_mean(lpc_points[l],closest,amplitudes,h)
        if zero_flag:
            print("++ Zero division error: exiting cycle ++")
            print("++ Zero division error: exiting cycle ++",file=log)
            break
        print("Local mean = ",m_vec[l],file=log)
        #compute the path length
        if l>0:
            pathl+=np.linalg.norm(m_vec[l]-m_vec[l-1])
        sigma = loc_cov(lpc_points[l],closest,m_vec[l],amplitudes,h)
        try:
            val,gamma=np.linalg.eigh(sigma)
        except np.linalg.LinAlgError:
            print("++ Eigenvalues did not converge: exiting ++")
            print("++ Eigenvalues did not converge: exiting ++",file=log)
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
            print("++ Could not reach convergence ++")
            print("++ Could not reach convergence ++",file=log)
            break
        #save the "old" variables
        closest_old=closest
        #amplitudes_old=amplitudes
        if l==0:
            R=1
        else:
            R=(pathl-pathl_old)/(pathl+pathl_old)
            print("R = ",R,file=log)
        #update the "old" variables
        gamma_old=gamma
        pathl_old=pathl
        if R<7*10**(-4):
            if f_b ==-1:
                print("++ Curve has converged: exiting ++")
                print("++ Curve has converged: exiting ++",file=log)
                break
            else:
                print("----------",file=log)
                print("Inverting",file=log)
                print("----------",file=log)
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
                    print("++ Reached max. number of points: exiting ++")
                    print("++ Reached max. number of points: exiting ++",file=log)
                    break
                else:
                    print("+++++++++++",file=log)
                    print("+Inverting+",file=log)
                    print("+++++++++++",file=log)
                    f_b=-1
                    lpc_points[l+1]=m0
                    c=1
                    count=0
    return lpc_points, m_vec

#3D fitting using the parametric equation of a straight line
def parametric_fit(m_vec,cycle,x,y,z,event_num,save_fol):
    #m_vec is already rescaled
    #cut the non-assigned points
    m_vec=m_vec[~np.all(m_vec==0, axis=1)]
    t = np.linspace(0,1,m_vec.shape[0])
    x_par=np.polyfit(t,m_vec[:,0],1)
    y_par=np.polyfit(t,m_vec[:,1],1)
    z_par=np.polyfit(t,m_vec[:,2],1)
    plot_save.plot_fit(m_vec,x,y,z,x_par,y_par,z_par,t,cycle,event_num,save_fol)
    #first and last projections of the LPC mean points by t parameter
    #define the starting point for the line and the unit direction vector
    a=np.column_stack((x_par[1],y_par[1],z_par[1]))
    n=np.column_stack((x_par[0],y_par[0],z_par[0]))
    #define the endpoint of the fit line
    b=a+n*1
    projections=a+(np.einsum('ij,kj-> ik',(m_vec-a),(b-a))/np.dot((b-a)[0],(b-a)[0]))*(b-a)[0]
    #find the t of the projected points
    proj_t=np.linalg.norm(projections-a[0],axis=1)/np.linalg.norm(n[0])
    #now sort the projections by their t
    proj_t_sort = proj_t.argsort()
    projections=projections[proj_t_sort[::+1]]
    #save the fit parameters in the log file
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    #rewrite the file if in the first LPC cycle, append the parameters if in the 
    #subsequent ones
    if(cycle == 0):
        with open('{0}/Ev{1}_fit_par.txt'.format(save_fol,event_num), 'w') as s:
            print('++ Event {0}'.format(event_num),file=s)
            print('++ Cycle {0} parameters'.format(cycle),file=s)
            print("x0: {0} cm".format(x_par[1]),file=s)
            print("x1: {0} cm".format(x_par[0]),file=s)
            print("y0: {0} cm".format(y_par[1]),file=s)
            print("y1: {0} cm".format(y_par[0]),file=s)
            print("z0: {0} cm".format(z_par[1]),file=s)
            print("z1: {0} cm".format(z_par[0]),file=s) 
            print("First LPC proj. point: {0} (cm)".format(projections[0]),file=s)
            print("Last LPC proj. point: {0} (cm)".format(projections[projections.shape[0]-1]),file=s)
            s.close()
    else:
        with open('{0}/Ev{1}_fit_par.txt'.format(save_fol,event_num), 'a') as s:
            print('++ Cycle {0} parameters'.format(cycle),file=s)
            print("x0: {0} cm".format(x_par[1]),file=s)
            print("x1: {0} cm".format(x_par[0]),file=s)
            print("y0: {0} cm".format(y_par[1]),file=s)
            print("y1: {0} cm".format(y_par[0]),file=s)
            print("z0: {0} cm".format(z_par[1]),file=s)
            print("z1: {0} cm".format(z_par[0]),file=s)
            print("First LPC proj. point: {0} (cm)".format(projections[0]),file=s)
            print("Last LPC proj. point: {0} (cm)".format(projections[projections.shape[0]-1]),file=s)
            s.close()