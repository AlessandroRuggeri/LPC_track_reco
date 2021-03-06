#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:47:08 2022

@author: alessandro
"""
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
#%%Drawing functions

def show_plot(X,cut_array,norm,x,y,z,event_num,save_fol):
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
    plt.title("Event {0} heatmap".format(event_num),size=20)
    plt.savefig(('{0}/Ev_{1}.pdf').format(save_fol,event_num))
    plt.draw()




def plot_fit(m_vec,x,y,z,x_par,y_par,z_par,t,cycle,event_num,save_fol):
    #maybe separate this in another function in plot_save
    #plot the fitted lines over the LPC mean curves
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.tight_layout()
    ax.set_xlim((x[0],x[x.shape[0]-1]))
    ax.set_ylim((y[0],y[y.shape[0]-1]))
    ax.set_zlim((z[0],z[z.shape[0]-1]))
    ax.set_xlabel('x',fontsize=14,weight='bold')
    ax.set_ylabel('y',fontsize=14,weight='bold')
    ax.set_zlabel('z',fontsize=14,weight='bold')
    img=ax.scatter(m_vec[:,0],m_vec[:,1],m_vec[:,2],c=np.arange(0,m_vec.shape[0]),s=20,marker='s')
    cb=plt.colorbar(img,shrink=0.5,orientation='vertical', pad=0.1)
    cb.set_label(label='Point index',size=14,weight='bold',labelpad=10.)
    plt.title("Ev {0} - {1} mean points fit".format(event_num,cycle),size=20)
    plt.plot(x_par[1]+t[:]*x_par[0],y_par[1]+t[:]*y_par[0],z_par[1]+t[:]*z_par[0],'r',linewidth=3)
    plt.savefig('{0}/Event{1}_fit_{2}.pdf'.format(save_fol,event_num,cycle))
    plt.draw()



def save_results(file_sel,event_num,save_fol,c_frac_l,c_frac_u,CM_width,G_CM_width,n_cyc,
                 n_width,n_neg):
    #save the event parameters
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    with open('{0}/Ev{1}_parameters.txt'.format(save_fol,event_num), 'w') as s:
        print("File: {0}".format(file_sel),file=s)
        print("Event: {0}".format(event_num),file=s)
        print("Lower cut fraction: {0}".format(c_frac_l),file=s)
        print("Upper cut fraction: {0}".format(c_frac_u),file=s)
        print("Local C.O.M neighborhood width: {0} cm".format(CM_width),file=s)
        print("Global C.O.M neighborhood width: {0} cm".format(G_CM_width),file=s)
        print("Cycles: {0}".format(n_cyc),file=s)
        print("LPC Neighborhood width: {0} cm".format(n_width),file=s)
        print("Neglected closest {0} cm when repeating".format(n_neg),file=s)
        s.close()

def show_wrap():
    plt.show()