#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:06:25 2022

@author: alessandro
"""

import PySimpleGUI as sg      
import matplotlib.pyplot as plt
import numpy as np
import pickle

def draw_plot(a,b,c,d):
    t = np.arange(-d, d,0.001)
    s=a*(t**2)+b*t+c
    plt.plot(t,s)
    plt.ylim(-10, 10)
    plt.title("Event heatmap")
    plt.pause(0.05)
    plt.show(block=False)

def plot_heatmap():
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





if  __name__ == "__main__":
    folder = "./Plots/gui_test"
    #Opening of the pickled file
    print("++++++++++++")
    file_sel = "./pickles/3drecoGrain.pkl"
    print("Selected .pkl: "+file_sel)
    dictionary=pickle.load(open(file_sel,"rb"))
    key_name, dictionary = next(iter(dictionary.items()))
    #take one of the events (this will have to remain )
    
    sg.theme('DarkAmber')    # Keep things interesting for your users
    font_title = ("Gill Sans", 20)
    font_corpus= ("Gill Sans", 16)
    def TextLabel(text): return sg.Text(text+': ', justification='r',font=font_corpus, size=(6,1))
    layout = [[sg.Text('Enter event',font=font_title)],      
              [sg.Combo(np.arange(len(dictionary)),default_value='0',font=font_corpus,key='-IN0-')],
              [sg.Text('Enter parameters',font=font_title)],
              [TextLabel('b_x'),sg.Input('10',key='-IN1-',justification='l',font=font_corpus)],  
              [TextLabel('b_y'),sg.Input('10',key='-IN2-',justification='l',font=font_corpus)],  
              [TextLabel('b_z'),sg.Input('10',key='-IN3-',justification='l',font=font_corpus)],
              [TextLabel('c_frac'),sg.Input('0.95',key='-IN4-',justification='l',font=font_corpus)],
              [sg.Button('Plot',font=font_corpus), sg.Exit(font=font_corpus)]]      
    window = sg.Window('Control Panel', layout)      

        
    while True:                             # The Event Loop
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
            
        array=np.empty_like(og_array)
        print(event,values)
        b_x=int(values.get('-IN1-'))
        b_y=int(values.get('-IN2-'))
        b_z=int(values.get('-IN3-'))
        c_frac=float(values.get('-IN4-'))
        array=og_array[b_x:(array.shape[0]-b_x),b_y:(array.shape[1]-b_y),b_z:(array.shape[2]-b_z)]
        #define the complete coordinate arrays
        x= np.arange(-array.shape[0]//2,array.shape[0]//2,1)
        y=np.arange(-array.shape[1]//2,array.shape[1]//2,1)
        z=np.arange(-array.shape[2]//2,array.shape[2]//2,1)
        #rescaling the amplitudes
        diff=np.max(array)-np.min(array)
        array = array/diff
        cut=c_frac*np.max(array)
        cut_array=array[array>=cut]
        x_cut=np.nonzero(array>=cut)[0]
        y_cut=np.nonzero(array>=cut)[1]
        z_cut=np.nonzero(array>=cut)[2]
        #defining the matrix of cut coordinates
        #reshaping the index vectors to get the right shape for X
        x_cut=x_cut.transpose()-array.shape[0]//2+0.5
        y_cut=y_cut.transpose()-array.shape[1]//2+0.5
        z_cut=z_cut.transpose()-array.shape[2]//2+0.5
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
            plot_heatmap()
    
    window.close()