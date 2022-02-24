#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:31:31 2022

@author: alessandro
"""

import PySimpleGUI as sg
import numpy as np
import pickle
#import of the functions
import lpc_functions
import plot_save

#%% GUI style definition
sg.theme('DarkAmber')    # Keep things interesting for your users
font_title = ("Gill Sans", 20)
font_corpus= ("Gill Sans", 18)
def TextLabel(text): return sg.Text(text+': ',
                                    justification='l',font=font_corpus,pad=(15,2), size=(16))


#make the browse window and layout in a function to allow selecting new pickles
def make_browse_window():
    #definition of the initial browse panel
    browse_layout = [[sg.T("")], [sg.Text("Choose a file: ",font=font_corpus),
                                  sg.Input(font=font_corpus),
                                  sg.FileBrowse(key="-File-",font=font_corpus,initial_folder=
                                                '/placeholder_path/placeholder.pkl')],
                     [sg.T("")],
                     [sg.Button("Open file",font=font_corpus),sg.Exit(font=font_corpus)]]
    return sg.Window("Browse Panel",browse_layout)

def main_gui():
    #Opening of the pickled file
    print("++++++++++++")
    browse_window = make_browse_window()
    while True:                             # The Event Loop
        event,values=browse_window.Read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            print("++ Closing program ++")
            break
        elif event == "Open file":
            file_sel = str(values.get('-File-'))
            try:
                dictionary=pickle.load(open(file_sel,"rb"))
            except FileNotFoundError:
                print("-- Please select a valid file! --")
                continue
            print("++ Selected .pkl: "+file_sel+" ++")
            key_name, dictionary = next(iter(dictionary.items()))
            #take the first of the events
            #get the initial and final event numbers
            in_event=list(dictionary)[0]
            fin_event=list(dictionary)[len(dictionary)-1]
            #use array to get the size of the volume
            key_list=list(dictionary[in_event])
            array=np.array(dictionary[in_event][key_list[0]]['amplitude'])
            lpc_layout = [[sg.T("")],
                          [sg.Text("Save folder: ",font=font_corpus),
                           sg.Input(font=font_corpus),
                           sg.FolderBrowse(key="-Fol-",
                                           font=font_corpus,initial_folder='/placeholder_folder')],
                          [sg.T("")],
                          [sg.Text('Enter event',font=font_title)],
                          [sg.Combo(np.arange(in_event,fin_event+1),default_value=str(in_event),font=font_corpus,key='-IN0-')],
                          [[sg.T("")],sg.Text('Enter amplitude cuts',font=font_title)],
                          [TextLabel('Lower amp. cut'),sg.Input('0.96',key='-IN5L-',justification='l',font=font_corpus,size=(5))],
                          [TextLabel('Upper amp. cut'),sg.Input('1.',key='-IN5U-',justification='l',font=font_corpus,size=(5))],
                          [[sg.T("")],sg.Button('Plot',font=font_corpus)],
                          [[sg.T("")],sg.Text('Enter C.O.M parameters',font=font_title)],
                          [TextLabel('Local CM width'),sg.Input('10',key='-INCM-',justification='l',font=font_corpus, size=(4))],
                          [TextLabel('Global CM width'),sg.Input('20',key='-INCMG-',justification='l',font=font_corpus, size=(4))],
                          [[sg.T("")],sg.Text('Enter LPC parameters',font=font_title)],
                          [TextLabel('Number of cyles'),sg.Input('1',key='-IN6-',justification='l',font=font_corpus, size=(3))],
                          [TextLabel('LPC neigh. width'),sg.Input('6',justification='l',key='-IN7-',font=font_corpus,size=(4))],
                          [TextLabel('Start excl. width'),sg.Input('10',justification='l',key='-IN8-',font=font_corpus,size=(4))],
                          [[sg.T("")],[sg.Button('Compute LPC',font=font_corpus),sg.Button('Compute C.O.M',font=font_corpus),sg.Button("Save Parameters",font=font_corpus)],
                          [sg.T("")],sg.Button('Change file',font=font_corpus),sg.Exit(font=font_corpus)]]
            proc_window = sg.Window('Control Panel', lpc_layout)
            browse_window.Close()
            while True:
    
                event,values = proc_window.read()
                #close the program if the button is pressed
                if event == sg.WIN_CLOSED or event == 'Exit':
                    break
                elif event=="Change file":
                    print("++ Select another file ++")
                    browse_window = make_browse_window()
                    break
                #initialize the parameters
                save_fol=str(values.get('-Fol-'))
                try:
                    j=int(values.get('-IN0-'))
                    key_list=list(dictionary[j])
                except (ValueError, KeyError):
                    print("-- Invalid event number! --")
                    continue
                ev=dictionary[j][key_list[0]]['amplitude']
                array=np.array(ev)
                #convert the event dic. to numpy array summing over the cameras
                for i in range(1,len(dictionary[j])):
                    ev=dictionary[j][key_list[i]]['amplitude']
                    array+=np.array(ev)
                #save the original side dimensions
                x_size=array.shape[2]/2
                y_size=array.shape[1]/2
                z_size=array.shape[0]/2
                #cut the closest 5 voxels to the sides to reduce artifacts
                array=array[5:(array.shape[0]-5),5:(array.shape[1]-5),5:(array.shape[2]-5)]
                #define the complete coordinate arrays
                x= np.arange(-x_size,x_size,1)
                y=np.arange(-y_size,y_size,1)
                z=np.arange(-z_size,z_size,1)
                #rescale the amplitudes
                array=array-np.min(array)
                array = array/np.max(array)
                #get the amplitude cuts
                try:
                    c_frac_l=float(values.get('-IN5L-'))
                    c_frac_u=float(values.get('-IN5U-'))
                except ValueError:
                    print("-- Invalid cut fraction! --")
                    continue
                #skip cycle if the cut values are not valid
                if (c_frac_l <0) or (c_frac_l >1.) or (c_frac_l >= c_frac_u):
                    print("-- Invalid lower cut fraction! --")
                    continue
                elif (c_frac_u <0) or (c_frac_u >1.):
                    print("-- Invalid upper cut fraction! --")
                    continue
                #perform the amplitude cuts on amp. and coord. arrays
                cut_array=array[(array>=c_frac_l) & (array<=c_frac_u)]
                x_cut=np.nonzero((array>=c_frac_l) & (array<=c_frac_u))[2]
                y_cut=np.nonzero((array>=c_frac_l) & (array<=c_frac_u))[1]
                z_cut=np.nonzero((array>=c_frac_l) & (array<=c_frac_u))[0]
                #defining the matrix of cut coordinates
                #reshaping the index vectors to get the right shape for X
                x_cut=x_cut.transpose()+5.5-x_size
                y_cut=y_cut.transpose()+5.5-y_size
                #flip an axis to get the right cartesian triplet
                z_cut=-(z_cut.transpose()+5.5-z_size)
                cut_array=cut_array.transpose()
                #rows for the events, columns for the coordinates
                X=np.column_stack((x_cut,y_cut,z_cut))
                #rescale the array of coordinates
                d = np.zeros([X.shape[0],X.shape[0]])
                d= np.linalg.norm(X,axis=1)
                try: 
                    norm=(d.max()-d.min())
                except ValueError:
                    print("-- No remaining points: try other cuts! --")
                    continue                
                if norm==0.:
                    print("-- Only one point: cannot normalize! --")
                    print("-- Try other cuts! --")
                    continue
                else:
                    X = X/norm
                #get the LPC parameters
                try:
                    CM_width=float(values.get('-INCM-'))
                    Glob_CM_width=float(values.get('-INCMG-'))
                    n_cyc=int(values.get('-IN6-'))
                    n_width=float(values.get('-IN7-'))
                    n_neg=float(values.get('-IN8-'))
                except ValueError:
                    print("-- Invalid LPC parameters! --")
                    continue
                if n_cyc<=0:
                    print("-- Invalid LPC cycle number! --")
                    continue
                
                if event=='Plot':
                    try:
                        print("++ Plotting ++")
                        plot_save.show_plot(X,cut_array,norm,x,y,z,j,save_fol)
                    except OSError:
                        print("-- Select a valid folder! --")
                        continue
                elif event=='Compute LPC':
                    try:
                        #set the max number of LPC points
                        N_p=200
                        lpc_functions.track_cycle(X,cut_array,norm,N_p,n_cyc,n_width,n_neg,
                                                      x,y,z,CM_width,j,save_fol)
                    except OSError:
                        print("-- Select a valid folder! --")
                        continue
                elif event=='Compute C.O.M':
                    try:
                        lpc_functions.COM_cycle(X,cut_array,norm,CM_width,Glob_CM_width,j,save_fol)
                    except OSError:
                        print("-- Select a valid folder! --")
                        continue
                elif event=="Save Parameters":
                    try:
                        print("++ Saving parameters ++")
                        plot_save.save_results(file_sel,j,save_fol,c_frac_l,
                                               c_frac_u,CM_width,Glob_CM_width,n_cyc,n_width,n_neg)
                    except OSError:
                        print("-- Select a valid folder! --")
                        continue
        proc_window.close()
    browse_window.close()
    
    
#%% Main
if  __name__ == "__main__":
    main_gui()