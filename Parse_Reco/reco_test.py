#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 21:31:31 2022

@author: alessandro
"""
#import of the libraries
import numpy as np
import pickle
import argparse as ap
import os
import pytest
#import of the function files
import lpc_functions
import plot_save

#%% Main function code
def main(a_pickle='/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
         a_sf='/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',a_ev=33,a_lc=0.97,
         a_uc=1.,a_cn=1,a_CMw=10,a_lpcw=6,a_exw=5,a_lpcc='y',a_sc='n'):
    parser = ap.ArgumentParser(description="Event reconstruction parameters")
    group1=parser.add_argument_group('Initial settings')
    group1.add_argument("--pickle",help="select the file to be opened",
                        default='/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl')
    group1.add_argument("--save_fol",help="select the save folder",
                        default='/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default')
    group1.add_argument("--event",help="select event to be checked",type=int,default=33)
    group2=parser.add_argument_group('Amplitude cuts')
    group2.add_argument('--lower_cut',type=float,help="Lower amplitude cut",default=0.97)
    group2.add_argument('--upper_cut',type=float,help="Upper amplitude cut",default=1.)
    group3=parser.add_argument_group('LPC parameters')
    group3.add_argument('--Cyc_num',type=int,help="Number of LPC cycles",default=1)
    group3.add_argument('--CM_neigh_width',type=int,help="Voxel width for CM computation",default=10)
    group3.add_argument('--LPC_neigh_width',type=int,help="Voxel width for LPC computation",default=6)
    group3.add_argument('--excl_width',type=int,help="Excluded voxel width for new starting point",default=5)
    group4=parser.add_argument_group('Confirms')
    group4.add_argument('--lpc_confirm',type=str,help="Compute the LPC? [y if yes]",default='y')
    group4.add_argument('--save_confirm',type=str,help="Save the parameters? [y if yes]",default='n')

    #parser.print_help()
    args=parser.parse_args(["--pickle",str(a_pickle),"--save_fol",str(a_sf),
                            "--event",str(a_ev),'--lower_cut',str(a_lc),'--upper_cut',str(a_uc),
                            '--Cyc_num',str(a_cn),'--CM_neigh_width',str(a_CMw),
                            '--LPC_neigh_width',str(a_lpcw),'--excl_width',str(a_exw),
                            '--lpc_confirm',str(a_lpcc),'--save_confirm',str(a_sc)])
    
    print("++++++++++++")
    file_sel=args.pickle
    # try:
    dictionary=pickle.load(open(file_sel,"rb"))
    # except FileNotFoundError as e:
    #     raise FileNotFoundError("-- Please select a valid file! --") from e
        
    print("++ Selected .pkl: "+file_sel+" ++")
    key_name, dictionary = next(iter(dictionary.items()))
#take one of the events (this will have to remain )
    #use og_array to get the size of the volume

    #initialize the save folder
    save_fol=args.save_fol
    try:
        os.mkdir(save_fol)
    except FileExistsError:
        pass
    #get the requested event from the file
    #added an explaining message to the standard KeyError one
    try:
        j=args.event
        key_list=list(dictionary[j])
    except KeyError as e:
        raise KeyError("-- Invalid event number! --") from e
    #get the amplitudes from the pickle dictionary as an array
    ev=dictionary[j][key_list[0]]['amplitude']
    array=np.array(ev)
    #convert the event dic. to numpy array summing over the cameras
    for i in range(1,len(dictionary[j])):
        ev=dictionary[j][key_list[i]]['amplitude']
        array+=np.array(ev)
    #define array to use in the following
    #save the original side dimensions
    x_size=array.shape[2]/2
    y_size=array.shape[1]/2
    z_size=array.shape[0]/2
    array=array[5:(array.shape[0]-5),5:(array.shape[1]-5),5:(array.shape[2]-5)]
    #define the complete coordinate arrays
    x= np.arange(-x_size,x_size,1)
    y=np.arange(-y_size,y_size,1)
    z=np.arange(-z_size,z_size,1)
    #rescaling the amplitudes
    array=array-np.min(array)
    array = array/np.max(array)
    
    c_frac_l=args.lower_cut
    c_frac_u=args.upper_cut
    #exit if the cut values are not valid
    if (c_frac_l <0) or (c_frac_l >1.) or (c_frac_l >= c_frac_u):
        raise ValueError("-- Invalid lower cut fraction! --")
    elif (c_frac_u <0) or (c_frac_u >1.):
        raise ValueError("-- Invalid upper cut fraction! --")
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
        raise ValueError("-- No remaining points: try other cuts! --")          
    if norm==0.:
        raise ValueError("-- Only one point: cannot normalize! --"+'\n'+"-- Try other cuts! --")
    else:
        X = X/norm
    #set the max number of LPC points as a main var.
    N_p=200
    #get the LPC parameters
    CM_width=args.CM_neigh_width
    n_cyc=args.Cyc_num
    n_width=args.LPC_neigh_width
    n_neg=args.excl_width
    if n_cyc<=0:
        raise ValueError("-- Invalid LPC cycle number! --")
    
    print("++ Plotting ++")
    plot_save.show_plot(X,cut_array,norm,x,y,z,j,save_fol)
    
    lpc_confirm=args.lpc_confirm
    save_confirm = args.save_confirm
    if lpc_confirm=='y':
        try:
            lpc_functions.track_cycle(X,cut_array,norm,N_p,n_cyc,n_width,n_neg,
                                      x,y,z,CM_width,j,save_fol)
        except OSError:
            raise OSError("-- Select a valid folder! --")
    else:
        print("++ LPC not to be computed ++")
    if save_confirm == 'y':
        try:
            print("++ Saving parameters ++")
            plot_save.save_results(file_sel,j,save_fol,c_frac_l,
                                   c_frac_u,CM_width,n_cyc,n_width,n_neg)
        except OSError:
            raise OSError("-- Select a valid folder! --")
    else:
        print("++ Parameters not to be saved ++")
        
    print("+++ Execution complete +++")


#%%Testing main() with pytest
#The tests fail if no exception is raised
#Tests for invalid parsed inputs pass if SystemExit is raised
#Tests for subsequent exceptions and invalid conditions pass if exceptions are raised

#testing the file name
def test_valid_file():
    with pytest.raises(Exception):
        print(" ")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl')
def test_invalid_file():
    with pytest.raises(FileNotFoundError):
        print(" ")
        print("test_invalid_file")
        main("/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3")
def test_void_folder():
    with pytest.raises(Exception):
        print(" ")
        print("test_void_folder")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',"")
def test_invalid_folder():
    with pytest.raises(Exception):
        print(" ")
        print("test_invalid_folder")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             "/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default*é°ML_")
def test_valid_folder():
    with pytest.raises(Exception):
        print(" ")
        print("test_valid_folder")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default')

#Testing the event number input
#invalid input
def test_inv_event():
    with pytest.raises(SystemExit):
        print(" ")
        print("test_eq_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default','2w')
#event outside range
def test_out_event():
    with pytest.raises(Exception):
        print(" ")
        print("test_eq_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',60)

#Testing the cut fractions input
#equal cut values
def test_eq_frac():
    with pytest.raises(Exception):
        print(" ")
        print("test_eq_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,0.97)

#cut fraction above 1
def test_inv_frac():
    with pytest.raises(Exception):
        print(" ")
        print("test_inv_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,2.)
#negative cut fraction
def test_neg_frac():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,-0.97)
#test lower cut frac. higher than the upper
def test_inverted_frac():
    with pytest.raises(Exception):
        print(" ")
        print("test_inverted_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.98,0.97)
#high cut fraction: no exception expected  
def test_high_frac():
    with pytest.raises(Exception):
        print(" ")
        print("test_high_frac")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.99,1.)
              
#testing the number of cycles
#0 cycles
def test_zero_cyc():
    with pytest.raises(Exception):
        print(" ")
        print("test_zero_cyc")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.0,0)
#negative number of cycles
def test_neg_cyc():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_cyc")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,-2)
#much more cycles than tracks
#CHECK THIS: in print("Starting point: {0}".format(start*norm),file=log) start has to be an array (np.array)
def test_many_cyc():
    with pytest.raises(Exception):
        print(" ")
        print("test_many_cyc")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,5)
        
#testing the C.O.M neighborhood width
#0 cm width
def test_0_com():
    with pytest.raises(Exception):
        print(" ")
        print("test_0_com")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,0)
#negative width
def test_neg_com():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_com")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,-20)
#small width
def test_small_com():
    with pytest.raises(Exception):
        print(" ")
        print("test_small_com")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,1)
#large width     
def test_large_com():
    with pytest.raises(Exception):
        print(" ")
        print("test_large_com")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,80)
        
#testing the neighborhood parameter
#0 cm width
def test_0_width():
    with pytest.raises(Exception):
        print(" ")
        print("test_0_width")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,0)
#test small width
def test_small_width():
    with pytest.raises(Exception):
        print(" ")
        print("test_small_width")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,1)
#test fractional width
def test_frac_width():
    with pytest.raises(SystemExit):
        print(" ")
        print("test_frac_width")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,3.5)
#test very large widths
def test_large_width():
    with pytest.raises(Exception):
        print(" ")
        print("test_large_width")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,10)
#test negative widths
def test_neg_width():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_width")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,-3)
#testing the exclusion width parameter
#0 cycles #error
def test_0_negl():
    with pytest.raises(Exception):
        print(" ")
        print("test_0_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,0)
#test small width
def test_small_negl():
    with pytest.raises(Exception):
        print(" ")
        print("test_small_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,1)
#test fractional width
def test_frac_negl():
    with pytest.raises(SystemExit):
        print(" ")
        print("test_frac_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,3.5)
#test very large widths
def test_large_negl():
    with pytest.raises(Exception):
        print(" ")
        print("test_large_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,30)
def test_neg_negl():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,-3)
#testing the LPC computation condition
def test_lpc_inv():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,-3,'something')
#then the valid computation condition
def test_lpcc_inv():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,-3,'something')
#testing the parameters save condition 
def test_par_inv():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,-3,'no','something')
#now the valid save condition
def test_save_inv():
    with pytest.raises(Exception):
        print(" ")
        print("test_neg_negl")
        main('/Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl3.pkl',
             '/Users/alessandro/TesiMag/MURA_code/TrackReco/Parse_Reco/Plots/default',33,0.97,1.,1,10,6,-3,'no','something')
