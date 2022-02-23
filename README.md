# LPC_track_reco
A Python program for the reconstruction of events in the GRAIN detector with a Local Principal Curve algorithm, using data obtained with an Hadamard-mask imaging technique.

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Dependencies](#dependencies)
* [Setup](#setup)
* [Usage](#usage)
* [Room for Improvement](#room-for-improvement)
 * [References](#references)

## General Information
The GRAIN sub-detector is set to complement the capabilities of the of the future DUNE experiment with its innovative use of UV imaging techniques in Liquid Argon.
One of such techniques makes use of Hadamard masks, patterned masks to be placed over the sensors adopted from X-ray astronomy and imaging [[1]](#1), allowing image reconstruction with an optimal depth of field [[2]](#2).

The best option for extracting observables, namely positions and directions of particle tracks, from data that is often marred by reconstruction artefacts, are **Local Principal Curve** (LPC) algorithms .

The purpose of the programs in this repository is to provide a way to quickly process GRAIN events coming from the simulations, retrieving the relevant position information:
- The positions of event vertices, estimated by the **centre of mass** (C.O.M) of the emission over a large enough neighbourhood.	
- The LPC reconstruction of *track events* with the parameters of the straight line fit, whose initial and final points estimate the start and endpoints of the track. The LPC algorithm is based on the one described in [[3]](#3).
## Features
The project, developed in Python 3.8, currently features:
- In the `grain_lpc_reco` folder: a **GUI-based** version of the program, made using the *PySimpleGUI* library, which allows to intuitively process the events, changing files, events and parameters at runtime.
- In the `Parse_reco` folder: a **command line** version of the program with parsed parameters, made using the *argparse* library, which allows for a swifter processing of individual events. 
- The two versions share the same same algorithm for the event reconstruction and plotting, implementing:
	- the 3D scatter plot of the event using Python's Matoplotlib library for the selected cuts in amplitude.
	- the computation of the centre of mass of the event in a neighbourhood of a selected width.
	- the LPC-based reconstruction of tracks with as of yet limited capabilities in distinguishing multiple tracks events.
	- the straight line fitting of the curve of LPC mean points with the estimation of the start and endpoint of the track.
	- plotting of the fitted LPC curves.
	- logs of the processing parameters and of the C.O.M, LPC and fit computations.
- A battery of tests for the parameter input and LPC functions implemented for the command line version in `Parse_Reco` using the `pytest` library.
	

## Dependencies
### Common dependencies
Both versions of the program require the following Python libraries:
- [numpy](https://numpy.org), which can be installed via command line with **pip** or **anaconda**:
    ```
    conda install numpy
    or
    pip install numpy
    ```
- [matplotlib](https://matplotlib.org), which can be installed by command line with **pip** or **anaconda**:
    ```
    pip install matplotlib
	or
    conda install matplotlib
    ```
- [scipy](https://scipy.org), which can be installed with either **pip** or **anaconda**, with the procedure detailed [here](https://scipy.org/install/)
- [math](https://docs.python.org/3/library/math.html), which is part of Python's standard library.
- [os](https://docs.python.org/3/library/os.html), which is part of Python's standard library.
- [pickle](https://docs.python.org/3/library/pickle.html), which is part of Python's standard library.
-  [pytest](https://docs.pytest.org/en/6.2.x/index.html), which can be installed by command line with: `pip install -U pytest`

### GUI-based version dependencies
In addition to the previously listed dependencies, the GUI-based version of the program needs the [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/) library, which can be installed  by command line with:
```
pip install pysimplegui
or
pip3 install pysimplegui
```
### Command line version dependencies
The [argparse](https://docs.python.org/3/library/argparse.html) library is required for the command line parsing version. The **argparse** module is part of the Python standard library.

## Setup
To start using the programs:
1. Clone this repository or download the folder containing the version that you need:
	- `grain_lpc_reco/` contains the GUI-based version of the program.
	- `Parse_reco/` contains the command line version.
2. Download the [shared folder] containing the .pkl files with the events.
3. Install the dependencies listed [above](#dependencies).
### Optional setup of grain_lpc_reco/
It is convenient to set default paths in which to search for the .pkl files and create the save folder. 

To set an initial folder for browsing the .pkl files, open `grain_lpc_reco/reco_gui.py` and, in the definition of `make_browse_window()` change:
```
 sg.FileBrowse(key="-File-",font=font_corpus,initial_folder='/placeholder_path')]
```
to the preferred path, for instance:
```
sg.FileBrowse(key="-File-",font=font_corpus,initial_folder='/valid_path')]
```
An initial path for the save folder can be set in the definition of `main_gui()` by changing

```
sg.FolderBrowse(key="-Fol-",font=font_corpus,initial_folder='/placeholder_save_folder')]
```
in the initialisation of `lpc_layout` to the preferred path, e.g.:
```
sg.FolderBrowse(key="-Fol-",font=font_corpus,initial_folder='/valid_save_folder')]
```
### Recommended setup of Parse_reco/
For the command line version it is recommended to set the default paths of the .pkl files and of the save folder.

To set the default .pkl file path open `reco_parse.py` and change:
```
group1.add_argument("--pickle",help="select the file to be opened",default='/placeholder_path/placeholder.pkl')
```
to the desired path, e.g.:
```
group1.add_argument("--pickle",help="select the file to be opened",		default='/valid_path/valid_pickle.pkl')
```
Then, to set the default save folder path change, just below:
```
group1.add_argument("--save_fol",help="select the save folder",default='/placeholder_save_folder')
```
to the desired path, e.g.:
```
group1.add_argument("--save_fol",help="select the save folder",default='/valid_save_folder')
```
### Setup of reco_test in Parse_reco/
In order to prime the battery of tests in `reco_test.py` it is necessary to change the placeholder default values of the `a_pickle` and `a_sf` arguments of `main` to valid paths in the `main` definition and in its calls in the test functions, e.g.:
```
main(a_pickle='/placeholder_path/placeholder.pkl',a_sf='/placeholder_path',a_ev=33,a_lc=0.97,
```
to:
```
main(a_pickle='/valid_path/valid_pickle.pkl',a_sf='/valid_folder',a_ev=33,a_lc=0.97,
```
and, in the test functions:
```
def test_valid_folder():
    with pytest.raises(Exception):
        print(" ")
        print("test_valid_folder")
        main('/placeholder_path/placeholder.pkl',
             '/placeholder_path/default')
```
to, for instance:
```
def test_valid_folder():
    with pytest.raises(Exception):
        print(" ")
        print("test_valid_folder")
        main('/valid_path/valid_pickle.pkl',
             '/valid_folder')
```
## Usage
### GUI-based version
1. To  start the GUI-based version, move to your Python environment and execute `reco_gui.py` from the command line with:
	```
	$ python reco_gui.py
	```
	or, using in the iPython shell:
	```
	In [1]: %run reco_gui.py
	```
2. This will open the *Browse Panel* window, where the .pkl file can be selected by clicking on **Browse**. Clicking on **Open file**  opens the selected .pkl and allows the program to proceed.
3. On the new *Control Panel* window, the Save folder must first be selected via the **Browse** button.
4. Set the event number with the **Event number** combo box and the [parameters](#the-execution-parameters) of your choice. The default values are clearly valid as well.
5. It is recommended, at first, to plot the event with the chosen amplitude cuts using the **Plot** button. Then:
	- click the **Compute LPC** button to start the LPC computation and fitting of the resulting curve, saving a log of the cycle in `/chosen_fol/Ev{n_ev}_lpc_log.txt` and the fit parameters in `/chosen_fol/Ev{n_ev}_fit_par.txt`.
	- click the **Compute C.O.M.** button to start the computation of the *global centre of mass* of the event, saving the resulting position in `/chosen_fol/Ev{n_ev}_COM_log.txt`.
	- click the **Save Parameters** button to save the current execution parameters in `/chosen_fol/Ev{n_ev}_parameters.txt`.
6. Once an event has been analyzed as needed, either:
	- click the **Exit** button to close the program.
	- click the **Change File** button select another file and restart the execution.
### Command line version
1. To start the command line version, move to your Python environment and execute `reco_parse.py` from the command line or from the iPython shell, **parsing the parameters** as [argparse](https://docs.python.org/3/library/argparse.html) arguments.
	- The arguments are **optional**, so they can be called in any order and are not required. All have default values that can be modified in `reco_parse.py`.
	- The scatterplot of the event is always shown.
2. Whenever needed, run the program with the `--help` argparse option to get a list of the argument names and descriptions. You may also want to look at the [following section](#the-execution-parameters).
3. In order to run the LPC or C.O.M. finding algorithms, or to save the execution parameters, the corresponding argparse arguments must be set to `y`, e.g.: 
	```
	$ python reco_parse.py --lpc_confirm y --com_confirm y --save_confirm y
	```
	to perform both computations and save the parameters. 
	- The LPC algorithm will compute the LPC curve, fit it and save a log of the cycle in `/chosen_fol/Ev{n_ev}_lpc_log.txt` and the fit parameters in `/chosen_fol/Ev{n_ev}_fit_par.txt`.
	- The C.O.M algorithm will compute the *global centre of mass* of the event, saving the resulting position in `/chosen_fol/Ev{n_ev}_COM_log.txt`.
4. For the program to finish the execution, all plots must be closed if working with matplotlib's auto backend.

An example of an execution from command line requesting the C.O.M. computation of `event 5` in `3dreco.pkl0.pkl` with a lower amplitude cut at 96%.
```
$ python reco_parse.py --pickle /Users/alessandro/TesiMag/MURA_code/TrackReco/Pickles/3dreco.pkl0.pkl --event 5 --lower_cut 0.97 --save_fol /Users/alessandro/TesiMag/MURA_code/TrackReco/Plots/ev_5 --lower_cut 0.96
```
### The execution parameters 
A brief guide to the execution parameters for the algorithm.
- The **Amplitude cuts** set the min. and max. probability amplitude for the voxels to be kept, as a fraction of the global maximum amplitude. It is recommended to start with **high lower cuts** (~96%), as features have amplitudes in this ballpark, and move down in small steps if needed.
- The **Centre of mass (C.O.M) parameters**:
	- The **Local CM width** is the width (in cm) of the neighbourhood of which to compute the *center of mass*, in order to find a starting point for the LPC algorithm.
	- The **Global CM width** is the width (in cm) of the neighbourhood over which the *centre of mass* of events is computed, centred at the voxel found with the Local CM width. The global CM of the event estimates the vertex position of *blob-like* events.
- The **LPC parameters**:
	- The **Number of cycles**  of the LPC algorithm that is required. For multiple track events a number $\geq 1$ might allow to separately reconstruct the tracks.
	- The **neighbourhood width** (in cm) around each LPC point with which to compute the next one in the cycle.
	- The **exclusion width** (in cm) around each LPC point whose voxels will be neglected when computing the starting point of cycles above the first.

## Room for Improvement
Room for improvement of current features:
- Improvement of the event plotting speed for low amplitude cuts by using a library optimized for 3D plotting.
- Implementation of geometric cuts that more closely match the detector volume.
- Improvements in the separate LPC reconstruction capabilities for multiple tracks.

Features to be introduced next:
- Definition of criteria to determine the amplitude cuts algorithmically.
- Finding of a way to identify artifacts.
- Definition criteria to distinguish automatically *track events* from *blob events*.

## References
<a id="1">[1]</a> Gottesman, Stephen & Fenimore, E. (1989). *New family of binary arrays for coded aperture imaging*. Applied optics. 28. 4344-52. 10.1364/AO.28.004344. 
<a id="2">[2]</a> Andreotti, M., Bernardini, P., Bersani, A. _et al._ *Coded masks for imaging of neutrino events.* _Eur. Phys. J. C_  81, 1011 (2021). 
<a id="3">[3]</a>  J. J. Back, G. J. Barker, S. B. Boyd, J. Einbeck, M. Haigh, B. Morgan, B. Oakley, Y. A. Ramachers and D. Roythorne. *Implementation of a local principal curves algorithm for neutrino interaction reconstruction in a liquid argon volume.* Eur. Phys. J. C, 74 3 (2014) 2832
