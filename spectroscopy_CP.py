#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:18:49 2022

@author: s1929920
"""

import bagpipes as pipes
import numpy as np
import pandas
import os 

from astropy.io import fits
from astropy.table import Table

path = "/home/s1929920/jwst/clara_code" #creating path for python to follow to find files
os.chdir(path)

filt_list = ["CFHT_u.dat",   #list of paths to data files for filter curves (wavlength v transmission)
             "CFHT_z.dat",
             "ACS_F606W.dat",
             "ACS_F814W.dat",
             "WFC3_F125W.dat",
             "WFC3_F140W.dat",
             "WFC3_F160W.dat",
             "IRAC_CH1.dat",
             "IRAC_CH2.dat",
             "IRAC_CH3.dat",
             "IRAC_CH4.dat"] #filter list

def bin(spectrum, binn):
    """ Bins up two or three column spectral data by a specified factor. """

    binn = int(binn)
    nbins = len(spectrum)/binn
    binspec = (np.zeros((int(nbins), spectrum.shape[1])))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 1] = np.mean(spec_slice[:, 1])

        if spectrum.shape[1] == 3:
            binspec[i,2] = (1./float(binn)*np.sqrt(np.sum(spec_slice[:, 2]**2)))
    
    return binspec


def load_vandels_spec(ID):
    """ Loads VANDELS spectroscopic data from file. """

    hdul = fits.open('vandels17433.fits')
    
    spectrum = np.c_[hdul[1].data["WAVE"][0],
                     hdul[1].data["FLUX"][0],
                     hdul[1].data["ERR"][0]]
    
    print(spectrum.shape[1])
    
    mask = (spectrum[:,0] < 9250.) & (spectrum[:,0] > 5250.)

    return bin(spectrum[mask], 2)



galaxy = pipes.galaxy("017433", load_vandels_spec, photometry_exists=False)

fig = galaxy.plot()