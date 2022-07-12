#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:14:30 2022
FITTING DATA
@author: s1929920
"""
import numpy as np
import bagpipes as pipes
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pymultinest as pmn
    
from astropy.table import Table
from astropy.io import fits


path = "/home/s1929920/jwst/clara_code" #creating path for python to follow to find files
os.chdir(path)

def load_data3(ID):
    """
    Function to load in data as a .fits file
    
    Parameter ID: ID number of galaxy, corresponds to row number on .fits file
    Returns photometry: n x 2 matrix containing fluxes and flux errors
    
    Then used to plot galaxy
    """
    #open category using pandas
    cat = Table.read("candels1.fits").to_pandas()
    
    #set index to ID values
    cat.index = cat["ID"].values
    
    #define row numbers
    row = int(ID) - 1
    
    #create list of fluxes and their associated errors that we want to see
    #must be in same order as filt_list
    flux_list = ["CFHT_u_FLUX", "CFHT_z_FLUX", "ACS_F606W_FLUX", "ACS_F814W_FLUX", "WFC3_F125W_FLUX", "WFC3_F140W_FLUX", "WFC3_F160W_FLUX", "IRAC_CH1_FLUXERR", "IRAC_CH2_FLUXERR", "IRAC_CH3_FLUX", "IRAC_CH4_FLUX"]
    fluxerr_list = ["CFHT_u_FLUXERR", "CFHT_z_FLUXERR", "ACS_F606W_FLUXERR", "ACS_F814W_FLUXERR", "WFC3_F125W_FLUXERR", "WFC3_F140W_FLUXERR", "WFC3_F160W_FLUXERR", "IRAC_CH1_FLUXERR", "IRAC_CH2_FLUXERR", "IRAC_CH3_FLUXERR", "IRAC_CH4_FLUXERR"]

    #select all rows and specific columns of data from table using lists above. Write all as floats 
    fluxes = (cat.loc[row, flux_list]).astype(float)
    fluxerrs = (cat.loc[row, fluxerr_list]).astype(float)
    
    #translate slice objects to  contatenation along second axis
    photometry = np.c_[fluxes, fluxerrs]
    
    # blow up the errors associated with any missing fluxes.
    for i in range(len(photometry)):
        if (photometry[i, 0] == 0.) or (photometry[i, 1] <= 0):
            photometry[i,:] = [0., 9.9*10**99.]
            
    # Enforce a maximum SNR of 20, or 10 in the IRAC channels.
    for i in range(len(photometry)):
        if i < 10:
            max_snr = 20.
            
        else:
            max_snr = 10.
        
        if photometry[i, 0]/photometry[i, 1] > max_snr:
            photometry[i, 1] = photometry[i, 0]/max_snr
            
    return photometry

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

#plot galaxy photometric data
galaxy = pipes.galaxy(2222, load_data = load_data3, filt_list=filt_list, spectrum_exists=False)
#fig = galaxy.plot()

exp = {}                            #Tau model star formation history component
exp["age"] = (0.1, 15)              #Gyr
exp["tau"] = (0.3, 10.)             #Gyr
exp["massformed"] = (1., 15.)       #log_10(M*/M_solar)
exp["metallicity"] = (0., 2.5)              #Z/Z_oldstar

dust = {}                           #Dust component
dust["type"] = "Calzetti"   #       Define shape of attenuation curve
dust["Av"] = (0., 2.)               #magnitudes

fit_instructions = {}                       #fit instructions dictionary
fit_instructions["redshift"] = (0., 10.)    #vary observed redshift from 0 to 10
fit_instructions["exponential"] = exp
fit_instructions["dust"] = dust

fit = pipes.fit(galaxy, fit_instructions)
fit.fit(verbose=False)

fig = fit.plot_spectrum_posterior(save=False, show=True)
fig = fit.plot_sfh_posterior(save=False, show=True)
fig = fit.plot_corner(save=False, show=True)

plt.show()
plt.close()