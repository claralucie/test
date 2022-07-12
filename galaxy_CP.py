#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4/07/2022 
Clara Pollock
GALAXY PLOTTING
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

#----------------------------------photometry----------------------------------------------
def load_phot_data(ID):
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

#test printing loaded data
print(load_phot_data(17433))

#plot galaxy photometric data
#fig = model.sfh.plot() #star formation history
#galaxy = pipes.galaxy(17433, load_data = load_phot_data, filt_list=filt_list, spectrum_exists=False)
#fig = galaxy.plot()

#------------------------------------spectroscopy---------------------------------------------------
def bin(spectrum, binn):
    """ Bins up two or three column spectral data by a specified factor. """

    binn = int(binn)
    nbins = len(spectrum)/binn
    binspec = (np.zeros((int(nbins), spectrum.shape[1]))) #have to specify nbins is an integer

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 1] = np.mean(spec_slice[:, 1])

        if spectrum.shape[1] == 3:
            binspec[i,2] = (1./float(binn)*np.sqrt(np.sum(spec_slice[:, 2]**2)))
    
    return binspec

def load_vandels_spec(ID):
    """ Loads VANDELS spectroscopic data from file. """

    hdul = fits.open('vandels17433.fits') #open as fits file
    
    spectrum = np.c_[hdul[1].data["WAVE"][0], #adds data from columns 'WAVE','FLUX','ERR' into one array
                     hdul[1].data["FLUX"][0],
                     hdul[1].data["ERR"][0]]
    
    mask = (spectrum[:,0] < 9250.) & (spectrum[:,0] > 5250.) #create mask for relevant wavelength range 

    return bin(spectrum[mask], 2)


#plot galaxy spectroscopic data

#galaxy = pipes.galaxy("017433", load_vandels_spec, photometry_exists=False)
#fig = galaxy.plot()

#-------------------------------------both----------------------------------------------------------
def load_both(ID):
    """Function to load data for both photometry and spectroscopy """
    spectrum = load_vandels_spec(ID)
    phot = load_phot_data(ID)

    return spectrum, phot

#plot both photometric and spectroscopic data
galaxy = pipes.galaxy("017433", load_both, filt_list=filt_list)
fig = galaxy.plot()