#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:30:41 2022

@author: s1929920
"""

import bagpipes as pipes
import numpy as np
import pandas
import os 
import pymultinest as pmn
import matplotlib.pyplot as plt

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

#-------------------------------phot data-------------------------------------------------------------
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
#-------------------------------------spec data--------------------------------------------------------
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
galaxy = pipes.galaxy("017433", load_vandels_spec, photometry_exists=False)
#fig = galaxy.plot()
#----------------------------------------load both----------------------------------------------------

def load_both(ID):
    """Function to load data for both photometry and spectroscopy """
    spectrum = load_vandels_spec(ID)
    phot = load_phot_data(ID)

    return spectrum, phot

#plot both photometric and spectroscopic data
#galaxy = pipes.galaxy(17433, load_both, filt_list=filt_list)
#fig = galaxy.plot()

#----------------------------fit dictionary ---------------------------------------------------------------

dblplaw = {}                        
dblplaw["tau"] = (0., 15.)            
dblplaw["alpha"] = (0.01, 1000.)
dblplaw["beta"] = (0.01, 1000.)
dblplaw["alpha_prior"] = "log_10"
dblplaw["beta_prior"] = "log_10"
dblplaw["massformed"] = (1., 15.)
dblplaw["metallicity"] = (0.1, 2.)
dblplaw["metallicity_prior"] = "log_10"

nebular = {}
nebular["logU"] = -3.

dust = {}
dust["type"] = "CF00"
dust["eta"] = 2.
dust["Av"] = (0., 2.0)
dust["n"] = (0.3, 2.5)
dust["n_prior"] = "Gaussian"
dust["n_prior_mu"] = 0.7
dust["n_prior_sigma"] = 0.3

fit_instructions = {}
fit_instructions["redshift"] = (0.75, 1.25)
fit_instructions["t_bc"] = 0.01
fit_instructions["redshift_prior"] = "Gaussian"
fit_instructions["redshift_prior_mu"] = 0.9
fit_instructions["redshift_prior_sigma"] = 0.05
fit_instructions["dblplaw"] = dblplaw 
fit_instructions["nebular"] = nebular
fit_instructions["dust"] = dust

fit_instructions["veldisp"] = (1., 1000.)   #km/s
fit_instructions["veldisp_prior"] = "log_10"

#---------------------------calibration dictionary------------------------------------------------------
calib = {}
calib["type"] = "polynomial_bayesian"

calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
calib["0_prior"] = "Gaussian"
calib["0_prior_mu"] = 1.0
calib["0_prior_sigma"] = 0.25

calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
calib["1_prior"] = "Gaussian"
calib["1_prior_mu"] = 0.
calib["1_prior_sigma"] = 0.25

calib["2"] = (-0.5, 0.5)
calib["2_prior"] = "Gaussian"
calib["2_prior_mu"] = 0.
calib["2_prior_sigma"] = 0.25

fit_instructions["calib"] = calib

#maximum-likelihood polynomial
mlpoly = {}
mlpoly["type"] = "polynomial_max_like"
mlpoly["order"] = 2

noise = {}
noise["type"] = "white_scaled"
noise["scaling"] = (1., 10.)
noise["scaling_prior"] = "log_10"
fit_instructions["noise"] = noise


#----------------------FIT--------------------------------------------
fit = pipes.fit(galaxy, fit_instructions)#, run="spectroscopy"
fit.fit(verbose=False)

#Figures
fig = fit.plot_spectrum_posterior(save=False, show=True)
fig = fit.plot_calibration(save=False, show=True)
fig = fit.plot_sfh_posterior(save=False, show=True)
fig = fit.plot_corner(save=False, show=True)

plt.show()
plt.close()