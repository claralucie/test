#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:40:06 2022
MODEL GALAXY
@author: clarapollock
"""

import bagpipes as pipes
import numpy as np

from astropy.table import Table
from astropy.io import fits
import os

path = "/home/s1929920/jwst/clara_code"
os.chdir(path) #create a path for python to follow to find files - this is for Linux machines in comp lab

filt_list = ["CFHT_u.dat",   #list of paths to data files for filter curves (wavelength v transmission)
             "CFHT_z.dat",      #save in file laid out by path above
             "ACS_F606W.dat",
             "ACS_F814W.dat",
             "WFC3_F125W.dat",
             "WFC3_F140W.dat",
             "WFC3_F160W.dat",
             "IRAC_CH1.dat",
             "IRAC_CH2.dat",
             "IRAC_CH3.dat",
             "IRAC_CH4.dat"] #filter list

exp = {}                    #Tau model star formation history component
exp["age"] = 2.5            #Gyr
exp["tau"] = 0.5            #Gyr
exp["massformed"] = 10      #log_10(M*/M_solar)
exp["metallicity"] = 1      #Z/Z_oldstar


dust = {}                   #Dust component
dust["type"] = "Calzetti"   #Define shape of attenuation curve
dust["Av"] = 0.5            #magnitudes
dust["eta"] = 2             #extra dust for young stars: multiplies Av

nebular = {}                #nebular emission component
nebular["logU"] = -3        #log_10 ionization parameter

model_components = {}                   #model components dictionary
model_components["redshift"] = 1.0      #Observed redshift
model_components["exponential"] = exp
model_components["dust"] = dust
model_components["veldisp"] = 250       #velocity displacement in km/s
model_components["t_bc"] = 0.01
model_components["nebular"] = nebular

#create model galaxy
model = pipes.model_galaxy(model_components, filt_list)

model.sfh.plot() # Plot the star-formation history
model.plot() # Plot the output spectrum







