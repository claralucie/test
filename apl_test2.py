#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:29:42 2022

@author: s1929920
"""
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import imread
import aplpy

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import os

path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
os.chdir(path)

def cutout(ra, dec, mos, size=5.):
    #wcs = WCS('WCSAXES')
    wcs = WCS(mos[1].header)
    wcs.sip = None
    
    if "CD1_1" in list(mos[1].header):
        cdelt = np.abs(mos[1].header["CD1_1"]*5000.)
        
    elif "CDELT1" in list(mos[1].header):
        cdelt = np.abs(mos[1].header["CDELT1"]*5000.)
        
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    cutout = Cutout2D(mos[1].data, coord, size/cdelt, wcs=wcs)
    
    return cutout.data


fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

all_axes = []

ra = 214.86608 #214.76063 #214.86608 #214.89566 #214.9887625
dec = 52.88423 #52.84534 #52.88423 #52.85652 #52.9905416667 
size = 8



#mos = fits.open('cutout_150.fits')
#cut = cutout(ra, dec, mos, size=size)
    
#plt.imshow(cutout.data)

#fig = aplpy.FITSFigure('cutout_150.fits')
#fig.show_greyscale()

#hdul = fits.open('example_cutout2.fits')
#hdul.info()

"""
co_cube = "cutout_150.fits"
#co_cube = "./jw01345-o002_t022_nircam_clear-f150w/jw01345-o002_t022_nircam_clear-f150w_i2d.fits"
f = aplpy.FITSFigure(co_cube, slices = [30], figsize = (8,6))
f.show_colorscale()
"""