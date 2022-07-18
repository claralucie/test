#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:22:11 2022

@author: s1929920
"""

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.io import fits
from astropy.wcs import WCS

from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import wcs_to_celestial_frame, custom_wcs_to_frame_mappings
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from astropy.units import u

import os

    
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

#filters = ["F200W", "F440W"]
filters = ["F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]

#observation 52

#path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
#path = "/localdisk/MAST_2022-07-15T0707/JWST"
path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
os.chdir(path)

#/localdisk/MAST_2022-07-15T0707/JWST
"""
images1 = ["./jw01345-o052_t022_nircam_clear-f200w/jw01345-o052_t022_nircam_clear-f200w_i2d.fits",
           "./jw01345-o052_t022_nircam_clear-f444w/jw01345-o052_t022_nircam_clear-f444w_i2d.fits"]
"""
"""
images1 = ["./jw01345-o004_t024_nircam_clear-f115w/jw01345-o004_t024_nircam_clear-f115w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f150w/jw01345-o004_t024_nircam_clear-f150w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f200w/jw01345-o004_t024_nircam_clear-f200w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f277w/jw01345-o004_t024_nircam_clear-f277w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f356w/jw01345-o004_t024_nircam_clear-f356w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f410m/jw01345-o004_t024_nircam_clear-f410m_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f444w/jw01345-o004_t024_nircam_clear-f444w_i2d.fits"]
"""
"""
images1 = ["./jw01345-o003_t023_nircam_clear-f115w/jw01345-o003_t023_nircam_clear-f115w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f150w/jw01345-o003_t023_nircam_clear-f150w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f200w/jw01345-o003_t023_nircam_clear-f200w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f277w/jw01345-o003_t023_nircam_clear-f277w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f356w/jw01345-o003_t023_nircam_clear-f356w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f410m/jw01345-o003_t023_nircam_clear-f410m_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f444w/jw01345-o003_t023_nircam_clear-f444w_i2d.fits"]
"""

images1 = ["./jw01345-o002_t022_nircam_clear-f115w/jw01345-o002_t022_nircam_clear-f115w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f150w/jw01345-o002_t022_nircam_clear-f150w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f200w/jw01345-o002_t022_nircam_clear-f200w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f277w/jw01345-o002_t022_nircam_clear-f277w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f356w/jw01345-o002_t022_nircam_clear-f356w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f410m/jw01345-o002_t022_nircam_clear-f410m_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits"]

"""
images1 = ["./jw01345-o001_t021_nircam_clear-f115w/jw01345-o001_t021_nircam_clear-f115w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f150w/jw01345-o001_t021_nircam_clear-f150w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f200w/jw01345-o001_t021_nircam_clear-f200w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f277w/jw01345-o001_t021_nircam_clear-f277w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f356w/jw01345-o001_t021_nircam_clear-f356w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f410m/jw01345-o001_t021_nircam_clear-f410m_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f444w/jw01345-o001_t021_nircam_clear-f444w_i2d.fits"]
"""
#images2 = [".jw01345-o002_t022_nircam_clear-f115w/jw01345-o002_t022_nircam_clear-f115w_i2d.fits",
           #]
#images52 = ["jw01345-o052_t022_nircam_clear-f200w_i2d.fits",
#          "jw01345-o052_t022_nircam_clear-f444w_i2d.fits"]

fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

all_axes = []

ra = 214.86608 #214.76063 #214.86608 #214.89566 #214.9887625
dec = 52.88423 #52.84534 #52.88423 #52.85652 #52.9905416667 
size = 4

j = 0

all_axes.append([plt.subplot(gs[j,p]) for p in range(7)])

#mos = []

for i in range(7):
    #mos = Table.read(images1[i]).to_pandas()
    mos = fits.open(images1[i])
    
    cut = cutout(ra, dec, mos, size=size)
    
    all_axes[-1][i].imshow(np.flipud(cut), cmap="binary_r",
                           norm=Normalize(vmin=np.percentile(cut, 0.5),
                                          vmax=np.percentile(cut, 99.5)))
    plt.setp(all_axes[-1][i].get_xticklabels(), visible=False)
    plt.setp(all_axes[-1][i].get_yticklabels(), visible=False)
    
    if j==0:
        if i<5:
            all_axes[-1][i].set_title(filters[i])
            
        else:
            all_axes[-1][i].set_title(filters[i])
            
    all_axes[-1][i].set_xticks([])
    all_axes[-1][i].set_yticks([])


all_axes[-1][0].set_ylabel(str(size) + "$^\{prime\prime}$ x" + str(size) + "$^{\prime\prime}$")

#plt.savefig("cutouts.pdf", bbox_inches="tight")
plt.show
#plt.close()

print(cutout.data)
#hdu = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())
#hdu.writeto('cropped_file.fits')

