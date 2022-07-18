#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:41:42 2022
individual cutouts
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
from astropy.utils.data import download_file

from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import os

def cutout(ra, dec, mos, size=5.):
    #wcs = WCS('WCSAXES')
    wcs = WCS(mos[1].header)
    wcs.sip = None
    
    if "CD1_1" in list(mos[1].header):
        cdelt = np.abs(mos[1].header["CD1_1"]*3600.)
        
    elif "CDELT1" in list(mos[1].header):
        cdelt = np.abs(mos[1].header["CDELT1"]*3600.)
        
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    cutout = Cutout2D(mos[1].data, coord, size/cdelt, wcs=wcs)
    
    mos[1].header.update(cutout.wcs.to_header())
    cutout_filename = 'example_cutout2.fits'
    mos.writeto(cutout_filename, overwrite=True)
                
    return cutout.data

"""
def download_image_save_cutout(name, ra, dec, size):
    # Download the image
    filename = imread(name)

    # Load the image and the WCS
    hdu = fits.open(filename)
    wcs = WCS(hdu.header)

    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, ra=ra, dec=dec, size=size, wcs=wcs)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file
    cutout_filename = 'cutout_115.fits'
    hdu.writeto(cutout_filename, overwrite=True)
"""
  
#filters = ["F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]
#filters = ["F115W", "F277W", "F444W"]

#filter115 = ["F115W"]
filter150 = ["F150W"]
filter277 = ["F277W"]
#filter444 = ["F444W"]


path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
os.chdir(path)

#/localdisk/MAST_2022-07-15T0707/JWST
"""
images1 = ["./jw01345-o002_t022_nircam_clear-f150w/jw01345-o002_t022_nircam_clear-f150w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f277w/jw01345-o002_t022_nircam_clear-f277w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits"]
"""
#images115 = ["./jw01345-o002_t022_nircam_clear-f115w/jw01345-o002_t022_nircam_clear-f115w_i2d.fits"]

images150 = ["./jw01345-o002_t022_nircam_clear-f150w/jw01345-o002_t022_nircam_clear-f150w_i2d.fits"]

images277= ["./jw01345-o002_t022_nircam_clear-f277w/jw01345-o002_t022_nircam_clear-f277w_i2d.fits"]

#images444 = ["./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits"]


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
"""
images1 = ["./jw01345-o002_t022_nircam_clear-f115w/jw01345-o002_t022_nircam_clear-f115w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f150w/jw01345-o002_t022_nircam_clear-f150w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f200w/jw01345-o002_t022_nircam_clear-f200w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f277w/jw01345-o002_t022_nircam_clear-f277w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f356w/jw01345-o002_t022_nircam_clear-f356w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f410m/jw01345-o002_t022_nircam_clear-f410m_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits"]
"""
"""
images1 = ["./jw01345-o001_t021_nircam_clear-f115w/jw01345-o001_t021_nircam_clear-f115w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f150w/jw01345-o001_t021_nircam_clear-f150w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f200w/jw01345-o001_t021_nircam_clear-f200w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f277w/jw01345-o001_t021_nircam_clear-f277w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f356w/jw01345-o001_t021_nircam_clear-f356w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f410m/jw01345-o001_t021_nircam_clear-f410m_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f444w/jw01345-o001_t021_nircam_clear-f444w_i2d.fits"]
"""
#------------------------------------------------------------------------------------------------
"""
fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

all_axes = []

ra = 214.86608 #214.76063 #214.86608 #214.89566 #214.9887625
dec = 52.88423 #52.84534 #52.88423 #52.85652 #52.9905416667 
size = 5

j = 0

all_axes.append([plt.subplot(gs[j,p]) for p in range(3)])

for i in range(3):
    #mos = Table.read(images1[i]).to_pandas()
    mos = fits.open(images1[i])
    
    cut = cutout(ra, dec, mos, size=size)
    
    
    #plt.savefig("cutout_277.fits", bbox_inches="tight")
    
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

#plt.savefig("cutout_150.png", bbox_inches="tight")

plt.show
"""
#----------------------------------------------------------------------------

fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

all_axes = []

ra = 214.86608 #214.76063 #214.86608 #214.89566 #214.9887625
dec = 52.88423 #52.84534 #52.88423 #52.85652 #52.9905416667 
size = 4

j = 0

all_axes.append([plt.subplot(gs[j,p]) for p in range(1)])

for i in range(1):
    #mos = Table.read(images1[i]).to_pandas()
    mos = fits.open(images150[i])
    
    cut = cutout(ra, dec, mos, size=size)
    
    #plt.imshow(cut)
    
    all_axes[-1][i].imshow(np.flipud(cut), cmap="binary_r",
                           norm=Normalize(vmin=np.percentile(cut, 0.5),
                                          vmax=np.percentile(cut, 99.5)))
    plt.setp(all_axes[-1][i].get_xticklabels(), visible=False)
    plt.setp(all_axes[-1][i].get_yticklabels(), visible=False)
    
    if j==0:
        if i<5:
            all_axes[-1][i].set_title(filter150[i])
            
        else:
            all_axes[-1][i].set_title(filter150[i])
            
    all_axes[-1][i].set_xticks([])
    all_axes[-1][i].set_yticks([])


all_axes[-1][0].set_ylabel(str(size) + "$^\{prime\prime}$ x" + str(size) + "$^{\prime\prime}$")

#plt.savefig("cutout_277.fits", bbox_inches="tight")
#cutout.writeto('cutout_277.fits', overwrite=True)
#plt.show()
"""
#---------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

all_axes = []

ra = 214.86608 #214.76063 #214.86608 #214.89566 #214.9887625
dec = 52.88423 #52.84534 #52.88423 #52.85652 #52.9905416667 
size = 5

j = 0

all_axes.append([plt.subplot(gs[j,p]) for p in range(1)])

for i in range(1):
    #mos = Table.read(images1[i]).to_pandas()
    mos = fits.open(images444[i])
    
    cut = cutout(ra, dec, mos, size=size)
    
    all_axes[-1][i].imshow(np.flipud(cut), cmap="binary_r",
                           norm=Normalize(vmin=np.percentile(cut, 0.5),
                                          vmax=np.percentile(cut, 99.5)))
    plt.setp(all_axes[-1][i].get_xticklabels(), visible=False)
    plt.setp(all_axes[-1][i].get_yticklabels(), visible=False)
    
    if j==0:
        if i<5:
            all_axes[-1][i].set_title(filter444[i])
            
        else:
            all_axes[-1][i].set_title(filter444[i])
            
    all_axes[-1][i].set_xticks([])
    all_axes[-1][i].set_yticks([])


all_axes[-1][0].set_ylabel(str(size) + "$^\{prime\prime}$ x" + str(size) + "$^{\prime\prime}$")

#plt.savefig("cutout_444.fits", bbox_inches="tight")
mos.writeto('cutout_444.fits', overwrite=True)
#plt.show()

#---------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

all_axes = []

ra = 214.86608 #214.76063 #214.86608 #214.89566 #214.9887625
dec = 52.88423 #52.84534 #52.88423 #52.85652 #52.9905416667 
size = 5

j = 0

all_axes.append([plt.subplot(gs[j,p]) for p in range(1)])

for i in range(1):
    #mos = Table.read(images1[i]).to_pandas()
    mos = fits.open(images115[i])
    
    cut = cutout(ra, dec, mos, size=size)
    
    all_axes[-1][i].imshow(np.flipud(cut), cmap="binary_r",
                           norm=Normalize(vmin=np.percentile(cut, 0.5),
                                          vmax=np.percentile(cut, 99.5)))
    plt.setp(all_axes[-1][i].get_xticklabels(), visible=False)
    plt.setp(all_axes[-1][i].get_yticklabels(), visible=False)
    
    if j==0:
        if i<5:
            all_axes[-1][i].set_title(filter115[i])
            
        else:
            all_axes[-1][i].set_title(filter115[i])
            
    all_axes[-1][i].set_xticks([])
    all_axes[-1][i].set_yticks([])


all_axes[-1][0].set_ylabel(str(size) + "$^\{prime\prime}$ x" + str(size) + "$^{\prime\prime}$")
"""

#plt.savefig("cutout_444.fits", bbox_inches="tight")
#mos.writeto('cutout_115.fits', overwrite=True)
#plt.show()

#aplpy.make_rgb_image(['cutout_150.fits', 'cutout_277.fits', 'cutout_444.fits'], 'rgb_gal.png')
#f = aplpy.FITSFigure('rgb_gal.png')
#f.show_rgb()


#gc = aplpy.FITSFigure('cutout_444.fits')
#gc.show_colorscale()

