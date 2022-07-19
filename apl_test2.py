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
from photutils.detection import DAOStarFinder, find_peaks
#from detection import find_peaks

import os

path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
os.chdir(path)

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
    
    return cutout

filters = ["F150W", "F200W", "F444W"]
#filters = ["F277W", "F410M", "F444W", "F356W"]
#1 
"""
images1 = ["./jw01345-o001_t021_nircam_clear-f115w/jw01345-o001_t021_nircam_clear-f115w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f150w/jw01345-o001_t021_nircam_clear-f150w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f200w/jw01345-o001_t021_nircam_clear-f200w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f277w/jw01345-o001_t021_nircam_clear-f277w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f356w/jw01345-o001_t021_nircam_clear-f356w_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f410m/jw01345-o001_t021_nircam_clear-f410m_i2d.fits",
           "./jw01345-o001_t021_nircam_clear-f444w/jw01345-o001_t021_nircam_clear-f444w_i2d.fits"]
"""
#24177 - 2

images1 = ["./jw01345-o002_t022_nircam_clear-f150w/jw01345-o002_t022_nircam_clear-f150w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f200w/jw01345-o002_t022_nircam_clear-f200w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits",]

"""
images1 = ["./jw01345-o002_t022_nircam_clear-f277w/jw01345-o002_t022_nircam_clear-f277w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f410m/jw01345-o002_t022_nircam_clear-f410m_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f356w/jw01345-o002_t022_nircam_clear-f356w_i2d.fits"]
"""
#14727 - 4
"""
images1 = ["./jw01345-o004_t024_nircam_clear-f150w/jw01345-o004_t024_nircam_clear-f150w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f200w/jw01345-o004_t024_nircam_clear-f200w_i2d.fits",
           "./jw01345-o004_t024_nircam_clear-f444w/jw01345-o004_t024_nircam_clear-f444w_i2d.fits"]
"""
#28830 - 3
"""
images1 = ["./jw01345-o003_t023_nircam_clear-f150w/jw01345-o003_t023_nircam_clear-f150w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f200w/jw01345-o003_t023_nircam_clear-f200w_i2d.fits",
           "./jw01345-o003_t023_nircam_clear-f444w/jw01345-o003_t023_nircam_clear-f444w_i2d.fits"]
"""

fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

#swirly boy = (214.905408, 52.896125) #14:19:37.29792, 52:53:46.05
#pos_LONG = (214.970696, 52.9617528)
#pos_long2 = (214.911546, 52.914025)
#position_24177 = (214.86608, 52.88423)
#position-14727 = (214.89556, 52.85652)
#position-28830 = (214.76063, 52.84534)
ra = 214.86608 #14 19 27.85908
dec = 52.88423 #52 53 32.28
size = 4

all_axes = []

axes = plt.subplot(1, 1, 1)
mos = fits.open(images1[0])
cut = cutout(ra, dec, mos, size)
#wcs = WCS(mos[1].header)

axes.imshow(np.flipud(cut.data), cmap='binary_r',
            norm = Normalize(vmin=np.percentile(cut.data, 0.5),
            vmax=np.percentile(cut.data,99.0)))

plt.show()
plt.close()

hdu = fits.PrimaryHDU(data=cut.data, header=mos[1].header)
hdu.header.update(cut.wcs.to_header())
hdu.writeto('24177_150.fits', overwrite=True)

#----------------------------------------------------------------------


#daofind = DAOStarFinder #(fwhm=3.0, threshold=5. * std)

peak = find_peaks(hdu.data, 1.1)
print(peak)
#x, y = peak["x_peak"], peak["y_peak"]

w = WCS(mos[1].header)
sky = hdu.pixel_to_skycoord(67, 54)
print(sky)