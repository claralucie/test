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

from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from photutils.detection import DAOStarFinder, find_peaks
#from detection import find_peaks

import os

#path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
#os.chdir(path)

path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722"
os.chdir(path)

def cutout(ra, dec, mos, size=5.):
    #wcs = WCS('WCSAXES')
    wcs = WCS(mos[0].header)
    wcs.sip = None
    
    if "CD1_1" in list(mos[0].header):
        cdelt = np.abs(mos[0].header["CD1_1"]*3600.)
        
    elif "CDELT1" in list(mos[0].header):
        cdelt = np.abs(mos[0].header["CDELT1"]*3600.)
        
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    
    cutout = Cutout2D(mos[0].data, coord, size/cdelt, wcs=wcs)
    
    return cutout

filters = ["F150W", "F200W", "F444W"]
#filters = ["F277W", "F410M", "F444W", "F356W"]
#1 
#c1000 
images1 =  ["jwst_ceers_first_nircam_f115w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f150w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f200w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f277w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f356w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f410m_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f444w_microJy_swarped.fits"]
"""
images1 = ["./jw01345-c1001_t021_nircam_clear-f115w/jw01345-c1001_t021_nircam_clear-f115w_i2d.fits",
           "./jw01345-c1001_t021_nircam_clear-f150w/jw01345-c1001_t021_nircam_clear-f150w_i2d.fits",
           "./jw01345-c1001_t021_nircam_clear-f200w/jw01345-c1001_t021_nircam_clear-f200w_i2d.fits",
           "./jw01345-c1001_t021_nircam_clear-f277w/jw01345-c1001_t021_nircam_clear-f277w_i2d.fits",
           "./jw01345-c1001_t021_nircam_clear-f356w/jw01345-c1001_t021_nircam_clear-f356w_i2d.fits",
           "./jw01345-c1001_t021_nircam_clear-f444w/jw01345-c1001_t021_nircam_clear-f444w_i2d.fits"]
"""
"""
images1 = ["./jw01345-c1000_t021_nircam_clear-f115w/jw01345-c1000_t021_nircam_clear-f115w_i2d.fits",
           "./jw01345-c1000_t021_nircam_clear-f150w/jw01345-c1000_t021_nircam_clear-f150w_i2d.fits",
           "./jw01345-c1000_t021_nircam_clear-f200w/jw01345-c1000_t021_nircam_clear-f200w_i2d.fits",
           "./jw01345-c1000_t021_nircam_clear-f277w/jw01345-c1000_t021_nircam_clear-f277w_i2d.fits",
           "./jw01345-c1000_t021_nircam_clear-f356w/jw01345-c1000_t021_nircam_clear-f356w_i2d.fits",
           "./jw01345-c1000_t021_nircam_clear-f444w/jw01345-c1000_t021_nircam_clear-f444w_i2d.fits"]
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
#24177 - 2
"""
images1 = ["./jw01345-o002_t022_nircam_clear-f115w/jw01345-o002_t022_nircam_clear-f115w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f200w/jw01345-o002_t022_nircam_clear-f200w_i2d.fits",
           "./jw01345-o002_t022_nircam_clear-f444w/jw01345-o002_t022_nircam_clear-f444w_i2d.fits",]
"""
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

#ID 10084 214.89284, 52.83085 14:19:34.2816, 52:49:51.06
#long swirl 214.9388125, 52.9543444 // 214.938617, 52.95426
#bright spiral 214.981708333, 52.99869444   no it's not ID 30405 redshift z=1.17954
#circ spiral 214.906433333, 52.919169444    ID 25668 redshift z=2.3042
#large 215.000308333, 52.988558333
#double 214.915158333, 52.9189638889
#very red = 214.988629167, 52.988558333
#c1001 = 214.992675, 52.9917917
#c1000 = 214.991196, 52.9860944
#swirly boy = (214.905408, 52.896125) #14:19:37.29792, 52:53:46.05
#pos_LONG = (214.970696, 52.9617528)
#pos_long2 = (214.911546, 52.914025)
#position_24177 = (214.86608, 52.88423) #14:19:27.8592, 52:53:3.228 // 21/07/22 214.866033, 52.8842528
#position-14727 = (214.89556, 52.85652)
#position-28830 = (214.76063, 52.84534)

#redshifts>6
#21748 214.77712, 52.80878
#25501 214.86305, 52.88945
#25525 214.97566, 52.96773
#28756 214.81807, 52.88303
#28872 214.7954, 52.86873
#28955 214.81987, 52.88488 z = 6.86636
#29463 214.8956, 52.93735

#redshift>7
#21980 214.86776, 52.87445 z=7.66409
#22267 214.98823, 52.95353
#25727 214.79395, 52.84155


ra = 215.039074 #98943    #14:19:39.4836
dec = 53.002678    #98838    #52:56:34.9188
size = 2

all_axes = []

axes = plt.subplot(1, 1, 1)


mos = fits.open(images1[1])
cut = cutout(ra, dec, mos, size)
#wcs = WCS(mos[1].header)

axes.imshow(np.flipud(cut.data), cmap='binary_r',
            norm = Normalize(vmin=np.percentile(cut.data, 0.5),
                             vmax=np.percentile(cut.data, 99.9))) #can change autocut here

plt.show()
plt.close()

filters = ['115', '150', '200', '277', '356', '410', '444']

hdu = fits.PrimaryHDU(data=cut.data, header=mos[0].header)
hdu.header.update(cut.wcs.to_header())

hdu.writeto('passive3_small_150.fits', overwrite=True)

#----------------------------------------------------------------------
#w=cut.wcs
#daofind = DAOStarFinder #(fwhm=3.0, threshold=5. * std)
"""
peak = find_peaks(hdu.data, 1.1)
print(peak)
x, y = peak["x_peak"], peak["y_peak"]

#skycoords = pixel_to_skycoord(x, y, wcs=cut.wcs)
#print(skycoords)

w = cut.wcs
sky = w.pixel_to_world(x, y)
print(sky)
"""