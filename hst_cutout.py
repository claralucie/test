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
from astropy.utils.data import get_pkg_data_filename

from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from photutils.detection import DAOStarFinder, find_peaks
#from detection import find_peaks

import os

#path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_200722"
#os.chdir(path)
path = "/storage/teaching/SummerProjects2022/s1929920/hst"
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

#filters = ["F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]
filters = ["WFC3 F105W", "WFC3 F125W", "WFC3 F140W", "WFC3 F160W", "ACS F606W", "ACS F814W"]

images1 = ["egs_all_wfc3_ir_f105w_030mas_v1.9_drz.fits",
           "egs_all_wfc3_ir_f125w_030mas_v1.9_drz.fits",
           "egs_all_wfc3_ir_f140w_030mas_v1.9_drz.fits",
           "egs_all_wfc3_ir_f160w_030mas_v1.9_drz.fits",
           "egs_all_acs_wfc_f814w_030mas_v1.9_drz.fits",
           "egs_all_acs_wfc_f606w_030mas_v1.9_drz.fits"]
"""
images1 =  ["jwst_ceers_first_nircam_f115w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f150w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f200w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f277w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f356w_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f410m_microJy_swarped.fits",
            "jwst_ceers_first_nircam_f444w_microJy_swarped.fits"]
"""
fig = plt.figure(figsize=(10,20))
gs = mpl.gridspec.GridSpec(10,7, wspace= 0.05, hspace=0.2)

ra = 214.98881    #866033 
dec = 52.98858    #8842528 
size = 5

all_axes = []

axes = plt.subplot(1, 1, 1)
mos = fits.open(images1[3])
cut = cutout(ra, dec, mos, size)

#circle = plt.Circle((125, 125), 2.93779, color='red', fill=False)
#fig = plt.gcf()
#axes = fig.gca()

#axes.add_patch(circle)

axes.imshow(np.flipud(cut.data), cmap='binary_r',
            norm = Normalize(vmin=np.percentile(cut.data, 0.5),
            vmax=np.percentile(cut.data,99.9)))

plt.show()
plt.close()

hdu = fits.PrimaryHDU(data=cut.data, header=mos[0].header)
hdu.header.update(cut.wcs.to_header())
hdu.writeto('red_triangle_277.fits', overwrite=True)

#----------------------------------------------------------------------
