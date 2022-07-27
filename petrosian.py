#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:01:30 2022
petrosian attempt
@author: s1929920
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import CCDData
from photutils.background import Background2D
from astropy.stats import SigmaClip
from photutils import MedianBackground
from photutils.psf import BasicPSFPhotometry

from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import os

path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722"
os.chdir(path)
"""
def calc_bg(a):
    #Estimate the background level and rms.
    good = ~np.isnan(a)
    assert good.sum(), 'no good pixels!'
    # poor man's source detection...
    vmax = np.percentile(a[good], 80)
    c0 = a[good] < vmax
    temp = a[good][c0]
    bg = np.median(temp)
    # now find the rms in the background
    belowbg = temp[temp < bg]
    # remove lowest 2% to get rid of any outliers
    flo = np.percentile(belowbg, 2)
    belowbg = belowbg[belowbg > flo]
    rms = np.concatenate([belowbg, 2*bg - belowbg]).std()
    return bg, rms
"""

#image = CCDData.read('jwst_ceers_first_nircam_f277w_microJy_swarped.fits', unit='deg')
image = CCDData.read('bright_spiral_444.fits', unit='deg')


plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['image.origin'] = 'lower'

vmax = 0.05 # vmax for matplotlib imshow
vmin = - vmax

#plt.imshow(image.data, vmin=vmin, vmax=vmax)
#plt.title("F277W CEERS")
#plt.xlabel("Pixels")
#plt.ylabel("Pixels")
#plt.show()

#rms_image = calc_bg(image)

"""
rms = fits.getdata('.fits')

plt.imshow(rms)
plt.title("RMS Image")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.show()
"""
coverage_mask = np.zeros([161,161])

sigma_clip = SigmaClip(sigma=3.)#, maxiters=10)
#bkg = MedianBackground(sigma_clip)
bkg_estimator = MedianBackground()

bg = Background2D(image.data, 100, filter_size=(3,3), sigma_clip = sigma_clip, bkg_estimator = MedianBackground(), exclude_percentile=36)

#bg = Background2D(image, 100, coverage_mask=True, fill_value = 0, exclude_percentile=36) #, mask=None, coverage_mask=True, fill_value=0.0
#plt.imshow(bg.data)
#plt.show()

#wcs = WCS(image.header)
#wcs.sip = None

hdu = fits.PrimaryHDU(data=bg.data)#, header=image[0].header)
#hdu.header.update(bg.wcs.to_header())
hdu.writeto('bg_test.fits', overwrite=True)


# Make cutout image, centerd at (100, 100) pixels, 40 pixels in size
cutout_image = Cutout2D(image.data, position=(80,80), size=80)

# Make cutout rms, centerd at (100, 100) pixels, 40 pixels in size
cutout_rms = Cutout2D(bg.data, position=(80,80), size=80)

#hdul = fits.open('bright_spiral_444.fits')
#hdul.info()


# Plot cutouts
# ------------

plt.imshow(cutout_image.data, vmin=vmin, vmax=vmax)
plt.title("Cutout Galaxy")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.show()


plt.imshow(cutout_rms.data)
plt.title("Cutout RMS")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.show()

#----------------------------------------------------------------------------------------------
from astropy.modeling import models
from petrofit.modeling import get_default_sersic_bounds
from astropy.modeling.models import Sersic2D

"""
sersic_model = models.Sersic2D(

        amplitude=10, # Intensity at r_eff
        r_eff=1, # Effective or half-lilght radius
        n=4, # Sersic index
        x_0=20, # center of model in the x direction
        y_0=20, # center of model in the y direction
        ellip=0.1, # Ellipticity
        theta=0.0, # Rotation angle in radians, counterclockwise from the positive x-axis.

        bounds=get_default_sersic_bounds(), # Parameter bounds
)
"""

x,y = np.meshgrid(np.arange(80), np.arange(80))
sersic_mod = Sersic2D(amplitude = 80, r_eff = 2, n=1, x_0=40, y_0=40, ellip=0.3, theta=-0.5)

img = sersic_mod(x, y)
log_img = np.log10(img)

plt.figure()
plt.imshow(log_img, origin='lower', interpolation='nearest',
           vmin=-1, vmax=2)
plt.xlabel('x')
plt.ylabel('y')
cbar = plt.colorbar()
cbar.set_label('Log Brightness', rotation=270, labelpad=25)
cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
plt.show()

"""
import webbpsf

nc = webbpsf.NIRCam()
nc.filter =  'F444W'
psf = nc.calc_psf(oversample=4)     # returns an astropy.io.fits.HDUlist containing PSF and header
plt.imshow(psf[0].data)             # display it on screen yourself, or
#webbpsf.display_psf(psf)            # use this convenient function to make a nice log plot with labeled axes
     
#psf = nc.calc_psf(filter='F470N', oversample=4)    # this is just a shortcut for setting the filter, then computing a PSF
     
#nc.calc_psf("myPSF.fits", filter='F480M')
#-------------------
#psf_sersic_model = PSFConvolvedModel2D(sersic_mod, psf=PSF, oversample=4)
"""

from petrofit.segmentation import make_catalog, plot_segments
from astropy.stats import sigma_clipped_stats

# Sigma clipped stats
image_mean, image_median, image_stddev = sigma_clipped_stats(image.data, sigma=3)

cat, segm, segm_deblend = make_catalog(
    image=image.data,  # Input image
    threshold=image_stddev*3,  # Detection threshold
    deblend=True,  # Deblend sources?
    kernel_size=3,  # Smoothing kernel size in pixels
    fwhm=3,  # FWHM in pixels
    npixels=4**2,  # Minimum number of pixels that make up a source
    plot=True, vmax=vmax, vmin=vmin # Plotting params
)