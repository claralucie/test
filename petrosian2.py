#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:57:15 2022

@author: s1929920
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.nddata import CCDData

"""
def one_over_eta(rs, fs, R):
    return ((np.pi*R**2)/np.sum(fs[rs<=R]))*(fs[rs==R])

def petrosian_radius(fs, rs, R):
    r_candidates = np.array([one_over_eta(rs, fs, R) for R in rs])
    return rs[np.square(r_candidates-0.2).argmin()]

def petrosian_flux(fs, rs, R_p):
    return np.sum(fs[rs<=2*R_p])

def R_x(rs, fs, x):
    x /=100
    sum_ratio = np.cumsum(fs)/np.sum(fs)
    return rs[np.square(sum_ratio-x).argmin()]

def petrosian_Re(R_50, R_90):
    P_3 = 8e-6
    P_4 = 8.47
    return R_50 / (1 - P_3*(R_90/R_50)**P_4)
"""

path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722"
os.chdir(path)

image = CCDData.read("24177_444.fits", unit="deg")
#image = fits.open("bright_spiral_444.fits")
#print(image[0].data)
#plt.imshow(image[0].data)
#wcs = WCS(image[0].header)

vmax = image.data.std() # Use the image std as max and min of all plots
vmin = - vmax

noise_cutout_pos = (130,130)
noise_cutout_size = (30)
noise_cutout = Cutout2D(image, noise_cutout_pos, noise_cutout_size)

noise_mean = noise_cutout.data.mean()
noise_sigma = noise_cutout.data.std()
noise_3_sigma = noise_sigma * 3.
noise_8_sigma = noise_sigma * 8.


plt.imshow(noise_cutout.data, vmax=noise_mean+noise_3_sigma, vmin=noise_mean-noise_3_sigma)
plt.title("Dark Patch")
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.show()

n, bins, patches = plt.hist(noise_cutout.data.flatten(), bins=35, align='left', color='black')
plt.plot(bins[:-1], n, c='r', linewidth=3)
plt.axvline(noise_mean, label="noise_mean", linestyle="--")

plt.xlabel('Flux Bins [{}]'.format(str(image.unit)))
plt.ylabel('Count')
plt.title('Noise Histogram')
plt.legend()

plt.show()

# Define detect threshold
threshold = noise_8_sigma

# Define smoothing kernel
kernel_size = 3
fwhm = 3
npixels = 4**2

#sigma = fwhm * gaussian_fwhm_to_sigma
#kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)

from petrofit.segmentation import make_catalog, plot_segments

cat, segm, segm_deblend = make_catalog(
    image.data,
    threshold,
    deblend=True,
    kernel_size=kernel_size,
    fwhm=fwhm,
    npixels=npixels,
    plot=True, vmax=vmax, vmin=vmin
)
#print("Number of targets: ", len(cat))

#plot_segments(segm_deblend, image=image.data, vmax=vmax, vmin=vmin)

from petrofit.segmentation import plot_segment_residual
plot_segment_residual(segm, image.data, vmax=vmax/5)

#----------------------------------PETROSIAN---------------------------------
from petrofit.photometry import order_cat

# Sort and get the largest object in the catalog
sorted_idx_list = order_cat(cat, key='area', reverse=True)
idx = sorted_idx_list[1] # index 0 is largest
source = cat[idx]  # get source from the catalog

from petrofit.photometry import make_radius_list

r_list = make_radius_list(
    max_pix=50, # Max pixel to go up to
    n=50 # the number of radii to produce
)

from petrofit.photometry import source_photometry

# Photomerty
flux_arr, area_arr, error_arr = source_photometry(

    # Inputs
    source, # Source (`photutils.segmentation.catalog.SourceCatalog`)
    image.data, # Image as 2D array
    segm_deblend, # Deblended segmentation map of image
    r_list, # list of aperture radii

    # Options
    cutout_size=max(r_list)*2, # Cutout out size, set to double the max radius
    bkg_sub=True, # Subtract background
    sigma=3, sigma_type='clip', # Fit a 2D plane to pixels within 3 sigma of the mean
    plot=True, vmax=vmax, vmin=vmin, # Show plot with max and min defined above
)
plt.show()

from petrofit.petrosian import Petrosian

p = Petrosian(r_list, area_arr, flux_arr)

"""petrosian radius"""
#print(p.r_petrosian) # in pixels

"""petrosian total flux radius"""
#print(p.r_total_flux) # pixels
#print(p.r_total_flux_arcsec(image.wcs)) #arcsec

"""petrosian half light radius"""
#print(p.r_half_light) #pixels
#print(p.r_half_light_arcsec(image.wcs)) #arcsec

"""fraction of flux radius"""
#print(p.fraction_flux_to_r(fraction=0.6)) #pixels 

#r_20, r_80, c2080 = p.concentration_index()
#print(r_20, r_80, c2080) #radii in pixels

"""
r_50, r_90, c5090 = p.concentration_index(
    fraction_1=0.5,
    fraction_2=0.9
)
print(r_50, r_90, c5090) #radii in pixels
"""

from petrofit.segmentation import (get_source_position, get_source_elong, get_source_theta,
                                   get_source_ellip, get_amplitude_at_r)


position = get_source_position(source)
elong = get_source_elong(source)
theta = get_source_theta(source)

p.imshow(position=position, elong=elong, theta=theta, lw=1.25)

plt.imshow(image.data, vmax=vmax, vmin=vmin)

plt.legend()
plt.show()

"""find estimated n sersic index?"""


#-------------------------------multiple targets-----------------------------------------------
#photometry loop 
from petrofit.petrosian import PetrosianCorrection 

max_pix=35

r_list = make_radius_list(
    max_pix=max_pix, # Max pixel to go up to
    n=max_pix # the number of radii to produce
)


petrosian_properties = {}

for idx, source in enumerate(cat):

    # Photomerty
    flux_arr, area_arr, error_arr = source_photometry(

        # Inputs
        source, # Source (`photutils.segmentation.catalog.SourceCatalog`)
        image.data, # Image as 2D array
        segm_deblend, # Deblended segmentation map of image
        r_list, # list of aperture radii

        # Options
        cutout_size=max(r_list)*2, # Cutout out size, set to double the max radius
        bkg_sub=True, # Subtract background
        sigma=3, sigma_type='clip', # Fit a 2D plane to pixels within 3 sigma of the mean
        plot=False, vmax=vmax, vmin=vmin, # Show plot with max and min defined above
    )
    plt.show()

    p = Petrosian(r_list, area_arr, flux_arr)

    petrosian_properties[source] = p

print("Completed for {} Sources".format(len(petrosian_properties)))

from photutils.isophote import EllipseGeometry, Ellipse
from astropy.modeling import models

# AstroPy Model List
model_list = []

# For each source
for source in list(petrosian_properties.keys()):

    # Get Petrosian
    p = petrosian_properties[source]

    # Estimate center
    position = get_source_position(source)
    x_0, y_0 = position

    # Estimate shape
    elong = get_source_elong(source)
    ellip = get_source_ellip(source)
    theta = get_source_theta(source)

    # Estimate Sersic index
    n = 1

    # Estimate r_half_light
    r_eff = p.r_half_light

    # Estimate amplitude
    amplitude = get_amplitude_at_r(r_eff, image, x_0, y_0, ellip, theta)

    # Allow for 4 pixel center slack
    center_slack = 4

    # Make astropy model
    sersic_model = models.Sersic2D(

            amplitude=amplitude,
            r_eff=r_eff,
            n=n,
            x_0=x_0,
            y_0=y_0,
            ellip=ellip,
            theta=theta,

            bounds = {
                'amplitude': (0., None),
                'r_eff': (0, None),
                'n': (0, 10),
                'ellip': (0, 1),
                'theta': (-2*np.pi, 2*np.pi),
                'x_0': (x_0 - center_slack/2, x_0 + center_slack/2),
                'y_0': (y_0 - center_slack/2, y_0 + center_slack/2),
            },
    )

    # Add to model list
    model_list.append(sersic_model)

    # Over-plot Petrosian radii
    p.imshow(position=position, elong=elong, theta=theta, lw=1.25)

# Plot image of sources
plt.imshow(image.data, vmax=vmax, vmin=vmin)
plt.show()

#-----------------------------------------------------------------------------------
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