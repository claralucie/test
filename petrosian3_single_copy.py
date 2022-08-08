#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:57:16 2022

@author: s1929920
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.nddata import CCDData

from photutils.isophote import EllipseGeometry, Ellipse
from astropy.modeling import models


from petrofit.modeling import plot_fit, PSFConvolvedModel2D, fit_model, model_to_image
from photutils.isophote import EllipseGeometry, Ellipse
from petrofit.petrosian import Petrosian, PetrosianCorrection
from petrofit.photometry import order_cat, make_radius_list, source_photometry
from petrofit.segmentation import (make_catalog, plot_segments, plot_segment_residual,
                                   get_source_position, get_source_elong, get_source_theta,
                                   get_source_ellip, get_amplitude_at_r)


path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722/petrofit_cutouts"
os.chdir(path)


#-------------------------------------IMPORT IMAGE ---------------------------------
"""    
image = CCDData.read("10109_F444W.fits", unit="deg")
vmax = image.data.std() # Use the image std as max and min of all plots
vmin = - vmax
"""
path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722/petrofit_cutouts"
image_list = sorted(os.listdir(path))

#a = list(range(100, 110))



def read_image(filename):
    image = CCDData.read(filename, unit="deg")
    return image


image = read_image(image_list[1899])
vmax = image.data.std()
vmin = -vmax

"""
for a in range(85,90):
image = read_image(image_list[a])
vmax = image.data.std()
vmin = -vmax
"""
"""
image = np.zeros(10)
vmax = np.zeros(10)
vmin = np.zeros(10)

for i in range(10):
    image.insert(read_image(image))
    vmax.insert(image.data.std())
    vmin.insert(-vmax)
"""
#---------------------------------NOISE // DARK PATCH----------------------------------------
def noise_cutout(image, pos, size):
    #pos = (20,20)
    size = 20
    return Cutout2D(image, pos, size)


def plot_noise(noise_cutout):
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

    #NOISE HISTOGRAM 
    plt.xlabel('Flux Bins [{}]'.format(str(image.unit)))
    plt.ylabel('Count')
    plt.title('Noise Histogram')
    plt.legend()
    
    plt.show()
    
    return noise_8_sigma

#===========================plot noise cutout image and histogram=====================================
cut = noise_cutout(image, (80,80), 20)
noise_8_sigma = (plot_noise(cut))
# Define detect threshold
threshold = noise_8_sigma

# Define smoothing kernel 
kernel_size = 3
fwhm = 3
npixels = 4**2

#sigma = fwhm * gaussian_fwhm_to_sigma
#kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)

#-----------------------------------SEGMENTATION----------------------------------------------------

def segmentation(image):
    cat, segm, segm_deblend = make_catalog(image.data, threshold, deblend=True,
                                           kernel_size=kernel_size, fwhm=fwhm,
                                           npixels=npixels, plot=True, vmax=vmax, vmin=vmin)
    plot_segment_residual(segm, image.data, vmax=vmax/5)
    return cat, segm, segm_deblend

#plot_segments(segm_deblend, image=image.data, vmax=vmax, vmin=vmin) # I think this should separate the objects?

#===============================plot segmentation plots===============================================
seg = segmentation(image)
segm_deblend = seg[2]
cat = seg[0]

#----------------------------------SORT OBJECTS IN CATALOGUE---------------------------------
#largest object
def objects(cat):
    
    # Sort and get the largest object in the catalog
    sorted_idx_list = order_cat(cat, key='area', reverse=True)
    idx = sorted_idx_list[0] # index 0 is largest
    source = cat[idx]  # get source from the catalog
    
    #list of radii needed to construct apertures - needed for curve of growth
    r_list = make_radius_list(
        max_pix=50, # Max pixel to go up to
        n=50 )# the number of radii to produce
    return r_list, source

#===============================set values for radius list and source=============================================
rad_list = (objects(cat))[0]
source = objects(cat)[1]

#----------------------------------------CURVE OF GROWTH---------------------------------------------------
def curve_of_growth(segm_deblend, rad_list):
    
    # Photometry
    #Plots Image and Aperture radius, and Curve of Growth
    flux_arr, area_arr, error_arr = source_photometry(
    
        # Inputs
        source, # Source (`photutils.segmentation.catalog.SourceCatalog`)
        image.data, # Image as 2D array
        segm_deblend, # Deblended segmentation map of image
        rad_list, # list of aperture radii
    
        # Options
        cutout_size=max(rad_list)*2, # Cutout out size, set to double the max radius
        bkg_sub=True, # Subtract background
        sigma=3, sigma_type='clip', # Fit a 2D plane to pixels within 3 sigma of the mean
        plot=True, vmax=vmax, vmin=vmin) # Show plot with max and min defined above
    
    plt.show()

    p = Petrosian(rad_list, area_arr, flux_arr)
    return p

#================================== set p==========================================================
p = curve_of_growth(segm_deblend, rad_list)

#----------------------------VALUES // PETROSIAN RADIUS, FLUX ETC. -------------------------------

def petrosian_radius():
    return p.r_petrosian 

def total_flux_radius():
    return p.r_total_flux, p.r_total_flux_arcsec(image.wcs)

def half_light_radius():
    return p.r_half_light, p.r_half_light_arcsec(image.wcs)

def frac_flux_radius():
    return p.fraction_flux_to_r(fraction = 0.6)

print("HALF LIGHT RADIUS/ EFFECTIVE RADIUS: ", half_light_radius())

#----------------------------------CONCENTRATION INDICES--------------------------------------


#-------------------------------PLOTTING PETROSIAN APERTURES ON ALL TARGETS-----------------------


#-------------------------------PLOTTING MULTIPLE APERTURES---------------------------------

    
#-----------------CORRECTION GRID // PETROSIAN GRAPH------------------------------------------------------------------

path = "/home/s1929920/jwst/psf_lw"
os.chdir(path)

pc = PetrosianCorrection("f444w_correction_grid.yaml")

#=========================values // epsilon, radii, theta, n //===================================

def estimated_n():
    return pc.estimate_n(p.r_half_light, p.concentration_index()[-1])

#estimated_n = pc.estimate_n(
#    p.r_half_light,
#    p.concentration_index()[-1]
#)


print("ESTIMATED SERSIC INDEX n: ", estimated_n)

def estimated_epsilon():
    return pc.estimate_epsilon(p.r_half_light, p.concentration_index()[-1])

#estimated_epsilon = pc.estimate_epsilon(
#    p.r_half_light,
#    p.concentration_index()[-1]
#)
est_epsilon = estimated_epsilon()

print("estimated epsilon: ", estimated_epsilon)

p_corrected = Petrosian(
    p.r_list,
    p.area_list,
    p.flux_list,
    epsilon=est_epsilon,
)

p_corrected.plot(plot_r=True, plot_normalized_flux=True)
plt.show()
#print("Uncorrected Flux = {}".format(p.total_flux * image.unit))
#print("Corrected Flux = {}".format(p_corrected.total_flux * image.unit))


theta = get_source_theta(source)
print("theta: ", theta)

n = pc.estimate_n(p.r_half_light, p.concentration_index()[-1])
print("n: " , n)

def corrected_r_eff():
    return p_corrected.r_half_light

#r_eff = p_corrected.r_half_light
print("CORRECTED HALF LIGHT RADIUS/ EFFECTIVE RADIUS r_eff: " , corrected_r_eff())


path = "/home/s1929920/jwst"
os.chdir(path)

data = pd.read_csv("important_info_petro3.csv", usecols=["ID", "RA", "DEC", "redshift_50", "stellar_mass_50", "r_eff_150", "sersic_n_150", "r_eff_444", "sersic_n_444"])
df = pd.DataFrame(data)

   
"""   
df.loc[i, "r_eff_150"] = df2.iat[i,14].r_half_light
df.loc[i, "sersic_n_150" ]= df2.iat[i,15]
    
#df.loc[0, "sersic_n"] = p3.estimated_n()
df.to_csv('important_info_petro3_TEST.csv', header=True, index=None, sep=',', mode='w')
   """ 
"""
df.loc[2, "r_eff_444"] = corrected_r_eff()
df.loc[2, "sersic_n_444"] = estimated_n()
df.to_csv('important_info_petro3.csv', header=True, index=None, sep=',', mode='w')
"""
#save to csv file? ID, RA, DEC, REDSHIFT, MASS, R_EFF, SERSIC INDEX n, 

