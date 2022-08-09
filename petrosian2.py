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

from photutils.isophote import EllipseGeometry, Ellipse
from astropy.modeling import models


from petrofit.modeling import plot_fit, PSFConvolvedModel2D, fit_model, model_to_image
from photutils.isophote import EllipseGeometry, Ellipse
from petrofit.petrosian import Petrosian, PetrosianCorrection
from petrofit.photometry import order_cat, make_radius_list, source_photometry
from petrofit.segmentation import (make_catalog, plot_segments, plot_segment_residual,
                                   get_source_position, get_source_elong, get_source_theta,
                                   get_source_ellip, get_amplitude_at_r)


path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722"
os.chdir(path)

#-------------------------------------IMPORT IMAGE ---------------------------------


image = CCDData.read("aug09_1_F150W.fits", unit="deg")
vmax = image.data.std() # Use the image std as max and min of all plots
vmin = - vmax


#---------------------------------NOISE // DARK PATCH----------------------------------------
def noise_cutout(image, pos, size):
    pos = (225,225)
    size = 50
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
cut = noise_cutout(image, (225,225), 50)
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
    idx = sorted_idx_list[0] # index 0 is largest/brightest
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
"""
#r_20, r_80, c2080 = p.concentration_index()
#print(r_20, r_80, c2080) #radii in pixels


r_50, r_90, c5090 = p.concentration_index(
    fraction_1=0.5,
    fraction_2=0.9
)
print(r_50, r_90, c5090) #radii in pixels
"""

#-------------------------------PLOTTING PETROSIAN APERTURES ON ALL TARGETS-----------------------

def pet_aperture():
    
    position = get_source_position(source)
    elong = get_source_elong(source)
    theta = get_source_theta(source)
    
    #plots one aperture on image
    p.imshow(position=position, elong=elong, theta=theta, lw=1.25)
    
    plt.imshow(image.data, vmax=vmax, vmin=vmin)
    
    plt.legend()
    plt.show()

large_aperture = pet_aperture()

#-------------------------------PLOTTING MULTIPLE APERTURES---------------------------------
#photometry loop 

def multiple_apertures():
    
    max_pix=35
    
    r_list = make_radius_list(max_pix=max_pix, n=max_pix) 
    #Max pixel to go up to and the number of radii to produce
    
    petrosian_properties = {}
    
    for idx, source in enumerate(cat):
    
        # Photometry
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
    return petrosian_properties

petrosian_properties = multiple_apertures()

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
    amplitude = get_amplitude_at_r(r_eff, image.data, x_0, y_0, ellip, theta)

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

    


#-----------------CORRECTION GRID // PETROSIAN GRAPH------------------------------------------------------------------

path = "/home/s1929920/jwst/psf_lw"
os.chdir(path)

pc = PetrosianCorrection("f150w_correction_grid.yaml")

estimated_n = pc.estimate_n(
    p.r_half_light,
    p.concentration_index()[-1]
)

#=========================values // epsilon, radii, theta, n //===================================
print("ESTIMATED SERSIC INDEX n: ", estimated_n)

estimated_epsilon = pc.estimate_epsilon(
    p.r_half_light,
    p.concentration_index()[-1]
)

print("estimated epsilon: ", estimated_epsilon)

p_corrected = Petrosian(
    p.r_list,
    p.area_list,
    p.flux_list,
    epsilon=estimated_epsilon,
)
p_corrected.plot(plot_r=True, plot_normalized_flux=True)
plt.show()
#print("Uncorrected Flux = {}".format(p.total_flux * image.unit))
#print("Corrected Flux = {}".format(p_corrected.total_flux * image.unit))


theta = get_source_theta(source)
print("theta: ", theta)

n = pc.estimate_n(p.r_half_light, p.concentration_index()[-1])
print("n: " , n)

r_eff = p_corrected.r_half_light
print("CORRECTED HALF LIGHT RADIUS/ EFFECTIVE RADIUS r_eff: " , r_eff)

#---------------------------------------fitting models-------------------------------------------------

compound_model = np.array(model_list).sum()

print(compound_model)


hdu = fits.open('PSF_F150Wcen_G5V_fov299px_ISIM41.fits')
PSF = hdu[0].data
PSF = PSF/PSF.sum()
print("PSF shape = {} ".format(PSF.shape))

#plt.imshow(PSF, vmin=0, vmax=5e-4)
#plt.show()

#print(PSF.std())

plt.imshow(PSF, vmin=0, vmax=PSF.std()/10)
plt.show()

psf_sersic_model = PSFConvolvedModel2D(compound_model, psf=PSF, oversample=4)

psf_sersic_model.fixed['psf_pa'] = True


fitted_model, _ = fit_model(image.data, psf_sersic_model,
                         maxiter=10000, epsilon=2, acc=1e-09) #epsilon = 1.4901161193847656e-08


# Make Model Image
# ----------------

# Set the size of the model image equal to the original image
full_fitted_image_size = image.data.shape[0]

# Center the model image at the center of the original image
# so the two images cover the same window
full_fitted_image_center = full_fitted_image_size // 2

# Generate a model image from the model
fitted_model_image = model_to_image(
    fitted_model,
    full_fitted_image_size,
)


fig, ax = plt.subplots(1, 3, figsize=[24, 12])

ax[0].imshow(image.data, vmin=vmin, vmax=vmax)
ax[0].set_title("Data")

ax[1].imshow(fitted_model_image, vmin=vmin, vmax=vmax)
ax[1].set_title("Model")

ax[2].imshow(image.data - fitted_model_image, vmin=vmin, vmax=vmax)
ax[2].set_title("Residual (Data - Model)")

plt.show()

# Plot Model Image
# ----------------

fig, ax = plt.subplots(2, 2, figsize=(15,15))

# Plot raw data
im = ax[0, 0].imshow(image.data, vmin=vmin, vmax=vmax)
ax[0, 0].set_title("JWST F150W Image")
ax[0, 0].set_xlabel("Pixels")
ax[0, 0].set_ylabel("Pixels")
#ax[0, 0].axis('off')

# Plot Petrosian radii
plt.sca(ax[0, 1])
for i, source in enumerate(petrosian_properties):
    p = petrosian_properties[source]

    position = get_source_position(source)
    x_0, y_0 = position

    elong = get_source_elong(source)
    ellip = get_source_ellip(source)
    theta = get_source_theta(source)

    p.imshow(position=position, elong=elong, theta=theta, lw=1.25)
    if i == 0:
        plt.legend()
ax[0, 1].imshow(image.data, vmin=vmin, vmax=vmax)
ax[0, 1].set_title("Petrosian Radii")
ax[0, 1].set_xlabel("Pixels")
ax[0, 1].set_ylabel("Pixels")
# ax[0, 1].axis('off')

# Plot Model Image
ax[1, 0].imshow(fitted_model_image, vmin=vmin, vmax=vmax)
ax[1, 0].set_title("Simultaneously Fitted Sersic Models")
ax[1, 0].set_xlabel("Pixels")
ax[1, 0].set_ylabel("Pixels")
# ax[1, 0].axis('off')

# Plot Residual
ax[1, 1].imshow(image.data - fitted_model_image, vmin=vmin, vmax=vmax)
ax[1, 1].set_title("Residual")
ax[1, 1].set_xlabel("Pixels")
ax[1, 1].set_ylabel("Pixels")
# ax[1, 1].axis('off')

plt.show()
