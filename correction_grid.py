#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:29:31 2022

@author: s1929920
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from petrofit.correction import generate_petrosian_sersic_correction
from petrofit.petrosian import PetrosianCorrection

path = "/home/s1929920/jwst/psf_lw"
os.chdir(path)


images0 = ["PSF_F150Wcen_G5V_fov299px_ISIM41.fits","PSF_F200Wcen_G5V_fov299px_ISIM41.fits",
           "PSF_F277Wcen_G5V_fov299px_ISIM41.fits", "PSF_F356Wcen_G5V_fov299px_ISIM41.fits",
           "PSF_F410Mcen_G5V_fov299px_ISIM41.fits", "PSF_F444Wcen_G5V_fov299px_ISIM41.fits"]

#for i in range(len(images0)):
#    hdu = fits.open(images0[i])
    
#hdu.info()   


image = fits.open("PSF_F444Wcen_G5V_fov299px_ISIM41.fits")
image.info()

cutout_image = Cutout2D(image[0].data, position=(600,600), size=160)

plt.imshow(cutout_image.data)
plt.show()

hdul = fits.PrimaryHDU(data=cutout_image.data)
hdul.writeto('psf_f444w.fits', overwrite=True)
"""
def get_psf_data(image):
    #create effective radius list to try
    r_eff_list = np.array([15, 20, 25])
    #create sersic index list to try
    n_list = np.array([1, 2, 4])
    
    PSF = fits.getdata(image)
    
    return r_eff_list, n_list, PSF

#images1 = ["psf_f200w.fits", "psf_f444w.fits"]
"""
#for i in range(len(images1)):
#    PSF = get_psf_data(images1[i])[2]

# Normalize PSF 
PSF = (image[1].data)
PSF = PSF / PSF.sum()
"""
# Note that the PSF shape is odd on all sides
print("PSF Shape = {}".format(PSF.shape))

# Plot PSF and use vmax and vmin to show difraction spikes
plt.imshow(PSF, vmin=0, vmax=5e-4)
plt.show()
"""
r_eff_list = np.array([15, 20, 25])
n_list = np.array([1, 2, 4])

petrosian_grid = generate_petrosian_sersic_correction(
    output_yaml_name='f444w_correction_grid.yaml',
    psf=PSF,
    r_eff_list=r_eff_list,
    n_list=n_list,
    oversample=('x_0', 'y_0', 10, 5),
    plot=False,

)
"""
petrosian_grid = generate_petrosian_sersic_correction(
    output_yaml_name='f150w_correction_grid.yaml',
    psf=PSF,
    r_eff_list = get_psf_data[0],
    n_list = get_psf_data[1],
    oversample=('x_0', 'y_0', 10, 5),
    plot=False)
"""


"""
pc = PetrosianCorrection('example_correction_gid.yaml')

corrected_epsilon = pc.estimate_epsilon(
    r_hl_pet=15, 
    c2080pet=3,  
    verbose=True
)

print(corrected_epsilon)
"""
