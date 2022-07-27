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

"""
hdu = fits.open("PSF_F444Wcen_G5V_fov299px_ISIM41.fits")
hdu.info()


plt.imshow(hdu[0].data)
plt.show()

cutout_image = Cutout2D(hdu[0].data, position=(600,600), size=80)

plt.imshow(cutout_image.data)
plt.show()

hdul = fits.PrimaryHDU(data=cutout_image.data)
hdul.writeto('psf_f444w.fits', overwrite=True)
"""

#create effective radius list to try
r_eff_list = np.array([15, 20, 25])
#create sersic index list to try
n_list = np.array([1, 2, 4])

PSF = fits.getdata("psf_f444w.fits")

# Normalize PSF 
PSF = PSF / PSF.sum()

# Note that the PSF shape is odd on all sides
print("PSF Shape = {}".format(PSF.shape))

# Plot PSF and use vmax and vmin to show difraction spikes
plt.imshow(PSF, vmin=0, vmax=5e-4)
plt.show()


petrosian_grid = generate_petrosian_sersic_correction(
    output_yaml_name='f444w_correction_grid.yaml',
    psf=PSF,
    r_eff_list=r_eff_list,
    n_list=n_list,
    oversample=('x_0', 'y_0', 10, 5),
    plot=False)



"""
pc = PetrosianCorrection('example_correction_gid.yaml')

corrected_epsilon = pc.estimate_epsilon(
    r_hl_pet=15, 
    c2080pet=3,  
    verbose=True
)

print(corrected_epsilon)
"""
