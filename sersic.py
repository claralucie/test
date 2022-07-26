#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:03:39 2022

@author: s1929920
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling.models import Sersic2D
from astropy.modeling.functional_models import Gaussian2D
import os

path = "/home/s1929920/galfit-example/EXAMPLE"
os.chdir(path)

galfit_test = fits.open("imgblock.fits")
#galfit_test.info()

img0 = galfit_test[0]
img1 = galfit_test[1]
img2 = galfit_test[2]
img3 = galfit_test[3]

#plt.figure()
#plt.imshow(img1.data)


x,y = np.meshgrid(np.arange(40), np.arange(40))

#mod = Gaussian2D(amplitude = 20, x_mean = 50, y_mean = 50, x_stddev = 0.208, y_stddev = 0.5, theta = 1.57) #x_stddev=0.8, y_stddev=1.3, theta=-1, cov_matrix=None)

mod = Sersic2D(amplitude = 10, r_eff = 1, n=4, x_0=20, y_0=20, ellip=0.1, theta=0)

img = mod(x, y)
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