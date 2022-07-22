#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:16:34 2022

@author: s1929920
"""
from astropy.io import fits
import os
import aplpy
#from aplpy import make_rgb_image

#path = "/storage/teaching/SummerProjects2022/s1929920/MAST_2022-07-15T0707/JWST"
#os.chdir(path)

path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_200722"
os.chdir(path)


#hdul = fits.open('please_work.fits')
#hdul = fits.open('test_cutout.fits')
#hdul.info()

#fig = aplpy.FITSFigure('please_work.fits')

#gc = aplpy.FITSFigure('please_work3.fits')
#gc.show_colorscale()

#gc2 = aplpy.FITSFigure('24177_3.fits')
#gc2.show_colorscale()
cube = aplpy.make_rgb_cube(['28955_356.fits', '28955_277.fits', '28955_150.fits'], '28955_cube.fits')

#cube = aplpy.make_rgb_cube(['c1001_444.fits', 'c1001_356.fits', 'c1001_277.fits'], 'c1001_cube5.fits')
#cube = aplpy.make_rgb_cube(['24177_444.fits', '24177_356.fits', '24177_277.fits'], '24177_cube_stretch.fits')
#cube = aplpy.make_rgb_cube(['large_24177_150.fits', 'large_24177_200.fits', 'large_24177_444.fits'], 'large_24177_cube.fits')
#cube = aplpy.make_rgb_cube(['mystery_444.fits', 'mystery_200.fits', 'mystery_150.fits'], 'mystery_cube.fits')
#cube = aplpy.make_rgb_cube(['24177_444.fits', '24177_410.fits', '24177_277.fits'], '24177_cube3.fits')
#cube = aplpy.make_rgb_cube(['28830_150.fits', '28830_200.fits', '28830_444.fits'], '28830_cube.fits')
#cube = aplpy.make_rgb_cube(['14727_444.fits', '14727_200.fits', '14727_150.fits'], '14727_cube.fits')

#gc3 = aplpy.FITSFigure('24177_cube_2d.fits')
#gc3.show_colorscale()

im = aplpy.rgb.make_rgb_image('28955_cube.fits', '28955_rgb.png', stretch_r = 'linear', stretch_g = 'linear', stretch_b = 'linear')
#im = aplpy.rgb.make_rgb_image('24177_cube2.fits', '24177_rgb2.png', indices=(0, 1, 2))

#image = aplpy.make_rgb_image(['24177_cube.fits'], 'cube_image.png', vmin_r=0., pmax_g=90.)
#image = aplpy.make_rgb_image(['24177_1.fits', '24177_2.fits', '24177_3.fits'], 'cube_image.png')

#g = aplpy.FITSFigure('24177_cube_stretch_2d.fits')
#g.show_colorscale(stretch='power')

f = aplpy.FITSFigure('28955_rgb.png')
f.show_circles()
f.show_rgb()


"""
g = aplpy.FITSFigure('24177_rgb.png')
g.show_rgb()
g.add_colorbar()
"""