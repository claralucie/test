#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:45:02 2022

@author: s1929920
"""
import os

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.nddata import CCDData
from petrofit.petrosian import Petrosian, PetrosianCorrection


#path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722/petrofit_cutouts"
#os.chdir(path)

#path = "/home/s1929920/jwst"
#os.chdir(path)
"""
data = pd.read_csv("petro_catalogue.csv", usecols = ["ID", "RA", "DEC", "redshift_50", "stellar_mass_50"])
df = pd.DataFrame(data)
df.to_csv('important_info_petro.csv', header=True, index=None, sep=',', mode='w')


r_eff = np.zeros(2703)
sersic_n = np.zeros(2703)

df['r_eff'] = r_eff
df['sersic_n'] = sersic_n

df.to_csv('important_info_petro.csv', header=True, index=None, sep=',', mode='w')

"""
"""
from petrosian3_single import (read_image, noise_cutout, plot_noise, segmentation, objects, curve_of_growth,
                        petrosian_radius, total_flux_radius, half_light_radius, frac_flux_radius,
                        estimated_n, estimated_epsilon, corrected_r_eff)
"""
import petrosian3_single as p3

#path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722/petrofit_cutouts"
#os.chdir(path)
#image_list = sorted(os.listdir(path))


#for i in range(0,10):
#    print(image_list[i])
path = "/home/s1929920/jwst"
os.chdir(path)

dfid = pd.read_csv('z3_catalogue.csv', usecols=["ID"])
ID = dfid["ID"]
image_list150 = []
image_list444 = []

for i in range(len(ID)):
    image_list150.append(str(ID[i]) + "_F150W.fits")
    image_list150.append(str(ID[i]) + "_F444W.fits")
  
print(len(image_list150))

path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722/petrofit_cutouts"
os.chdir(path)

df2 = pd.DataFrame(np.zeros((1056, 16)), columns = ['read', 'vmax', 'vmin', 'cut', 'plot', 'noise_8_sigma',
                                                 'segment', 'deblend', 'categ', 'objects1', 'radius_list',
                                                 'source1', 'curve', 'estim_epsilon', 'p_corrected_test',
                                                 'estim_n'])

#print(df2)

df2['read'] = df2['read'].astype(object)
df2['vmax'] = df2['vmax'].astype(object)
df2['vmin'] = df2['vmin'].astype(object)
df2['cut'] = df2['cut'].astype(object)
df2['plot'] = df2['plot'].astype(object)
df2['noise_8_sigma'] = df2['noise_8_sigma'].astype(object)
df2['segment'] = df2['segment'].astype(object)
df2['deblend'] = df2['deblend'].astype(object)
df2['categ'] = df2['categ'].astype(object)
df2['objects1'] = df2['objects1'].astype(object)
df2['radius_list'] = df2['radius_list'].astype(object)
df2['source1'] = df2['source1'].astype(object)
df2['curve'] = df2['curve'].astype(object)
df2['estim_epsilon'] = df2['estim_epsilon'].astype(object)
df2['p_corrected_test'] = df2['p_corrected_test'].astype(object)
df2['estim_n'] = df2['estim_n'].astype(object)
#print(df2)

"""
#file = np.zeros(10)
read = np.zeros((10))

#vmax1 = []
vmax1 = np.zeros(10)
vmin1= np.zeros(10)
#cut1 = np.zeros(10)
plot = np.zeros(10)
noise_8_sigma_1 = np.zeros(10)
threshold1 = np.zeros(10)
#segment = np.zeros([10,3])
deblend = np.zeros(10)
categ = np.zeros(10)
#objects1 = np.zeros([10,3])
#radius_list = np.zeros(10)
#source1 = np.zeros(10)
curve = np.zeros(10)
"""
path = "/home/s1929920/jwst"
os.chdir(path)
data = pd.read_csv("z3_catalogue_fit.csv", usecols=["ID", "RA", "DEC", "redshift_50", "stellar_mass_50",
                                                    "r_eff_150", "sersic_n_150", "r_eff_444", "sersic_n_444"])
df3 = pd.DataFrame(data)

r_eff = []

for i in range(1, 11, 2):
    
    path = "/storage/teaching/SummerProjects2022/s1929920/derek_ceers_210722/petrofit_cutouts"
    os.chdir(path)
        
    df2.iat[i,0] = p3.read_image(image_list150[i]) #read
    
    df2.iat[i,1] = (df2.iat[i,0]).data.std() #vmax
    df2.iat[i,2] = -(df2.iat[i,1]) #vmin
    #print(df2.iat[i,2])
    df2.iat[i,3] = p3.noise_cutout(df2.iat[i,0], (20,20), 20) #cut
    df2.iat[i,4] = p3.plot_noise(df2.iat[i,0], df2.iat[i,3], df2.iat[i,1], df2.iat[i,2]) #plot
    df2.iat[i,5] = df2.iat[i,4] #noise_8_sigma
        
    try:
        df2.iat[i,6] = (p3.segmentation(df2.iat[i,0], df2.iat[i,5], df2.iat[i,1], df2.iat[i,2])) #segment
    
    except ValueError:
        print("an error occurred")
    
    else:
        #print("segment:" , df2.iat[i,6])
        df2.iat[i,7] = df2.iat[i,6][2] #deblend
        df2.iat[i,8] = df2.iat[i,6][0] # categ
        df2.iat[i,9] = p3.objects(df2.iat[i,8]) #objects1
        #print("objects1: ", df2.iat[i,9][0])
        df2.iat[i,10] = df2.iat[i,9][0] #radius_list
        #print("radius_list: ", df2.iat[i,10])
        df2.iat[i,11] = df2.iat[i,9][1] #source1
        df2.iat[i,12] = p3.curve_of_growth(df2.iat[i,7], df2.iat[i,10], df2.iat[i,0], df2.iat[i,11], df2.iat[i,1], df2.iat[i,2]) #curve
        #df2.iat[i,7] = 
        """
        #read[i] = CCDData.read(image_list[i], unit="deg")
        #print(read.shape)
        read[i] = p3.read_image(image_list[i])
        #file[i]=(image_list[i])
        #read.insert(i, p3.read_image(image_list[i]))
        #read.append(p3.read_image(image_list[i]))
        vmax1[i] = (read.data.std())
        vmin1[i] = -vmax1[i] 
        cut1 = p3.noise_cutout(read, (20,80), 20)
        plot[i] = p3.plot_noise(read, cut1, vmax1[i], vmin1[i])
        noise_8_sigma_1[i] = plot[i]
        threshold1[i] = noise_8_sigma_1[i]
        
        segment=(p3.segmentation(read, threshold1[i], vmax1[i], vmin1[i]))
        deblend = segment[2]
        categ = segment[0]
        objects1 = p3.objects(categ )
        radius_list = objects1[0]
        source1 = objects1[1]
        curve = p3.curve_of_growth(deblend, radius_list, read, source1, vmax1[i], vmin1[i])
        """
        path = "/home/s1929920/jwst/psf_lw"
        os.chdir(path)
        
        pcorr = PetrosianCorrection("f444w_correction_grid.yaml")
        
        path = "/home/s1929920/jwst"
        os.chdir(path)
        
        #estim_epsilon
        df2.iat[i,13] = pcorr.estimate_epsilon((df2.iat[i,12]).r_half_light, (df2.iat[i,12]).concentration_index()[-1])
        
        #p_corrected_test
        df2.iat[i,14] = Petrosian((df2.iat[i,12]).r_list,
                                  (df2.iat[i,12]).area_list,
                                  (df2.iat[i,12]).flux_list,
                                  epsilon = df2.iat[i,13])
        
        df2.iat[i,14].plot(plot_r=True, plot_normalized_flux=True)
        plt.show()
        
        #estim_n
        df2.iat[i,15] = pcorr.estimate_n((df2.iat[i,12]).r_half_light, (df2.iat[i,12]).concentration_index()[-1])
        #estimated_n = pc.estimate_n(
        #p.r_half_light,
        #p.concentration_index()[-1]
        
        print("testing new: index is ", i, "effective radius is ", df2.iat[i,14].r_half_light)
        
        
        r_eff.append(df2.iat[i,14].r_half_light)
        
        #print(r_eff)
        
        
        #df.loc[(i-1)/2, "r_eff_444"] = df2.iat[i,14].r_half_light
        #df.loc[(i-1)/2, "sersic_n_444"] = df2.iat[i, 15]
        df3.loc[(i-1)/2, "r_eff_444"] = df2.iat[i,14].r_half_light
        df3.loc[(i-1)/2, "sersic_n_444"]= df2.iat[i,15]
        
        #df.loc[0, "sersic_n"] = p3.estimated_n()
        #df3.to_csv('z3_catalogue_fit.csv', header=True, index=None, sep=',', mode='w')
  

#path = "/home/s1929920/jwst"
#os.chdir(path)
        

        
