#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:25:25 2022
MIRI currently commented out
@author: clarapollock
"""

import pysiaf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch                        
import matplotlib.image as mpimg
from matplotlib.collections import PatchCollection
from copy import deepcopy

import time
from astropy.table import Table
import pandas as pd

def plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa, color="red"):
    nrc_siaf = pysiaf.Siaf('NIRCam')
    miri_siaf = pysiaf.Siaf('MIRI')

    attmat = pysiaf.utils.rotations.attitude_matrix(v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

    #full arrays for nircam lw
    aperfill = ['NRCB5_FULL','NRCA5_FULL']
    nircam_poly = []

    for i in aperfill:
        lw = nrc_siaf[i]
        lw.set_attitude_matrix(attmat)

        c = lw.corners('tel')
        clist = np.array([[c[0][i], c[1][i]] for i in [0, 1, 2, 3]])
                
        for i in range(clist.shape[0]):
            clist[i, :] = lw.tel_to_sky(clist[i, 0], clist[i, 1])

        

        ax.plot([clist[-1, 0], clist[0, 0]], [clist[-1, 1], clist[0, 1]], color="blue")
        for i in range(clist.shape[0]-1):
            ax.plot(clist[i:i+2, 0], clist[i:i+2, 1], color="blue")

        line = plt.Polygon(clist, facecolor="blue", closed=True, alpha=0.3)
        nircam_poly.append(deepcopy(line))
        ax.add_patch(line)
        
    #sub arrays for nircam sw
    aperfill = ['NRCA1_FULL','NRCA2_FULL','NRCA3_FULL','NRCA4_FULL','NRCB1_FULL','NRCB2_FULL','NRCB3_FULL','NRCB4_FULL']

    for i in aperfill:
        lw = nrc_siaf[i]
        lw.set_attitude_matrix(attmat)

        c = lw.corners('tel')
        clist = np.array([[c[0][i], c[1][i]] for i in [0, 1, 2, 3]])

        for i in range(clist.shape[0]):
            clist[i, :] = lw.tel_to_sky(clist[i, 0], clist[i, 1])

        
        
        ax.plot([clist[-1, 0], clist[0, 0]], [clist[-1, 1], clist[0, 1]], color="blue")
        for i in range(clist.shape[0]-1):
            ax.plot(clist[i:i+2, 0], clist[i:i+2, 1], color="blue")

        line = plt.Polygon(clist, facecolor="blue", closed=True, alpha=0.3)
        nircam_poly.append(deepcopy(line))
        ax.add_patch(line)

    #for MIRI
    miri_poly = []
    
    #MIRI array
    aperfill = ["MIRIM_ILLUM"]
    
    for i in aperfill:
        lw = miri_siaf[i]
        lw.set_attitude_matrix(attmat)

        c = lw.corners('tel')
        clist = np.array([[c[0][i], c[1][i]] for i in [0, 1, 2, 3]])

        for i in range(clist.shape[0]):
            clist[i, :] = lw.tel_to_sky(clist[i, 0], clist[i, 1])

        ax.plot([clist[-1, 0], clist[0, 0]], [clist[-1, 1], clist[0, 1]], color="green")
        for i in range(clist.shape[0]-1):
            ax.plot(clist[i:i+2, 0], clist[i:i+2, 1], color="green")

        line = plt.Polygon(clist, facecolor="green", closed=True, alpha=0.3)
        miri_poly.append(deepcopy(line))
        ax.add_patch(line)
        
        
    return nircam_poly, miri_poly

fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()

ras = [215.16198, 215.07260, 214.98013, 215.03239, 215.07260]
decs = [53.05128, 52.98783, 52.92483, 52.90606, 52.98783]
v2s = [-453.559, -453.559, -453.559, -453.559, -453.559]
v3s = [-373.814, -373.814, -373.814, -373.814, -373.814]
v3pa = 130.75

nircam_polys = []
miri_polys = []
cat = Table.read("candels2.fits").to_pandas()

for i in range(len(ras)):
    nc1_poly, miri1_poly = plot_primer_pointing(ax, v2s[i], v3s[i], ras[i], decs[i], v3pa) #IF INCLUDING MIRI need to add ", miri1_poly" after nc1_poly
    nircam_polys.append(nc1_poly)
    miri_polys.append(miri1_poly)
    
    ax.add_patch(nc1_poly[0])     
    ax.add_patch(miri1_poly[0]) 

EGS_coords = [cat["RA"], cat["DEC"]]
EGS_coords = cat[["RA", "DEC"]].values

nc_idlist = []
for i, p in enumerate(nircam_polys):
    poly_idlist = []
    
    for square in p:
        selectionBool = square.get_path().contains_points(EGS_coords)
        poly_idlist.append(cat.loc[selectionBool].ID.values)
    
    nc_idlist.append(poly_idlist)
    
nc_mask = np.repeat(False, cat.shape[0])
mask = np.zeros(cat.shape[0]).astype(bool)

miri_idlist = []

for i, p in enumerate(miri_polys):
    poly_idlist = []
    
    selectionBool = p[0].get_path().contains_points(EGS_coords)
    poly_idlist.append(cat.loc[selectionBool].ID.values)
    
    miri_idlist.append(poly_idlist)

lst = []

for i in range(len(miri_idlist)):
    egs_in_miri = cat.loc[cat.ID.isin(miri_idlist[i][0])]
    lst.append(egs_in_miri)
  
    
for j in range(len(nc_idlist)):
    for i in range(8):
        egs_in_nc = cat.loc[cat.ID.isin(nc_idlist[j][i])]
        lst.append(egs_in_nc)
        
egs_in_ceers = pd.concat(lst)
egs_in_ceers.to_csv(r'/Users/clarapollock/Desktop/JWST/nircam2.csv', header=True, index=None, sep=',', mode='w')

ax.scatter(cat["RA"], cat["DEC"], color="gray", alpha = 0.8, s=2, lw=0)
ax.scatter(egs_in_ceers["RA"], egs_in_ceers["DEC"], color="red", alpha = 0.8, s=2, lw=0)

ax.set_xlabel("RA")
ax.set_ylabel("DEC")

ax.set_xlim(215.3, 214.6)
ax.set_ylim(53.2, 52.5)

plt.show()


