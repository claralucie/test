#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:32:14 2022

@author: s1929920
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

#---------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa, color="red"):
    nrc_siaf = pysiaf.Siaf('NIRCam')
    miri_siaf = pysiaf.Siaf('MIRI')

    attmat = pysiaf.utils.rotations.attitude_matrix(v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

    #full arrays for nircam lw
    aperfill = ['NRCB5_FULL','NRCA5_FULL']
    nircam1_poly = []
    nircam2_poly = []

    for i in aperfill:
        lw = nrc_siaf[i]
        lw.set_attitude_matrix(attmat)

        c = lw.corners('tel')
        clist = np.array([[c[0][i], c[1][i]] for i in [0, 1, 2, 3]])
                
        for i in range(clist.shape[0]):
            clist[i, :] = lw.tel_to_sky(clist[i, 0], clist[i, 1])

        #print(clist)

        ax.plot([clist[-1, 0], clist[0, 0]], [clist[-1, 1], clist[0, 1]], color="blue")
        for i in range(clist.shape[0]-1):
            ax.plot(clist[i:i+2, 0], clist[i:i+2, 1], color="blue")

        line = plt.Polygon(clist, facecolor="blue", closed=True, alpha=0.3)
        nircam1_poly.append(deepcopy(line))
        ax.add_patch(line)
        
    #sub arrays for nircam sw
    aperfill = ['NRCB1_FULL','NRCB2_FULL','NRCB3_FULL','NRCB4_FULL','NRCA1_FULL','NRCA2_FULL','NRCA3_FULL','NRCA4_FULL']

    for i in aperfill:
        lw = nrc_siaf[i]
        lw.set_attitude_matrix(attmat)

        c = lw.corners('tel')
        clist = np.array([[c[0][i], c[1][i]] for i in [0, 1, 2, 3]])

        for i in range(clist.shape[0]):
            clist[i, :] = lw.tel_to_sky(clist[i, 0], clist[i, 1])

        print(clist)
        
        ax.plot([clist[-1, 0], clist[0, 0]], [clist[-1, 1], clist[0, 1]], color="blue")
        for i in range(clist.shape[0]-1):
            ax.plot(clist[i:i+2, 0], clist[i:i+2, 1], color="blue")

        line = plt.Polygon(clist, facecolor="blue", closed=True, alpha=0.3)
        nircam2_poly.append(deepcopy(line))
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
        
        
    return nircam1_poly, nircam2_poly, miri_poly


fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()


ax.text(214.85, 52.95, 'NIRCam', color='Blue', ha='center')
ax.text(215.1, 52.9, 'MIRI', color='Green', ha='center')
#---------------------------------------------------------------------------------------------------


# Tile 6 
#                                          !         !
v2_ref, v3_ref, ra_ref, dec_ref, v3pa = [-0.32, -492.59, +214.9, 52.9117, 130.78]
nc1_poly6, nc2_poly6, miri_poly6 = plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

# Tile 7
#v2_ref, v3_ref, ra_ref, dec_ref, v3pa = [-453.559, -373.814, +34.34514, -5.27129, 130.78]
#nc1_poly7, nc2_poly7 = plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

# Tile 22
#v2_ref, v3_ref, ra_ref, dec_ref, v3pa = [-453.559, -373.814, +34.44305, -5.27085, 130.78]
#nc1_poly22, nc2_poly22 = plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

nircam1_poly = nc1_poly6 #+ nc1_poly7 + nc1_poly22
nircam2_poly = nc2_poly6 #+ nc2_poly7 + nc2_poly22
miri_poly = miri_poly6

nircam_poly = nircam1_poly + nircam2_poly

nircam1_poly[0].set_label("NIRCamLW")
ax.add_patch(nircam1_poly[0])

nircam2_poly[0].set_label("NIRCamSW")
ax.add_patch(nircam2_poly[0])

miri_poly[0].set_label("MIRI")
ax.add_patch(miri_poly[0])

#---------------------------------------------------------------------------------------------

cat = Table.read("candels1.fits").to_pandas()
#ground_cat = Table.read("vandels17433.fits").to_pandas() #my vandels data doesn't work with pandas?

# Select ...? objects in nircam pointings
mask = np.zeros(cat.shape[0]).astype(bool)
#ground_mask = np.zeros(ground_cat.shape[0]).astype(bool)

ax.scatter(cat["RA"], cat["DEC"], color="gray", alpha=0.8, s=2, lw=0)
#ax.scatter(ground_cat["RA"], ground_cat["DEC"], color="gray", alpha=0.5, s=1, lw=0)

for i in range(cat.shape[0]):
    in_poly = [p.contains_point(cat.loc[i, ["RA", "DEC"]].values) for p in nircam_poly]

    mask[i] = np.max(in_poly)
    if not i % 1000:
        print("Done:", i, "in pointings:", np.sum(mask))


RA_list = cat["RA"]
DEC_list = cat["DEC"]

RA_red = mask*RA_list
DEC_red = mask*DEC_list
RA_grey = RA_list - RA_red
DEC_grey = DEC_list - DEC_red

#(NIRCAM) V2 range from 214.88472393 to 214.96897441, and 214.83160114 to 214.91549774
#V3 range from 52.90504548 to 52.95574134, and 52.86863805 to 52.91931832

"""
for i in range(len(RA_list)):
    if (RA_list[i]<= 214.96897441):
        if (RA_list[i]>= 214.88472393):
            if (DEC_list[i]<=52.95574134):
                if (DEC_list[i] >= 52.90504548):
                    RA_target.append(RA_list[i])
                    DEC_target.append(DEC_list[i])
    else:
        RA_grey.append(RA_list[i])
        DEC_grey.append(DEC_list[i])
   """                 
   
ax.scatter(RA_grey, DEC_grey, color="gray", alpha=0.8, s=2, lw=0)
ax.scatter(RA_red, DEC_red, color="red", alpha=0.8, s=2, lw=0)

        

ax.set_xlabel("RA")
plt.ylabel("DEC")

ax.set_xlim(215.3, 214.6)
ax.set_ylim(53.2, 52.5)

#---------------------------------------master apertures-------------------------------------------------
"""
# plot 'master' apertures
from pysiaf.siaf import plot_master_apertures
plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k'); plt.clf()
plot_master_apertures(mark_ref=True)
plt.show()
"""
#---------------------------------------------------------------------------------------------------------------
#plot targets from mast data
pd.options.display.max_rows = 234

df = pd.read_csv('MAST_egs.csv', header=4, usecols=('s_ra', 's_dec'))

RA = df.iloc[:,0]
DEC = df.iloc[:,1]
print(df)

fig = plt.figure()
#RA, DEC = zip(*sorted(zip(RA, DEC)))
plt.scatter(RA, DEC)
plt.xlabel('RA')
plt.ylabel('Dec')

plt.show()


