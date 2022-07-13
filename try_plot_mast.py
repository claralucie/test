#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:09:11 2022

@author: s1929920
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from astropy.table import Table
"""
filename = "MAST_egs.csv"
delimiter = ','
nheaderlines = 4


usecols = ['target_name', 's_ra', 's_dec']

data = pd.read_csv(filename, delimiter, skiprows=(4))

RA_list = data[:,usecols[1]]
DEC_list = data[:,usecols[2]]

fig = plt.figure()
plt.plot(RA_list, DEC_list)
plt.xlabel('RA')
plt.ylabel('Dec')

plt.show()
"""
cat = Table.read("candels1.fits").to_pandas()
#ground_cat = Table.read("vandels17433.fits").to_pandas() #my vandels data doesn't work with pandas?

# Select ...? objects in nircam pointings
mask = np.zeros(cat.shape[0]).astype(bool)
#ground_mask = np.zeros(ground_cat.shape[0]).astype(bool)

pd.options.display.max_rows = 234

df = pd.read_csv('MAST_egs.csv', header=4, usecols=('s_ra', 's_dec'))

RA_list = df.iloc[:,0]
DEC_list = df.iloc[:,1]

fig = plt.figure(figsize=(20, 12))
ax = plt.subplot()
ax.scatter(cat["RA"], cat["DEC"], color="gray", alpha=0.8, s=1, lw=0)
#RA, DEC = zip(*sorted(zip(RA, DEC)))
plt.scatter(RA_list, DEC_list, color="red", alpha = 1.0, marker = 'o', s=10, lw=0)
plt.title('JWST MAST plottings over CEERS')
plt.xlabel('RA')
plt.ylabel('Dec')

plt.show()