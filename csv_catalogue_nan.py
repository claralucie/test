#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:47:09 2022

@author: s1929920
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "/home/s1929920/jwst"
os.chdir(path)

data = pd.read_csv("z3_catalogue_fit_nan.csv", usecols=["ID", "RA", "DEC", "redshift_50", "stellar_mass_50",
                                                          "r_eff_150", "sersic_n_150", "r_eff_444", "sersic_n_444",
                                                          "value_is_NaN"])

df = pd.DataFrame(data)


r_eff_150 = df["r_eff_150"]
sersic_n_150 = df["sersic_n_150"]
r_eff_444 = df["r_eff_444"]
sersic_n_444 = df["sersic_n_444"]

results_r_eff_150 = []
results_sersic_n_150 = []
results_r_eff_444 = []
results_sersic_n_444 = []

#df.loc[df['r_eff_150'].isnull(), 'value_is_NaN'] = 0
#df.loc[df['r_eff_150'].notnull(), 'value_is_NaN'] = 1
"""
value = df["value_is_NaN"]
for i in range(len(r_eff_150)):
    if value[i] == 1:
        results_r_eff_150.append(r_eff_150[i])
        results_sersic_n_150.append(sersic_n_150[i])
        results_r_eff_444.append(r_eff_444[i])
        results_sersic_n_444.append(sersic_n_444[i])
        
df = data[data["r_eff_150"].isin(results_r_eff_150)]  

df.to_csv('./z3_catalogue_fit_nan.csv', header=True, index=None, sep=',', mode='w')
"""

for i in range(len(r_eff_150)):
    if r_eff_150[i] != 0.0:
        results_r_eff_150.append(r_eff_150[i])
        results_sersic_n_150.append(sersic_n_150[i])
        results_r_eff_444.append(r_eff_444[i])
        results_sersic_n_444.append(sersic_n_444[i])

"""
for i in range(len(r_eff_150)):
    if r_eff_150[i] == 0:
        df.loc['value_is_zero'] = 0
    else:
        df.loc['value_is_zero'] = 1
""" 



df = df[df["r_eff_150"].isin(results_r_eff_150)]
df.to_csv('./z3_catalogue_fit_nan.csv', header=True, index=None, sep=',', mode='w')


"""
r_eff_150 = df['r_eff_150'].tolist()
sersic_n_150 = df['sersic_n_150'].tolist()

r_eff_150_arcsec = np.zeros(len(r_eff_150))
sersic_n_150_arcsec = np.zeros(len(r_eff_150))

r_eff_444 = df['r_eff_444'].tolist()
sersic_n_444 = df['sersic_n_444'].tolist()

r_eff_444_arcsec = np.zeros(len(r_eff_150))
sersic_n_444_arcsec = np.zeros(len(r_eff_150))

for i in range(len(r_eff_150)):
    r_eff_150_arcsec[i] = r_eff_150[i] * 0.031
    sersic_n_150_arcsec[i] = sersic_n_150[i] * 0.031
    r_eff_444_arcsec[i] = r_eff_444[i] * 0.031
    sersic_n_444_arcsec[i] = sersic_n_444[i] * 0.031
    
plt.figure()
plt.scatter(r_eff_150_arcsec, r_eff_444_arcsec)
plt.title("r_eff ratio")
plt.xlabel("effective radius F150W")
plt.ylabel("effective radius F444W")
plt.show()
plt.close()

mass = df['stellar_mass_50'].tolist()

log_r_eff_ratio = np.zeros(len(r_eff_150))


for i in range(len(r_eff_150)):
    log_r_eff_ratio[i] = np.log(r_eff_444[i]/r_eff_150[i])

plt.figure()
plt.scatter(mass, log_r_eff_ratio)
plt.xlabel("log(M*/M_solar)")
plt.ylabel("log of r_eff ratio 444/150")
plt.show()

for i in range(len(log_r_eff_ratio)):
    if log_r_eff_ratio[i]>=30:
        print("ratio is: ", log_r_eff_ratio[i], " and index is; ",  i)
"""