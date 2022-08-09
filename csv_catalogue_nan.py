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
import matplotlib.lines 

path = "/home/s1929920/jwst"
os.chdir(path)

data1 = pd.read_csv("z1_catalogue_fit_nan.csv", usecols=["ID", "RA", "DEC", "redshift_50", "stellar_mass_50",
                                                          "r_eff_150", "sersic_n_150", "r_eff_444", "sersic_n_444",
                                                          "value_is_NaN"])

df1 = pd.DataFrame(data1)

data2 = pd.read_csv("z2_catalogue_fit_nan.csv", usecols=["ID", "RA", "DEC", "redshift_50", "stellar_mass_50",
                                                          "r_eff_150", "sersic_n_150", "r_eff_444", "sersic_n_444",
                                                          "value_is_NaN"])

df2 = pd.DataFrame(data2)

df3 = pd.read_csv("z3_catalogue_fit_nan.csv", usecols=["ID", "RA", "DEC", "redshift_50", "stellar_mass_50",
                                                       "r_eff_150", "sersic_n_150", "r_eff_444", "sersic_n_444",
                                                        "value_is_NaN"])
"""
r_eff_150 = df["r_eff_150"]
sersic_n_150 = df["sersic_n_150"]
r_eff_444 = df["r_eff_444"]
sersic_n_444 = df["sersic_n_444"]

results_r_eff_150 = []
results_sersic_n_150 = []
results_r_eff_444 = []
results_sersic_n_444 = []
"""
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
"""
for i in range(len(r_eff_150)):
    if r_eff_150[i] != 0.0:
        results_r_eff_150.append(r_eff_150[i])
        results_sersic_n_150.append(sersic_n_150[i])
        results_r_eff_444.append(r_eff_444[i])
        results_sersic_n_444.append(sersic_n_444[i])
"""
"""
for i in range(len(r_eff_150)):
    if r_eff_150[i] == 0:
        df.loc['value_is_zero'] = 0
    else:
        df.loc['value_is_zero'] = 1
""" 



#df = df[df["r_eff_150"].isin(results_r_eff_150)]
#df.to_csv('./z3_catalogue_fit_nan.csv', header=True, index=None, sep=',', mode='w')



r_eff_150_1 = df1['r_eff_150'].tolist()
sersic_n_150_1 = df1['sersic_n_150'].tolist()

r_eff_150_arcsec_1 = np.zeros(len(r_eff_150_1))
sersic_n_150_arcsec_1 = np.zeros(len(r_eff_150_1))

r_eff_444_1 = df1['r_eff_444'].tolist()
sersic_n_444_1 = df1['sersic_n_444'].tolist()

r_eff_444_arcsec_1 = np.zeros(len(r_eff_150_1))
sersic_n_444_arcsec_1 = np.zeros(len(r_eff_150_1))

for i in range(len(r_eff_150_1)):
    r_eff_150_arcsec_1[i] = r_eff_150_1[i] * 0.031
    sersic_n_150_arcsec_1[i] = sersic_n_150_1[i] * 0.031
    r_eff_444_arcsec_1[i] = r_eff_444_1[i] * 0.031
    sersic_n_444_arcsec_1[i] = sersic_n_444_1[i] * 0.031
   
    """
#FIGURE 1 - RADIUS RATIO
plt.figure()
plt.scatter(r_eff_150_arcsec_1, r_eff_444_arcsec_1)
plt.title("2.0 < z < 2.5")
plt.xlabel("effective radius F150W")
plt.ylabel("effective radius F444W")
plt.plot(np.unique(r_eff_150_arcsec_1), np.poly1d(np.polyfit(r_eff_150_arcsec_1, r_eff_444_arcsec_1, 1))(np.unique(r_eff_150_arcsec_1)))
plt.show()
plt.close()
"""
mass_1 = df1['stellar_mass_50'].tolist()

log_r_eff_ratio_1 = np.zeros(len(r_eff_150_1))


for i in range(len(r_eff_150_1)):
    log_r_eff_ratio_1[i] = np.log(r_eff_444_1[i]/r_eff_150_1[i])

#FIGURE 1 - MASS RADIUS
plt.figure()
plt.scatter(mass_1, log_r_eff_ratio_1)
plt.title('1.0 < z < 1.5')
plt.xlabel("log(M*/M_solar)")
plt.ylabel("log of r_eff ratio 444/150")
plt.ylim(-0.75, 0.5)
plt.plot(np.unique(mass_1), np.poly1d(np.polyfit(mass_1, log_r_eff_ratio_1, 1))(np.unique(mass_1)))
plt.show()
plt.close()

#-----------------------------------------------------------------------------------------------------
r_eff_150_2 = df2['r_eff_150'].tolist()
sersic_n_150_2 = df2['sersic_n_150'].tolist()

r_eff_150_arcsec_2 = np.zeros(len(r_eff_150_2))
sersic_n_150_arcsec_2 = np.zeros(len(r_eff_150_2))

r_eff_444_2 = df2['r_eff_444'].tolist()
sersic_n_444_2 = df2['sersic_n_444'].tolist()

r_eff_444_arcsec_2 = np.zeros(len(r_eff_150_2))
sersic_n_444_arcsec_2 = np.zeros(len(r_eff_150_2))

for i in range(len(r_eff_150_2)):
    r_eff_150_arcsec_2[i] = r_eff_150_2[i] * 0.031
    sersic_n_150_arcsec_2[i] = sersic_n_150_2[i] * 0.031
    r_eff_444_arcsec_2[i] = r_eff_444_2[i] * 0.031
    sersic_n_444_arcsec_2[i] = sersic_n_444_2[i] * 0.031

mass_2 = df2['stellar_mass_50'].tolist()
log_r_eff_ratio_2 = np.zeros(len(r_eff_150_2))
    
for i in range(len(r_eff_150_2)):
    log_r_eff_ratio_2[i] = np.log(r_eff_444_2[i]/r_eff_150_2[i])
    
#FIGURE 2 - MASS RADIUS
plt.figure()
plt.scatter(mass_2, log_r_eff_ratio_2)
plt.title('1.5 < z < 2.0')
plt.xlabel("log(M*/M_solar)")
plt.ylabel("log of r_eff ratio 444/150")
plt.ylim(-0.75, 0.5)
plt.plot(np.unique(mass_2), np.poly1d(np.polyfit(mass_2, log_r_eff_ratio_2, 1))(np.unique(mass_2)))
plt.show()
plt.close()

#-------------------------------------------------------------------------------------------------------
r_eff_150_3 = df3['r_eff_150'].tolist()
sersic_n_150_3 = df3['sersic_n_150'].tolist()

r_eff_150_arcsec_3 = np.zeros(len(r_eff_150_3))
sersic_n_150_arcsec_3 = np.zeros(len(r_eff_150_3))

r_eff_444_3 = df3['r_eff_444'].tolist()
sersic_n_444_3 = df3['sersic_n_444'].tolist()

r_eff_444_arcsec_3 = np.zeros(len(r_eff_150_3))
sersic_n_444_arcsec_3 = np.zeros(len(r_eff_150_3))

for i in range(len(r_eff_150_3)):
    r_eff_150_arcsec_3[i] = r_eff_150_3[i] * 0.031
    sersic_n_150_arcsec_3[i] = sersic_n_150_3[i] * 0.031
    r_eff_444_arcsec_3[i] = r_eff_444_3[i] * 0.031
    sersic_n_444_arcsec_3[i] = sersic_n_444_3[i] * 0.031

mass_3 = df3['stellar_mass_50'].tolist()
log_r_eff_ratio_3 = np.zeros(len(r_eff_150_3))
    
for i in range(len(r_eff_150_3)):
    log_r_eff_ratio_3[i] = np.log(r_eff_444_3[i]/r_eff_150_3[i])
    
#FIGURE 3 - MASS RADIUS
plt.figure()
ax = plt.subplot()
plt.scatter(mass_3, log_r_eff_ratio_3)
plt.title('2.0 < z < 2.5')
plt.xlabel("log(M*/M_solar)")
plt.ylabel("log of r_eff ratio 444/150")
plt.ylim(-0.75, 0.5)
#plt.axhline(y=0, xmin=9, xmax=11.5, c="black", linewidth=2, zorder=0)
plt.plot(np.unique(mass_3), np.poly1d(np.polyfit(mass_3, log_r_eff_ratio_3, 1))(np.unique(mass_3)))
plt.show()
plt.close()


#------------------------------------------------------------------------------------------------------
r_eff_444_total = []

for i in range(len(r_eff_444_arcsec_1)):
    r_eff_444_total.append(r_eff_444_arcsec_1[i])
    
for i in range(len(r_eff_444_arcsec_2)):
    r_eff_444_total.append(r_eff_444_arcsec_2[i])
    
for i in range(len(r_eff_444_arcsec_3)):
    r_eff_444_total.append(r_eff_444_arcsec_3[i])

r_eff_150_total = []

for i in range(len(r_eff_150_1)):
    r_eff_150_total.append(r_eff_150_arcsec_1[i])
for i in range(len(r_eff_150_2)):
    r_eff_150_total.append(r_eff_150_arcsec_2[i])
for i in range(len(r_eff_150_3)):
    r_eff_150_total.append(r_eff_150_arcsec_3[i])

#print(len(r_eff_150_total))
#print(len(r_eff_444_total))


#FIGURE 4 - RADIUS RATIO
plt.figure()
ax = plt.subplot()
plt.scatter(r_eff_150_total, r_eff_444_total)
plt.title("1.0 < z < 2.5")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("effective radius F150W")
plt.ylabel("effective radius F444W")
line = matplotlib.lines.Line2D([0, 1], [0, 1], color='red', linestyle ='dashed')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()
plt.close()