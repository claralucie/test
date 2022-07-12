import pysiaf
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.table import Table
import pandas as pd
from copy import deepcopy


def plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa, color="red"):
    nrc_siaf = pysiaf.Siaf('NIRCam')
    miri_siaf = pysiaf.Siaf('MIRI')

    attmat = pysiaf.utils.rotations.attitude_matrix(v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

    aperfill = ['NRCB5_FULL', 'NRCA5_FULL']
    nircam_poly = []
    miri_poly = []

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


# Tile 6
v2_ref, v3_ref, ra_ref, dec_ref, v3pa = [-453.559, -373.814, 34.35043, -5.29117, 253.2]
nc_poly6, miri_poly6 = plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

# Tile 7
v2_ref, v3_ref, ra_ref, dec_ref, v3pa = [-453.559, -373.814, +34.34514, -5.27129, 254]
nc_poly7, miri_poly7 = plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

# Tile 22
v2_ref, v3_ref, ra_ref, dec_ref, v3pa = [-453.559, -373.814, +34.44305, -5.27085, 250.5]
nc_poly22, miri_poly22 = plot_primer_pointing(ax, v2_ref, v3_ref, ra_ref, dec_ref, v3pa)

nircam_poly = nc_poly6 + nc_poly7 + nc_poly22
miri_poly = miri_poly6 + miri_poly7 + miri_poly22

nircam_poly[0].set_label("NIRCam")
ax.add_patch(nircam_poly[0])

miri_poly[0].set_label("MIRI")
ax.add_patch(miri_poly[0])


cat = Table.read("uds_salim_1000_with_photom.fits").to_pandas()
ground_cat = Table.read("VANDELS_UDS_GROUND_PHOT_v1.0.fits").to_pandas()


# Select HST objects in nircam pointings
mask = np.zeros(cat.shape[0]).astype(bool)
ground_mask = np.zeros(ground_cat.shape[0]).astype(bool)
"""
for i in range(cat.shape[0]):
    in_poly = [p.contains_point(cat.loc[i, ["RA", "DEC"]].values) for p in nircam_poly]

    mask[i] = np.max(in_poly)
    if not i % 1000:
        print("Done:", i, "in pointings:", np.sum(mask))
"""
"""
for i in range(ground_cat.shape[0]):
    in_poly = [p.contains_point(ground_cat.loc[i, ["RA", "DEC"]].values) for p in nircam_poly]

    ground_mask[i] = np.max(in_poly)
    if not i % 1000:
        print("Done:", i, "in pointings:", np.sum(ground_mask))
"""

ax.scatter(cat["RA"], cat["DEC"], color="gray", alpha=0.8, s=2, lw=0)
ax.scatter(ground_cat["RA"], ground_cat["DEC"], color="gray", alpha=0.5, s=1, lw=0)

# Overwrite in nircam fits table
#cat = cat.groupby(mask).get_group(True)
#Table.from_pandas(cat).write("uds_salim_with_photom_in_nircam.fits", overwrite=True)

#ground_cat = ground_cat.groupby(ground_mask).get_group(True)
#Table.from_pandas(ground_cat).write("VANDELS_UDS_GROUND_PHOT_v1.0_in_nircam.fits", overwrite=True)

cat = Table.read("uds_salim_with_photom_in_nircam.fits").to_pandas()
ax.scatter(cat["RA"], cat["DEC"], color="red", alpha=0.8, s=2, lw=0)

ground_cat = Table.read("VANDELS_UDS_GROUND_PHOT_v1.0_in_nircam.fits").to_pandas()
ax.scatter(ground_cat["RA"], ground_cat["DEC"], color="red", alpha=0.8, s=2, lw=0)

ax.set_xlabel("RA")
plt.ylabel("DEC")

ax.set_xlim(34.62, 34.2)
ax.set_ylim(-5.33, -5.05)

vandels_cat = Table.read("vandels_uds_both_spec_july_primer.fits").to_pandas()
ax.scatter(vandels_cat["RA"], vandels_cat["DEC"], marker="+", s=40, color="black", label="VANDELS\ spectra")

ax.legend(frameon=False, loc=2)

plt.savefig("first_primer_pointings.pdf", bbox_inches="tight")
