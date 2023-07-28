# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:52:40 2023

@author: Thomas Ball
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import os
import sys

# load aoh data
aoh = pd.read_csv("remaining_aoh.csv",
                       index_col = 0).sort_values("perc", ascending = False)

def get_xy_from_kde(Z, X, Y):
    zs = Z.ravel()
    xs = X.ravel()
    ys = Y.ravel()
    pos = np.random.choice(np.arange(0, len(zs), 1), p = zs)
    return xs[pos],ys[pos]

def homogeneous_areas(base_res, grid_size, scale):
    return np.full((int(grid_size[0]/scale),
                    int(grid_size[1]/scale)),
                   base_res**2*scale**2, dtype = float)

def lat_w_areas(base_res, grid_size, scale):
    arr = homogeneous_areas(base_res, (1, grid_size[1]), scale)
    step = np.pi / (grid_size[1] - 1)
    w = np.arange(-0.5*np.pi, 0.5*np.pi + step, step)
    return arr * np.cos(w)

#species
class Species:
    def __init__(self, curr, perc, scale, pix_size):
        self.curr = curr / scale
        self.perc = perc
        self.pnv = self.curr / self.perc
        self.pix = self.curr // (pix_size**2)

def marginal_deltap(curr, pnv, pix_size, intv_size, ppower):
    """
    curr : float, Current AOH (m2)
    pnv : float, AOH in historic baseline (m2)
    pixel_size : float, Edge-size of pixels (m)
    inv_size : float, Modelled intervention area (m2)
    ppower : float, persistence power
    """
    num_pix = intv_size // (pix_size**2)
    term1 = num_pix * (((curr - pix_size**2)/pnv)**0.25)
    term2 = num_pix * ((curr / pnv)**0.25)
    val = term1 - term2
    return val

def true_deltap(curr, pnv, pix_size, intv_size, ppower):
    """
    curr : float, Current AOH (m2)
    pnv : float, AOH in historic baseline (m2)
    pixel_size : float, Edge-size of pixels (m)
    inv_size : float, Modelled intervention area (m2)
    ppower : float, persistence power
    """
    if curr > intv_size:
        term1 = ((curr - intv_size)/pnv)**0.25
    else: term1 = 0
    term2 = (curr / pnv)**0.25
    val = term1 - term2
    return val

def deltap_dev(curr, pnv, pix_size, intv_size, ppower):
    dp_marg = marginal_deltap(curr, pnv, pix_size, intv_size, ppower)
    dp_true = true_deltap(curr, pnv, pix_size, intv_size, ppower)
    # diff = dp_marg - dp_true
    # perc_diff = diff / dp_true
    return dp_marg, dp_true
    
    
#%%
# probability distribution of current and %remaining aoh
linspace = 300
dat = np.vstack([aoh.curr, aoh.perc])
kde = gaussian_kde(dat)
x_values = np.linspace(np.min(aoh.curr), np.max(aoh.curr), linspace)
y_values = np.linspace(np.min(aoh.perc), np.max(aoh.perc), linspace)
X, Y = np.meshgrid(x_values, y_values)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)
Z = Z / Z.sum()

#%%
# parameters
I = 5
S = 20000
pix_size = 93
ppower = 0.25
world_size = (400000,200000)
scale = 1
world_area = 5.1E14
itv_range = lambda: np.random.uniform(1E4, 1E12)
k = int(sys.argv[1])
dat_path = f"output/mod_deltap_err_{k}.csv"
reps = 300



#%%
cols = ["int_idx", "intv_size", "dp_marg", "dp_true", "dp_diff"]

for r in range(reps):
    if os.path.isfile(dat_path) == True:
        df = pd.read_csv(dat_path,index_col = 0)
        base_i = df.int_idx.max()
    else:
        df = pd.DataFrame(columns = cols)
        base_i = 0
    # interventions
    for i in range(I):
        print(f"proc: {k}, rep: {r}, {i + base_i}")
        i = i + base_i
        intv_size = itv_range()
        #iterate species:
        for s in range(S):
            # create a species
            curr, perc = get_xy_from_kde(Z, X, Y)
            sp = Species(curr, perc, scale, pix_size)
            # does the intervention affect the species
            if np.random.choice([True,False],
                                p = [sp.curr/world_area, 1 - (sp.curr/world_area)]):
                # marg and true delta p
                dp_marg, dp_true, = deltap_dev(sp.curr, 
                                               sp.pnv,
                                               pix_size, 
                                               intv_size, ppower)
                dat = [i, intv_size, dp_marg, dp_true, dp_marg-dp_true]
                df = pd.concat([df, pd.DataFrame(data = dat, index=cols).T])
            # sum marg and true and calculate different
    df.to_csv(dat_path)

# #%%
# y = []
# x = []
# xerr = []
# for int_ in df.int_idx.unique():
#     dat = df[df.int_idx == int_]
    
#     dpm = dat.dp_marg.sum()
#     dpt = dat.dp_true.sum()
    