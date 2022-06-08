#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:32:53 2021

@author: Alex Yeh
"""

import glob
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import integrate
from scipy import stats

def read_dump(filename):
    """Reads the timesteps and geometric data from a LAMMPS dump file"""
    time_acc = []
    pnum = None
    coord_acc = []
    
    with open(filename) as outfile:
        file_iter = enumerate(outfile)
        for i, line in file_iter:
            if line.startswith('ITEM: TIMESTEP'):
                i, line = next(file_iter)
                time_acc.append(int(line.strip()))
            # currently can only parse simulations with constant particle #
            if line.startswith('ITEM: NUMBER OF ATOMS') and pnum is None:
                i, line = next(file_iter)
                pnum = int(line.strip())
            if line.startswith('ITEM: ATOMS id type'):
                frame = []
                for n in range(pnum):
                    i, line = next(file_iter)
                    #grab only the x y z coords as floats
                    frame.append([float(i) for i in line.split()[2:5]])
                        
                coord_acc.append(frame)
    multiple = np.array(coord_acc)
    times = np.array(time_acc)
    return multiple, times

def read_infile(filename):
    """extracts experimental parameters from .in file. Returns a dict of each
    value set in the calculation."""
    
    starts = [['unit', 1, 'unit'],
              ['timestep', 1, 'timestep'],  # split idx for timestep
              ['fix temp all langevin', 6, 'damp'],  # split idx for damp
              ['pair_coeff 1 1', 3, 'bpp'], # [kT]
              ['pair_style yukawa/colloid', 2, 'kappa_2a'], # [1/(2a)]
              ['fix step all nve/manifold/rattle', -1, 'rad'], #get shell radius
              ]
    
    txt = []
    with open(filename) as infile:
        for line in infile:
            txt.append(line)
            
    txt_arr = np.array(txt)
    
    out = {}
    for pre, offset, name in starts:
        mask = np.char.startswith(txt_arr, pre)
        if np.any(mask):
            curr = txt_arr[mask][0]
            if pre == 'unit':
                out[name] = curr.split()[offset]
            else:
                out[name] = float(curr.split()[offset])
    
    if out:
        if 'timestep' not in out:
            out['timestep'] = 0.005 #default time step for lj units in LAMMPS
        return out
    else:
        raise ValueError(f'no valid lines in {filename}')

def mto_msd_part(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd per particle up to the given max_step, defaulting to the maximum 
    number of non-overlapping multiple time origins. Overlapping time orgins 
    can be given by specifying a skip param less than 2*max_lag.
    Returns a [T x N x 3] array of msds
    """
    if skips is None:
        skips = 2*max_lag #non-overlapping mtos
    
    pnum = coords.shape[1]
    total_steps = coords.shape[0]
    orig_num = int(total_steps/(skips))
    final_step = (orig_num-1)*skips + (max_lag-1) #last necessary timestep
    assert final_step<total_steps, f'{final_step} will exceed array size({total_steps}), must specify smaller number of skips'
    
    msd = np.zeros((max_lag, pnum, 3))
    
    for t in range(max_lag):
        for tstart in range(1, orig_num*skips, skips):
            tend = tstart + t
            msd[t] += (coords[tend] - coords[tstart])**2
    
    return msd/orig_num

def mto_msd(coords, max_lag, skips=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the msd up to the given max_step, defaulting to the maximum number of 
    non-overlapping multiple time origins. Overlapping time orgins can be
    given by specifying a skip param less than 2*max_lag.
    Returns a [T x 3] array of msd values
    """
    msd_part = mto_msd_part(coords, max_lag, skips=skips)
    return np.average(msd_part, axis=1)

def bootstrap_mto_msd(msd_part, trials, confidence=95, rng=None):
    """Given a set of T timesteps of N particles ([T x N x 3]), computes 
    the confidence interval on the msd using the bootstrap with the given # of 
    trials, skips specifies the number of frames between time origins and
    further details can be found in mto_msd_part.
    
    The bootstrapping operation was based on code used by Coscia and Shirts,
    e.g. https://doi.org/10.1021/acs.jpcb.9b04472, which can be accessed in the 
    following repository: https://github.com/shirtsgroup/LLC_Membranes under 
    the path /LLC_Membranes/timeseries/msd.py
    
    Returns a [T x 2] array with lower and upper bounds at each frame.
    """
    if rng is None:
        rng = default_rng()
        
    summed_msd = np.sum(msd_part, axis=-1)
    ave_msd = np.average(summed_msd, axis=-1)
    boot_msd = np.zeros((msd_part.shape[0], trials))
    
    for b in range(trials):
        # get indices with replacement
        boot_idx = rng.integers(0, msd_part.shape[1], msd_part.shape[1])
        # average over msds for each bootstrap trial
        boot_msd[:, b] = np.average(summed_msd[:, boot_idx], axis=-1)
        
    #get confidence intervals
    msd_ci = np.zeros((msd_part.shape[0], 2))
    low = (100 - confidence)/2
    high = 100 - low
    
    msd_ci[:,0] = ave_msd - np.percentile(boot_msd, low, axis=1)
    msd_ci[:,1] = np.percentile(boot_msd, high, axis=1) - ave_msd
    return msd_ci

#%%
if __name__=='__main__':
    #read the first input file and dump file in current directory
    infiles = glob.glob('*.in')
    lammps_params = read_infile(infiles[0])
    coordfiles = glob.glob('*.dump')
    multiple, ts = read_dump(coordfiles[0])
    
    dt = lammps_params['timestep']
    shell_radius = lammps_params['rad']
    damp = lammps_params['damp']
    times = ts*dt
    
    def sphere_msd(taus, damp, shell_radius = 10):
        """Theoretical msd in 3d of particles confined onto the surface of a 
        sphere as reported in:
        Paquay and Kusters Biophysical Journal 2016, 110 (6), 1226–1233. 
        https://doi.org/10.1016/j.bpj.2016.02.017.
        """
        return 2*(shell_radius**2)*(1-np.exp(-2*damp*taus*shell_radius**-2))
    
    #%% calculate msd
    msd_time_scale = 501 # number of frames used to calculate msd
    msd_times = times[:msd_time_scale]
    
    # below contains x, y, z components of msd for each atom []
    msd_part = mto_msd_part(multiple, msd_time_scale) 
    msd_comp = msd_part.mean(axis=1) # averaging over atoms
    msd = msd_comp.sum(axis=-1) # summing over the components gives the overall msd

    #%% get bootstrap error
    trials = 1000
    rng = default_rng()
        
    #get confidence intervals
    msd_ci = bootstrap_mto_msd(msd_part, trials, rng=rng)
    
    # set radius in msd function to ensure we don't include it as optimizable param
    msd_func = lambda x, damp: sphere_msd(x, damp, shell_radius=shell_radius)
    
    diff_coef, diff_cov = curve_fit(msd_func, msd_times, msd, p0=[1e-1])
    theo = sphere_msd(msd_times, damp, shell_radius)
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(msd_times, msd, label='mto msd')
    ax.fill_between(msd_times, msd-msd_ci[:,0], msd+msd_ci[:,1],
                    alpha=0.3, label='95% bootstrap ci')
    ax.plot(msd_times, theo, color='k', ls=':', label=f'D={damp:0.1e}')
    ax.plot(msd_times, msd_func(msd_times, *diff_coef), 
            color='C0', ls='-.',
            label=f'D={diff_coef[0]:0.3f} (fit)')
    ax.legend()
    ax.set_xlabel("[$\\tau$]", fontsize=12)
    ax.set_xlim([0, msd_times[-1]])
    ax.set_ylabel("[$\sigma ^2$]", fontsize=12)
    ax.set_ylim([0, 1.1*msd_func(msd_times[-1], damp)])
    title = f"msd (R: {shell_radius:0.3f} "+r"[2a], pair_style none)"
    ax.set_title(title)