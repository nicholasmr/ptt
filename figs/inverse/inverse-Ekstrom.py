#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2025-

"""
Invert Ekstrom CMP data for profiles of fabric and firn structure
"""

import numpy as np
from scipy.io import loadmat
import sys, code # code.interact(local=locals()) 
sys.path.append("../../lib")
from MG import *

"""
Load data
"""

data = loadmat("../../data/Ekstrom/pCMP_dTWTT_VV-HH_from_phase.mat")

H = data["depthBase"][0][0]
L = data["offset"][0]/2 # half-offset
L[0] = 0
print('*** Running Ekstrom inversion where H=%.1f'%(H))

x_obs = L/H
d = data['depthReflections'][0]
#print(np.median(np.diff(d))) # median vertical resolution
z_obs = (H-d)/H

#dTWTTbase_obs = data['twttDiffBase']
dTWTT_obs = -data['twttDiff']
X_obs, Z_obs = np.meshgrid(x_obs, z_obs)

"""
Figure and style
"""

scale = 0.65
fig = plt.figure(figsize=(10.5*scale, 2.5*scale))
kmeas = 1.4
gs0 = fig.add_gridspec(1, 7, wspace=0.2, width_ratios=[0.8,0.8, 1.4, 0.35, kmeas,kmeas,kmeas])
ax1 = fig.add_subplot(gs0[0, 0])
ax2 = fig.add_subplot(gs0[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs0[0, 2], sharey=ax1)
ax_ = fig.add_subplot(gs0[0, 3], sharey=ax1)
ax4 = fig.add_subplot(gs0[0, 4], sharey=ax1)
ax5 = fig.add_subplot(gs0[0, 5], sharey=ax1)
ax6 = fig.add_subplot(gs0[0, 6], sharey=ax1)

axes = (ax1,ax2,ax3,ax4,ax5,ax6)

FSLEG = 11

def plot_profiles(ppi=0):
    kw = dict(lw=lwi[ppi], c=ci[ppi])
    MG.plot_density(ax1, **kw)
    MG.plot_eccentricity(ax2, **kw)
    MG.plot_eigenvalues(ax3, cy=ci[ppi], **kw)
    kw = dict(color=ci[ppi], fontsize=FSLEG-2, rotation=90, va='bottom', ha='left')
    ax1.text(0.4+0.22*ppi, 0.05, r'$\hat{z}_{\mathrm{bco}}=%.2f$ %s'%(MG.zbco, elbl[ppi]), **kw)
    ax2.text(0.3+0.295*ppi, 0.05, "$p=%.1f$, $e_0=%.1f$"%(MG.p,MG.e0), **kw)

""" 
Inverse experiment
"""

kwi_pp = [dict(p=0.5,e0=0.8), dict(p=0.9,e0=0.72)] 
c1, c2 = "#238b45", "#8c510a"
ci = (c1, c2)
lwi = [1.7,1.15]
elbl = ['', '']

def plot_inverse_experiment(X_ssobs, Z_ssobs, dTWTT_ssobs, ppi=0):
#    args_guess=(0.8, 1/3,1/3, 1/3,1/3) # uninformed
    args_guess=(0.9, 1/3,0, 1/3,1) # used in paper
    args_infr = MG.infer_params(X_ssobs, Z_ssobs, dTWTT_ssobs, args_guess=args_guess, kw_pp=kwi_pp[ppi])
    plot_profiles(ppi=ppi)
    return args_infr

resmul = 1 
#resmul = 0.1 # reduced resolution for debugging 

L = np.linspace(0,H,int(25*resmul)) # half-offsets sampled (L=0 is nadir)
MG = MaxwellGarnettColumn(H=H, L=L, Nl=int(30*resmul))
inp = (X_obs.flatten(), Z_obs.flatten(), dTWTT_obs.flatten())

args_infr1 = plot_inverse_experiment(*inp, ppi=0)
MG.set_physprops(*args_infr1)
h = MG.plot_dTWTT_expr(fig, ax5, MG.X, MG.Z, MG.dTWTT()[0], label=r'Inferred 1', clbl=ci[0])

args_infr2 = plot_inverse_experiment(*inp, ppi=1)
MG.set_physprops(*args_infr2)
h = MG.plot_dTWTT_expr(fig, ax6, MG.X, MG.Z,  MG.dTWTT()[0], label=r'Inferred 2', clbl=ci[1], caxpos=[0.795, 0.73, 0.09, 0.05])

hleg = ax3.legend([r'$\lambda_{z}$',r'$\lambda_{x}$',r'$\lambda_{y}$'], loc=1, frameon=False, handlelength=1.1, handletextpad=0.3, labelspacing=0.05, bbox_to_anchor=(1.13,1.08), fontsize=FSLEG)
for ii in range(3): hleg.legend_handles[ii].set_color('k')

zi = [0.05, 0.475, 0.90]
ax_.set_axis_off()
MG.plot_ODFcolumn(ax_, fig, zi, WH=0.2, lvlmax=0.25)

"""
Plot dTWTT
"""

h = MG.plot_dTWTT_expr(fig, ax4, X_obs, Z_obs, dTWTT_obs, label='Observed')
ax4.text(0.90, 0.90, "N/A", c='k', fontsize=FSLEG-1, ha='right', va='top')

"""
Warp up and save plot
"""

for ii, ax in enumerate(axes):
    if ii>0: plt.setp(ax.get_yticklabels(), visible=False)
    else:    MG._set_yaxis(ax)
        
    kw = dict(loc=2, frameon=False, prop=dict(size=11), bbox_to_anchor=(-0.2 +(ii>=3)*0.05, 1.25), bbox_transform=ax.transAxes)
    at = AnchoredText(r'{(%s)}'%(chr(ord('a')+ii)), **kw)
    ax.add_artist(at)

plt.savefig('inverse-Ekstrom.png', dpi=200, bbox_inches='tight', pad_inches=0.04)

