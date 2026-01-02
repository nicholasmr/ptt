#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2025-

"""
Invert synthetic TWTT experiments of an idealized ice column for profiles of fabric and firn structure
"""

import sys, code # code.interact(local=locals())
import numpy as np
sys.path.append("../../lib")
from MG import *

""" 
Init column model
"""

resmul = 1
#resmul = 0.1 # reduced resolution for debugging

H = 250 # ice thickness (m)
L = np.linspace(0,H,int(25*resmul)) # half-offsets sampled (L=0 is nadir)
MG = MaxwellGarnettColumn(H=H, L=L, Nl=int(30*resmul))

"""
Setup TRUE profile and calculate dTWTT map
"""

EXP = int(sys.argv[1])
print('*** Running experiment %i'%(EXP))

if EXP==0:
    cx0, cz0 = 1/3, 1/3
    cx1, cz1 = 0, 1
    
if EXP==1: 
    cx0, cz0 = 1/3, 1/3
    cx1, cz1 = 0, 1/2

if EXP==2:
    cx0, cz0 = 0.2, 0.5
    cx1, cz1 = 0.0, 0.75

zbc0 = 0.9
args_true = (zbc0, cx0,cx1, cz0,cz1) 
print('*** args_true:  %s'%(MG.argsstr(*args_true)))
MG.set_physprops(*args_true)
dTWTT_true, *_ = MG.dTWTT()

"""
Construct observed dTWTT map in measurement space
"""

std = 0.25e-9
noise = np.random.normal(loc=0.0, scale=std, size=dTWTT_true.shape)
dTWTT_obs = dTWTT_true + noise

alphalimit = 70
dTWTT_obs[MG.alphadeg > alphalimit] = np.nan # assume dTWTT cannot be measured for grazing-angle acquisitions
#dTWTT_obs = dTWTT_true # perfect observations (debug)

X_obs = MG.X
Z_obs = MG.Z

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

"""
Plot PP profiles
"""

elbl = ['(true)', '(infr.)', '(infr.)']

def plot_profiles(lw=1.65, c='k', ytxti=0):

    kw = dict(lw=lw, c=c)
    MG.plot_density(ax1, **kw)
    MG.plot_eccentricity(ax2, **kw)
    MG.plot_eigenvalues(ax3, noy=True, **kw)
    
    kw = dict(c=c, fontsize=FSLEG-2, rotation=90)
    ax1.text(0.35+0.21*ytxti,  0.05, r'$\hat{z}_{\mathrm{bco}}=%.2f$ %s'%(MG.zbco, elbl[ytxti]), **kw)

plot_profiles()
ax2.text(0.2, 0.1, "$e_0=%.1f$\n$p=%.1f$"%(MG.e0,MG.p), fontsize=FSLEG-2, rotation=90)
ax3.legend(loc=1, frameon=False, fontsize=FSLEG, handlelength=1.0, handletextpad=0.2, labelspacing=0.04, bbox_to_anchor=(1.13,1.08))

zi = [0.05, 0.475, 0.90]
ax_.set_axis_off()
MG.plot_ODFcolumn(ax_, fig, zi, WH=0.20)

""" 
Inverse experiments
"""

   
def plot_inverse_experiment(X_ssobs, Z_ssobs, dTWTT_ssobs, ci=0, mrk='o', fs='none', ms=10):

    args_infr = MG.infer_params(X_ssobs, Z_ssobs, dTWTT_ssobs)
    plot_profiles(lw=0.75, c=cd[ci], ytxti=ci+1)
    
    if len(Z_ssobs) < 20: # plot stencil if consists of less than this number of points?
        ax4.plot(X_ssobs, Z_ssobs, marker=mrk, c=cd[ci], ls='none', fillstyle=fs, clip_on=False, markersize=ms, zorder=20)
    
    return args_infr

cd = ['#238b45', '#6a3d9a']

style0 = dict(ci=0)
style1 = dict(ci=1, mrk='o', fs='none', ms=6)

### Full

if 1:
    args_infr = plot_inverse_experiment(X_obs.flatten(), Z_obs.flatten(), dTWTT_obs.flatten(), **style0)
    MG.set_physprops(*args_infr)
    dTWTT_infr, *_ = MG.dTWTT()
else:
    # debug
    dTWTT_infr = dTWTT_true

### Subsampled stencil(s)

inp = (X_obs, Z_obs, dTWTT_obs)

sspace = (np.linspace(0,0.5,3), np.linspace(0,0.75,4)) # sample space (xi,zi)
args_infr1 = plot_inverse_experiment(*subsample_row(*inp, *sspace), **style1)
    
# Debug
#sspace = (np.linspace(0,0.5,3), np.linspace(0,0.75,4)) # sample space (xi,zi)
#args_infr1 = plot_inverse_experiment(*subsample_row(*inp, *sspace), **style2)

"""
Plot dTWTT
"""

XZ = (MG.X, MG.Z)
h = MG.plot_dTWTT_expr(fig, ax4, *XZ, dTWTT_obs,  label='``Observed"')
h = MG.plot_dTWTT_expr(fig, ax5, *XZ, dTWTT_infr, label='Inferred')
h = MG.plot_dTWTT_expr(fig, ax6, *XZ, dTWTT_true, label='True', caxpos=[0.795, 0.73, 0.09, 0.05])

ax4.text(0.93, 0.94, "N/A", c='k', fontsize=FSLEG-1, ha='right', va='top')

"""
Warp up and save plot
"""

for ii, ax in enumerate(axes):
    if ii>0: plt.setp(ax.get_yticklabels(), visible=False)
    else:    MG._set_yaxis(ax)
        
    kw = dict(loc=2, frameon=False, prop=dict(size=11), bbox_to_anchor=(-0.33 +(ii>=2)*0.1, 1.25), bbox_transform=ax.transAxes)
    at = AnchoredText(r'{(%s)}'%(chr(ord('a')+ii+len(axes)*EXP)), **kw)
    ax.add_artist(at)
    
plt.savefig('inverse-ideal-E%i.pdf'%(EXP), bbox_inches='tight', pad_inches=0.04)

