#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2025-

"""
Calculate dTWTT for an idealized solid ice column
"""

import numpy as np
import sys, code # code.interact(local=locals())
sys.path.append("../../lib")
from MG import *

""" 
Initialize model
"""

Nl, Ns = (50,30)

### Geometry

H = 250 # ice thickness (m)
L = np.linspace(0,H,Ns) # half-offsets sampled (L=0 is nadir)

### Model parameters

m = (None, 1/3,0, 1/3,0.75)

### Init

MG = MaxwellGarnettColumn(H=H, L=L, Nl=Nl)
MG.set_params_solidice(*m[1:]) # disable densification and bubble anisotropy (=> solid anisotropic ice)

"""
Setup figure
"""

scale = 0.65
fig = plt.figure(figsize=(4.2*scale, 2.42*scale))
gs0 = fig.add_gridspec(1, 3, wspace=0.1, width_ratios=[1.3,0.7,1.7])
ax1 = fig.add_subplot(gs0[0, 0])
ax2 = fig.add_subplot(gs0[0, 1], sharey=ax1) 
ax3 = fig.add_subplot(gs0[0, 2], sharey=ax1) 
axes = (ax1,ax3)

FSLEG = 10.5

"""
Plot parameter profiles
"""

kw = dict(lw=1.15)
MG.plot_eigenvalues(ax1, cy='k', xticklbls=['0', '0.5', '1'], **kw)
ax1.legend(loc=1, frameon=False, fontsize=FSLEG, handlelength=0.8, handletextpad=0.15, labelspacing=0.04, bbox_to_anchor=(1.14,1.08))

zi = [0.05, 0.5, 0.95]
ax2.set_axis_off()
MG.plot_ODFcolumn(ax2, fig, zi, WH=0.225)

"""
Plot dTWTT
"""

kw_dTWTT = dict(fig=fig, cby=0.8, cbh=0.045, cbdw=0.17, fsc=FSLEG)
MG.plot_dTWTT(ax3, direct=True, **kw_dTWTT)

"""
Warp up and save plot
"""

for ii, ax in enumerate(axes):
    if ii>0: plt.setp(ax.get_yticklabels(), visible=False)
    else:    MG._set_yaxis(ax, yticks=["0","1"], labelpad=-3)
    
plt.savefig('CMP-solid.pdf', bbox_inches='tight', pad_inches=0.02)

