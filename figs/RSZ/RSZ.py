#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2025-

"""
Calculate propagation angles, refractive shadow zone (RSZ), and dTWTT for an idealized column
"""

import numpy as np
import sys, code # code.interact(local=locals())
sys.path.append("../../lib")
from MG import *

""" 
Initialize model
"""

Nl, Ns = 50, 30

### Geometry

H = 250 # ice thickness (m)
L = np.linspace(0,H,Ns) # half-offsets sampled (L=0 is nadir)

### Model parameters

m = (0.8, 1/3,0, 1/3, 0.75)

### Init

MG = MaxwellGarnettColumn(H=H, L=L, Nl=Nl)
MG.set_params(*m)

### Debug by enabling/disabling different parts of the PP profiles

#MG.set_params(*m, e0=0) # disable bubble (structural) anisotropy
#MG.set_params_solidice(*[1/3]*4) # disable fabric anisotropy
#MG.set_params_solidice(*m[1:]) # disable densification AND bubble anisotropy (solid, anisotropic ice)

"""
Setup figure
"""

scale = 0.65
fig = plt.figure(figsize=(10.5*scale, 2.5*scale))

k = 2.05
gs0 = fig.add_gridspec(1, 7, wspace=0.15, width_ratios=[1,1,1.7, k,k,k,k])
ax1 = fig.add_subplot(gs0[0, 0])
axes = [ax1,]
axes += [fig.add_subplot(gs0[0, 1+_], sharey=ax1) for _ in range(6)]
(ax1,ax2,ax3,ax4,ax5,ax6,ax7) = axes

FSLEG = 10

"""
Plot parameter profiles
"""

kw = dict(lw=1.25)
MG.plot_density(ax1, **kw)
MG.plot_eccentricity(ax2, **kw)
MG.plot_eigenvalues(ax3, cy='k', **kw)
ax3.legend(loc=1, frameon=False, fontsize=FSLEG, handlelength=1.0, handletextpad=0.2, labelspacing=0.04, bbox_to_anchor=(1.16,1.08))

kw = dict(fontsize=FSLEG-0.5, rotation=90, ha='left', va='bottom', ma='center')
ax1.text(0.5, 0.1, r'$\hat{z}_{\mathrm{bco}}=%.1f$'%(MG.zbco), **kw)
ax2.text(0.2, 0.1, "$e_0=%.1f$\n$p=%.1f$"%(MG.e0,MG.p), **kw)

"""
Plot angles
"""

kw_ang = dict(fig=fig, cby=-0.23, cbh=0.04, fsc=FSLEG-1)

ml = [(x, 0.1+0.5*x) for x in np.linspace(0.1, 0.8, 6)]
MG.plot_ang(ax4, MG.th0deg, clabel=r'$\theta_0$ $(^\circ)$', ml=ml, **kw_ang)

y0 = 0.0
yml = lambda x: y0 + (0.8-y0)*(x*1.4)**1
xv = np.linspace(0.15, 0.85, 6)
ml = [(x, yml(x)) for x in xv] # manual locations
MG.plot_ang(ax6, MG.alphadeg, clabel=r'$\theta_0$ $(^\circ)$', ml=ml, **kw_ang)

ax4.text(0.96, 0.975, "refractive\nshadow\nzone", c='0.97', fontsize=FSLEG-1, ha='right', va='top', ma='right')

"""
Plot dTWTT
"""

kw_dTWTT = dict(fig=fig, cby=kw_ang['cby'], cbh=0.0425, fsc=kw_ang['fsc'])

MG.plot_dTWTT(ax5, **kw_dTWTT)
MG.plot_dTWTT(ax7, direct=True, **kw_dTWTT)

labels = [r"Bending ray",]*2 + ["Straight ray",]*2
for ii,ax in enumerate((ax4,ax5,ax6,ax7)):
    ax.text(0.98, 1.03, r'\textit{%s}'%(labels[ii]), fontsize=FSLEG-0.5, ha='right', va='bottom')

for ax in (ax5,ax7):
    c = "0.15"
    kw = dict(fontsize=FSLEG-1, ha='center', ma='center', c=c)
    ax.text(0.5, 0.55, "bubble\n anisotropy", va='bottom', **kw)
    ax.text(0.5, 0.45, "fabric\n anisotropy", va='top', **kw)
    kwarrprop = dict(arrowstyle="-|>", fc=c, ec=c)
    ax.annotate("", xytext=(0.5, 0.75), xy=(0.7, 0.9), arrowprops=kwarrprop)
    ax.annotate("", xytext=(0.5, 0.25), xy=(0.3, 0.1), arrowprops=kwarrprop)

"""
Warp up and save plot
"""

for ii, ax in enumerate(axes):
    if ii>0: plt.setp(ax.get_yticklabels(), visible=False)
    else:    MG._set_yaxis(ax)
        
    kw = dict(loc=2, frameon=False, prop=dict(size=11), bbox_to_anchor=(-0.4 +(ii>=2)*0.15, 1.25), bbox_transform=ax.transAxes)
    at = AnchoredText(r'{(%s)}'%(chr(ord('a')+ii)), **kw)
    ax.add_artist(at)
    
plt.savefig('RSZ.pdf', dpi=200, bbox_inches='tight', pad_inches=0.02)

