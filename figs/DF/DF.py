#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2025-

"""
Maxwell--Garnett dielectric model applied to firn cores from Dome Fuji
"""

import numpy as np
import pandas as pd
import sys, code # code.interact(local=locals())
sys.path.append("../../lib")
from MG import *

"""
Dome Fuji shallow cores
"""

Hfirn = 115 # firn thickness

files = ('DFS10.csv', 'DF93.csv')

def get_obs(f):
    df = pd.read_csv('../../data/Dome-Fuji/%s'%(f), delim_whitespace=True)
    dfm = df.groupby(['SampleID']).mean()
    depth = dfm['Depth'].to_numpy()
    DA_avg = dfm['DA'].to_numpy()
    DA_std = df.groupby(['SampleID']).std()['DA'].to_numpy()
    drel = abs(depth/Hfirn)
    return (drel, DA_avg, DA_std)

### PP profiles

# Best-fit parameters for DFS10 and DF93 cores
ecc0 = [0.8, 0.725]
eccp = [0.5, 0.9]

def get_MG(ii):
    MG = MaxwellGarnettColumn(Nl=100)
    MG.set_physprops(0, 1/3,0.25, 1/3,0.5, e0=ecc0[ii], p=eccp[ii], rhoheva=None)
    MG.rhoh, MG.rhoh_err = f_rhoh(MG.z)
    MG.rhoh_obs = MG.rhoh.copy()
    MG.set_permittivities()
    return MG

def f_rhoh(z):
    df = pd.read_csv('../../data/Dome-Fuji/Density_DFS2010_core.csv', header=1, names=['depth','N','rho'])
    df['z'] = (1-df['depth'].to_numpy()/Hfirn) # relative height in firn column
    df['rhoh'] = df['rho'].to_numpy()/917 # relative density
    znew = np.concatenate(([-0.05,],z)) # placeholder entry to ensure correct dimensions 
    groups = df['rhoh'].groupby(pd.cut(df['z'].to_numpy(), znew))
    rhoh_mean, p0, p1 = groups.mean().to_numpy(), groups.quantile(q=0).to_numpy(), groups.quantile(q=1).to_numpy()
    rhoh_err = [rhoh_mean-p0,p1-rhoh_mean]
    return rhoh_mean, rhoh_err

MG = []
MG.append(get_MG(0)) # DFS10
MG.append(get_MG(1)) # DF93
z = MG[0].z # shorthand

"""
Figure setup and style
"""

scale = 0.65
fig = plt.figure(figsize=(10.5*scale, 3.0*scale))
gs0 = fig.add_gridspec(1, 6, wspace=0.15, width_ratios=[1.25,1.25,1,0.85,0.8,1])
ax1 = fig.add_subplot(gs0[0, 0])
ax2 = fig.add_subplot(gs0[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs0[0, 2], sharey=ax1)
ax4 = fig.add_subplot(gs0[0, 3], sharey=ax1)
ax5 = fig.add_subplot(gs0[0, 4], sharey=ax1)
ax6 = fig.add_subplot(gs0[0, 5], sharey=ax1)

# Styles
cred, cblue = "#ef3b2c", "#4eb3d3"
ci = (cred, cblue)
lw = 1.4
FSLEG = 9.5
kwleg = dict(frameon=False, fontsize=FSLEG, labelspacing=0.2, handlelength=1, handletextpad=0.3)

"""
PP profiles
"""

labels = [files[ii][:-4] for ii in range(2)]

### Eigenvalues

ax = ax5
MG[0].plot_eigenvalues(ax, xticklbls=["0.3", "0.5"], xlim=[0.25,0.5], lw=lw)
ax.legend(loc="lower center", bbox_to_anchor=(0.50,-0.06), **kwleg)

### Eccentricity

ax = ax3
kw = dict(lw=lw, xticklbls=["0", "0.5", "1"])
MG[0].plot_eccentricity(ax, c=ci[0], label=labels[0], **kw)
MG[1].plot_eccentricity(ax, c=ci[1], label=labels[1], **kw)
ax.legend(loc=4, bbox_to_anchor=(1.11,-0.06), **kwleg)
kw = dict(rotation=68, fontsize=FSLEG-0.5, ha='right', multialignment='center')
ax.text(0.51, 1-0.32, '$p=%.1f$\n$e_0=%.2f$'%(eccp[0],ecc0[0]), color=ci[0], **kw)
ax.text(0.98, 1-0.46, '$p=%.1f$\n$e_0=%.2f$'%(eccp[1],ecc0[1]), color=ci[1], **kw)

### Snell angles 

cy  = ['#33a02c', '#ff7f00', '#6a3d9a']
cxz = ['#b2df8a', '#fdbf6f', '#cab2d6']
ls  = ['-', '--']

th0 = np.array([30, 50, 70]) # initial angle (colatitude) at top layer

ax = ax6

for ii, _ in enumerate(th0):
    thy, thxz = MG[0].get_propagtionangles(_, degree=True)
    ax.plot(thy,  z, c=cy[ii],  lw=lw, ls=ls[0], label=r'$%i^\circ$'%(_))
    ax.plot(thxz, z, c=cxz[ii], lw=lw, ls=ls[1]) 
    ax.text(_, 1, r'$%i^{\circ}$'%(_), color=cy[ii], ha='center', va='bottom', fontsize=FSLEG)

ax.text(th0[1], 1.1, r'$\theta_{0}=$', ha='center', va='bottom', fontsize=FSLEG)
ax.text(th0[-1]-12, 0.02, r'$\uparrow$ toward nadir', ha='left', va='bottom', rotation=90, fontsize=FSLEG-0.5, c='k')

ax.set_xlabel(r'$\theta$')
thmax = th0[-1]+0
ax.set_xticks(np.arange(10,thmax+1,20))
ax.set_xticks(np.arange(0,thmax+1,10), minor=True)
ax.set_xlim([0,thmax])
ax.grid(axis='x', ls=':')

"""
Observations and model results
"""

### Dielectric anisotropy 

axi = (ax1,ax2)

for ii, f in enumerate(files):

    ax = axi[ii]

    # Model

    delta = MG[ii].eps_firn[2] - MG[ii].eps_firn[0]
    ax.plot(delta, z, zorder=10, label='Model', lw=lw+0.35, color=ci[ii])

    if 1:
        MG[ii].set_physprops_solidice() # if firn was solid ice...
        delta = MG[ii].eps_firn[2] - MG[ii].eps_firn[0]
        ax.plot(delta, z, zorder=10, label='Solid ice', lw=lw+0.25, color=ci[ii], ls=':')

    # Observed
        
    (drel, DA_avg, DA_std) = get_obs(f)
    kw_errbar = dict(fmt='o', c='k', markersize=2.5, lw=0.7, ecolor='0.5',)
    ax.errorbar(DA_avg, 1-drel, xerr=DA_std, label=f[:-4], **kw_errbar)

    x0 = -0.01
    ax.set_xticks(np.arange(x0,0.1,0.02))
    ax.set_xticks(np.arange(x0,0.1,0.01), minor=True)
    ax.set_xlim([0,0.06])
    ax.set_xlabel(r'$(\epsilon_z-\epsilon_h)/\epsilon_0$')
    
    ax.legend(loc=4, bbox_to_anchor=(1.09,-0.05), **kwleg)
    
### Observed density

ax = ax4    
ax.errorbar(MG[0].rhoh_obs, z, xerr=MG[0].rhoh_err, label='DFS10', **kw_errbar)
ax.legend(loc=4, bbox_to_anchor=(1.06,-0.05), **kwleg)
xticklbls = ["0.4","1"]
ax.set_xticks([float(_) for _ in xticklbls])
ax.set_xticklabels(xticklbls)
dx=0.1
ax.set_xticks(np.arange(0.0, 1+dx, dx), minor=True)
ax.set_xlim([0.3, 1.01])
ax.set_xlabel(r'$\hat{\rho}$')
    

"""
Warp up and save plot
"""

for ii, ax in enumerate((ax1,ax2,ax3,ax4,ax5,ax6)):

    if ii>0: 
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set_yticks(np.arange(0,1.1,0.5))
        ax.set_yticks(np.arange(0,1.1,0.1), minor=True)
        ax.set_ylim([0,1])
        ax.set_ylabel(r'$z/H$')

for ii, ax in enumerate((ax1,ax2,ax3,ax4,ax5,ax6)):
    kw = dict(loc=2, frameon=False, prop=dict(size=11), bbox_to_anchor=(-0.15 -(ii>=2)*0.1, 1.22), bbox_transform=ax.transAxes)
    ax.add_artist(AnchoredText(r'{(%s)}'%(chr(ord('a')+ii)), **kw))
    
plt.savefig('DF.png', dpi=200, bbox_inches='tight', pad_inches=0.02)

