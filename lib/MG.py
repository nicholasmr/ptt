#!/usr/bin/python3
# Nicholas M. Rathmann <rathmann@nbi.ku.dk>, 2025-

"""
Dielectric model for a firn-ice column, using:

    - a Maxwell--Garnett dielectric mixing model
    - a single stage Herron--Langway density model
    - a power law profile of bubble eccentricities (bubbles are standing ellipsoids, horizontally symmetric)
"""

import code # code.interact(local=locals())
import numpy as np
from scipy.optimize import fsolve, least_squares, minimize
from scipy import integrate

### Plotting

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams, rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True) 
rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,physics,siunitx} \usepackage{newtxtext,newtxmath}' # newtx is RSPA-like font

# white axis faces, transparent background
rcParams.update({
    "axes.facecolor":    (1, 1, 1, 1),
    "savefig.facecolor": (1, 1, 1, 0),
})

FSLEG = 11

"""
Constants
"""

# in vacuum
mu0  = 4*np.pi*1e-7
eps0 = 8.8541878188*1e-12
S0   = np.sqrt(mu0*eps0) 

# in ice
mu     = 1
epsa   = 3.16
deps   = 0.034
epsavg = epsa + deps/3 # epsa + (epsc-epsa)/3 = (2*epsa + epsc)/3
Savg   = np.sqrt(mu*epsavg) # isotropic slowness

# in air
eps_air = 1

"""
Parameter vectors and bounds
"""

m_null   = (0.75, 1/3,1/3, 1/3,1/3) # uninformed guess (zbco, lx0,lx1, lz0,lz1)
m_bounds = [(0.5, 0.95), (0,1),(0,1), (0,1),(0,1)] # upper and lower bounds on parameters

#m_bounds = [(0.5, 0.95), (0,1/3),(0,1/3), (1/3,1),(1/3,1)] # debug

"""
Maxwell--Garnett model
"""

class MaxwellGarnettColumn():

    m_null = m_null
    
    def __init__(self, H=500, L=np.linspace(0,500/2,10), Nl=50):
    
        self.H, self.L = H, L # ice thickness and half-offsets (in meters)
        self.Nl = Nl # number of vertical levels
        self.Ns = len(self.L) # number of CMP offsets

        self.z = np.linspace(0, 1, self.Nl) # normalized height (z=1 is the surface)
        self.x = self.L/H # normalized CMP half-offsets (x=0 is nadir)
        self.X, self.Z = np.meshgrid(self.x, self.z) # gridded
                
        self.alpha = np.pi/2 - np.arctan2(1-self.Z,self.X) # straight-ray CMP angle for a given x and z
        
        thres = np.deg2rad(89.99)
        self.alpha[self.alpha > thres] = thres
        self.alpha[self.alpha < 0.01] = np.deg2rad(0.01)
        
        self.alphadeg = np.rad2deg(self.alpha)        
        
    """
    Parameter profiles (density, eccentricity, fabric)
    """

    def set_params(self, zbco, lx0,lx1, lz0,lz1, rhos=0.35, e0=0.8, p=0.5, rhoheva=0.99):
        self.rhoh = self.rhoh_HL(zbco,rhos) # rho hat (relative rho)
        self.lami = self.lami_seriesexp(lx0,lx1, lz0,lz1)
        zeva = 0 if rhoheva is None else self.rhoh2z_HL(rhoheva, zbco,rhos) # z-coordinate for vanishing eccentricity (eva) given by where rhoh(z)=rhoheva
        self.e    = self.e_powerlaw(e0,p,zeva)
        self.Ni   = self.e2Ni(self.e)
        self.set_permittivities()
        self.set_eigenslownesses()
        self.zbco, self.rhos, self.e0, self.p = zbco, rhos, e0, p
        
    def set_params_solidice(self, *args):
        # set solid ice column
        self.rhoh = self.z*0 + 1 # solid ice
        if len(args) == 4 and (np.any(args is None) is not None): self.lami = self.lami_seriesexp(*args)
        self.e    = self.z*0
        self.Ni   = self.lami*0 + 1/3 # isotropic inclusions
        self.set_permittivities()
        self.set_eigenslownesses()
        self.zbco, self.rhos, self.e0, self.p = 0, 1, 0, 0
        
    def rhoh_HL(self, zbco, rhos, rhobco=0.81):
        # Single stage Herron--Langway normalized density profile
        alpha, beta = self.HL_alpha_beta(zbco, rhos, rhobco)
        alpha_z = np.multiply(alpha, 1-self.z)
        rhoh = np.divide(1, 1+np.exp(alpha_z+beta))
        return rhoh
        
    def HL_alpha_beta(self, zbco, rhos, rhobco):
        beta  = np.log(1/rhos-1)
        alpha = np.divide(np.log(1/rhobco-1)-beta, 1-zbco)
        return (alpha, beta)
        
    def rhoh2z_HL(self, rhoh, zbco, rhos, rhobco=0.81):
        # Get the depth of given relative density "rhoh" from single stage Herron--Langway model
        alpha, beta = self.HL_alpha_beta(zbco, rhos, rhobco)
        z = 1 - (np.log(1/rhoh-1)-beta)/alpha
        return z

    def lami_seriesexp(self, lx0,lx1, lz0,lz1):
        # Fabric eigenvalue profile (linearization of x- and z-component, whereas y-component is given by normalization)
        d = 1-self.z # depth
        lamx = lx0+(lx1-lx0)*d
        lamz = lz0+(lz1-lz0)*d
        lamy = 1-lamx-lamz
        lami = np.array([lamx,lamy,lamz])
        return lami
        
    def e_powerlaw(self, e0, p, z1, e1=1e-3):
        # Power law eccentricity profile
        if e0>1:  e0 = 1-1e-3
        if e0<e1: e0 = e1
        ecc = e0 + (e1 - e0) * ((1-self.z)/(1-z1))**p # =e0 at surface z=1
        ecc[self.z<z1] = e1 # below cutoff depth z1 is solid ice
        return ecc
        
    def e2Ni(self, ecc):
        # Depolarization tensor
        A = np.divide(1-np.power(ecc,2), 2*np.power(ecc,3))
        B = np.log(np.divide(1+ecc,1-ecc))-2*ecc
        Nz = np.multiply(A,B)
        Nx = (1-Nz)/2 # Nx=Ny
        Ni = np.array([Nx,Nx,Nz])
        return Ni
            
    """
    Permittivity
    """   
    
    def set_permittivities(self):

        ### Eigenpermittivities of *solid ice*
        self.eps_ice = epsa + self.lami*deps 
        
        ### Ice--air weighting function "phi" of Maxwell--Garnett model
        eps_rel = np.divide(eps_air, self.eps_ice) - 1
        N_rhoh = np.multiply(self.Ni, self.rhoh)
        denom = 1 + np.multiply(N_rhoh, eps_rel)
        phi = np.divide(1-self.rhoh, denom)
    
        ### Eigenpermittivities of *firn*
        self.eps_firn = self.eps_ice + np.multiply(phi, eps_air-self.eps_ice)
        
    """
    Wave propagation 
    """
    
    ### Slownesses

    def set_eigenslownesses(self):
        self.Sx, self.Sy, self.Sz = self.Si()
        self.Vx, self.Vy, self.Vz = 1/self.Sx, 1/self.Sy, 1/self.Sz # velocities
        self.set_th0()
        
    def Si(self, repeat_like=None): 
        Si = np.sqrt(mu*self.eps_firn)
        return Si # [i,z,th]

    def Sxz(self, th, ii=slice(None)):
        th = np.array(th, ndmin=2)
        xc = np.einsum('zj,z->zj', np.cos(th), np.array(self.Vx[ii], ndmin=1)) # x component
        zc = np.einsum('zj,z->zj', np.sin(th), np.array(self.Vz[ii], ndmin=1)) # z component
        Sxz = np.divide(1, np.sqrt(np.power(xc,2) + np.power(zc,2)))
        return Sxz[:,0] if th.shape[1]==1 else Sxz # [z,th]

    ### Propagation constants

    def py( self, th, ii=slice(None)): return np.sin(th) * self.Sy[ii] # p = n*sin(th) = c*S*sin(th)
    def pxz(self, th, ii=slice(None)): return np.sin(th) * self.Sxz(th, ii=ii) #self.Sxz(th)[ii]

    ### Snell angle solutions

    def thy( self, th_guess, p0, ii=slice(None)): return np.arcsin(np.divide(p0,self.Sy[ii])) # use analytical result for thy instead of numerical solver below
    #def thy( self, th_guess, p0, ii=slice(None)): return fsolve(lambda th, ii:  self.py(th, ii=ii) - p0, th_guess, args=(ii,))
    def thxz(self, th_guess, p0, ii=slice(None)): return fsolve(lambda th, ii: self.pxz(th, ii=ii) - p0, th_guess, args=(ii,))
    
    ### Propagation angle profiles
    
    def set_th0(self):
    
        def F(p0, ii, jj):
            z = self.H*self.z[ii:]
            u = np.divide(p0, self.Sy)[ii:]
            u[u>=1] = 1-1e-5 # fix bound because of poor guess
            f = np.divide(u, np.sqrt(1 - np.power(u,2)))
            #I = integrate.cumtrapz(f, z, initial=0, axis=0)[-1]
            I = integrate.trapz(f, z) # int_z^H f(u(z)') dz'
            F = self.L[jj] - I
            return F

        Sy0 = self.Sy[-1]
        p0_guess = np.sin(self.alpha)*Sy0 # straight CMP angle prop. const
#        p0_guess[:] = 1.2697 # best guess by trial and error for fsolve() instead of least_squares()
        
        p0 = np.array([ [least_squares(F, x0=p0_guess[ii,jj], args=(ii,jj)).x for ii in range(self.Nl)] 
                    for jj in range(self.Ns)])

        p0 = p0.squeeze()
        th0 = np.arcsin(p0/Sy0)
        self.th0 = th0.T
        self.th0deg = np.rad2deg(self.th0)

    def get_propagtionangles(self, th0, degree=True): 

        Itop = self.Nl-1
        
        if degree: th0 = np.deg2rad(th0)
        th0 = np.array(th0, ndmin=1)
        shape = (self.Nl, len(th0))
        thetay, thetaxz = np.zeros(shape), np.zeros(shape)
        thetay[Itop,:], thetaxz[Itop,:] = th0, th0

        dii, zrng = +1, np.arange(Itop-1,0-1,-1)
        for jj, _ in enumerate(th0):
            py0, pxz0 = self.py(_, ii=Itop), self.pxz(_, ii=Itop) # propagation constants at top level
            for ii in zrng:
                thetay[ii,jj]  = self.thy(  thetay[ii+dii,jj], py0,  ii=ii) 
                thetaxz[ii,jj] = self.thxz(thetaxz[ii+dii,jj], pxz0, ii=ii) 

        if degree: thetay, thetaxz = np.rad2deg(thetay), np.rad2deg(thetaxz)

        return (thetay[:,0], thetaxz[:,0]) if shape[1]==1 else (thetay, thetaxz) # [z,th0]
        
    """
    TWTT
    """

    def dTWTT(self, direct=False):

        th0 = self.alpha if direct else self.th0
 
        # prop constant at surface compatible with a given x-z combination
        p0 = np.sin(th0)*self.Sy[-1] # first index i is depth, second index j is horizontal distance
        
        # first index (k) is now how sin(thy) changes with depth for a given x-z prop constant at surface
        sin_thy = np.einsum('k,ij->kij', self.Vy, p0) # recall Vy = 1/Sy

        # corresponding prop angles with depth (first index) for a given x-z prop constant at surface
        th = np.arcsin(sin_thy) # virtually identical to thxz, so use this for both (much easier to calculate and therefore also speeds up inverse problems)        

        # geometric (oblique) integral factor
        g = np.sqrt(1+np.power(np.tan(th),2)) 

        # height (z) in m
        z = self.z*self.H
        
        Sy   = S0 * np.einsum('i,j->ij', self.Sy, np.ones(self.Ns)) # repeat (broadcast) Sy to all sections (repeat columns for each x)
        Sy_g = np.einsum('kij,ij->kij', g, Sy)

        # construct Sxz
        Vx = np.einsum('i,j->ij', self.Vx, np.ones(self.Ns)) # broadcast to all sections
        Vz = np.einsum('i,j->ij', self.Vz, np.ones(self.Ns)) # broadcast to all sections
        xc = np.einsum('kij,ij->kij', np.cos(th), Vx) # x component
        zc = np.einsum('kij,ij->kij', np.sin(th), Vz) # z component
        Sxz   = S0 * np.divide(1, np.sqrt(np.power(xc,2) + np.power(zc,2)))
        Sxz_g = np.einsum('kij,kij->kij', g, Sxz)

        dTWTT = np.zeros(self.X.shape)
        
        for ii in range(self.Nl-1):
            ty  = 2 * np.trapz( Sy_g[ii:,ii,:], x=z[ii:], axis=0)
            txz = 2 * np.trapz(Sxz_g[ii:,ii,:], x=z[ii:], axis=0)
            dTWTT[ii,:] = txz - ty
        dTWTT[-1,:] = dTWTT[-2,:] # fix top point where numerical integration gives zero

        return dTWTT, th0        

    """
    Inverse problem
    """
    
    def set_observed_dTWTT(self, x_true, z_true, dTWTT_true):
        # Do this before running inversion
        self.x_true     = x_true.flatten() 
        self.z_true     = z_true.flatten()
        self.dTWTT_true = dTWTT_true.flatten()
        self.Npts_true = np.sum(~np.isnan(self.dTWTT_true)) # number of sampling points in acquisition space
    
    def J(self, dTWTT):
        # Misfit measure (objective function)
        J_misfit = 0
        for kk in range(len(self.z_true)):
            ii = np.argmin(np.abs(self.z_true[kk] - self.z)) # nearest z-pos in model grid
            jj = np.argmin(np.abs(self.x_true[kk] - self.x)) # nearest x-pos in model grid
            error = dTWTT[ii,jj]-self.dTWTT_true[kk]
            if not np.isnan(error):
                J_misfit += np.power(error, 2)
        J = 1e9*np.sqrt(J_misfit)/self.Npts_true # regularization not found to be needed
        return J
    
    def infer_params(self, *obs, kw_pp=dict(), \
                            m_guess=m_null, \
                            tol=1e-3 # stop at this relative decrease in J (found to suffice by trial and error)
        ):
        
        self.set_observed_dTWTT(*obs)
        
        def cost(m):
            self.set_params(*m, **kw_pp)
            dTWTT, *_ = self.dTWTT()
            J = self.J(dTWTT)
            print('J=%.2e // %s'%(J, self.mstr(*m)))
            return J
            
        print('\n*** m_guess: %s'%(self.mstr(*m_guess)))

        # Add constraints to help infer girdle fabrics and horizontal pole, assuming fabrics generally strengthen with depth
        Ix0,Ix1, Iz0,Iz1 = 1,2, 3,4
        m_constraints = [
            {'type': 'ineq', 'fun': lambda m: (m[Iz1]>m[Ix1])*(m[Iz1]-m[Iz0]) }, # if lz1>lx1 (vertical pole) then constrain lz1>lz0 (vertical pole strengthens with depth) 
#            {'type': 'ineq', 'fun': lambda m: (m[Iz1]>m[Ix1])*(m[Iz0]-m[Ix0]) }, # if lz1>lx1 (vertical pole) then constrain lz0>lx0 (eigenvalue profiles don't cross) 
        ]
        
        res = minimize(cost, m_guess, method='SLSQP', tol=tol, bounds=m_bounds, constraints=m_constraints)
        m_infr = res.x
        
        return m_infr

    def mstr(self, zbco, lx0,lx1, lz0,lz1):
        return 'zbco=%.2f :: lx0=%.2f, lx1=%.2f :: lz0=%.2f, lz1=%.2f'%(zbco, lx0,lx1, lz0,lz1)

    """
    Plotting
    """
    
    def plot_density(self, ax, xticklbls=["0.3", "1"], c='k', lw=1, dx=0.1, xlim=[0.3, 1.01], **kw):
        ax.plot(self.rhoh, self.z, c=c, lw=lw, **kw)
        ax.set_xticks([float(_) for _ in xticklbls])
        ax.set_xticklabels(xticklbls)
        ax.set_xticks(np.arange(0.0, 1+dx, dx), minor=True)
        ax.set_xlim(xlim)
        ax.set_xlabel(r'$\hat{\rho}$')
        
    def plot_eccentricity(self, ax, xticklbls=["0", "1"], c='k', lw=1, dx=0.2, xlim=[-0.01,1], **kw):
        ax.plot(self.e, self.z, lw=lw, color=c, **kw)
        ax.set_xticks([float(_) for _ in xticklbls])
        ax.set_xticklabels(xticklbls)
        ax.set_xlim(xlim)
        ax.set_xticks(np.arange(0,1+dx,dx), minor=True)
        ax.set_xlabel(r'$e$')
        
    def plot_eigenvalues(self, ax, xticklbls=["0", "0.5", "1"], c='k', cy='0.6', lw=1, dx=0.1, xlim=[0,1], noy=False, **kw):
        ax.plot(self.lami[2], self.z, c=c,  ls='--', lw=lw, label=r'$\lambda_{z}$', **kw)
        ax.plot(self.lami[0], self.z, c=c,  ls='-',  lw=lw, label=r'$\lambda_{x}$', **kw)
        if not noy: ax.plot(self.lami[1], self.z, c=cy, ls=':',  lw=lw, label=r'$\lambda_{y}$', **kw)
        ax.set_xticks([float(_) for _ in xticklbls])
        ax.set_xticklabels(xticklbls)
        ax.set_xticks(np.arange(0,1+dx,dx), minor=True)
        ax.set_xlim(xlim)
        ax.set_xlabel(r'$\lambda_i$')
        
    def _set_map_axes(self, fig,ax,h, xticklbls,dx, clabel,cby,cbh,cbdw):
        ax.set_xlabel(r'$L/H$')
        ax.set_xticks([float(_) for _ in xticklbls])
        ax.set_xticklabels(xticklbls)
        ax.set_xticks(np.arange(0,1+dx,dx), minor=True)     
        if fig is not None:    
            p = ax.get_position().get_points()
            px0, px1 = p[0][0], p[1][0]
            w = px1-px0
            dw = cbdw*w
            cax = fig.add_axes([px0 + dw, cby, w-2*dw, cbh])
            hcb = plt.colorbar(h, cax=cax, pad=0.15, orientation='horizontal')
            hcb.set_label(clabel, labelpad=+1)
            for label in hcb.ax.xaxis.get_ticklabels()[1::2]: label.set_visible(False)
        
    def _set_yaxis(self, ax, yticks=["0","1"], labelpad=-3):
        # used in paper plots
        ax.set_yticks([float(_) for _ in yticks])
        ax.set_yticklabels(yticks)
        ax.set_yticks(np.arange(0,1.1,0.1), minor=True)
        ax.set_ylim([0,1])
        ax.set_ylabel(r'$z/H$', labelpad=labelpad)
        
    def plot_ang(self, ax, F, \
                    clabel='angle', cmap='PiYG_r', fsc=11, ml=None,  \
                    fig=None, cby=-0.23, cbh=0.04, cbdw=0.005, \
                    lvl0=10, lvl1=85+1, dlvl=15, \
                    xticklbls=["0","0.5","1"], dx=0.1): 
    
        lvls = np.arange(lvl0, lvl1+1, dlvl)
        h = ax.contourf(self.X, self.Z, F, levels=lvls, cmap=cmap, extend='both')
        CS = ax.contour(self.X, self.Z, F, lvls[0::1], linewidths=[0.7,], colors=['k',])
        ax.clabel(CS, fontsize=fsc, inline=1, rightside_up=True, fmt=r'$%i^\circ$', manual=ml)
        self._set_map_axes(fig,ax,h, xticklbls,dx, clabel,cby,cbh,cbdw)         
        return h, CS
        
    def plot_dTWTT(self, ax, \
                    direct=False, \
                    clabel=r'$\Delta\tau_{xz,y}$ (ns)', cmap='coolwarm_r', fsc=11, \
                    fig=None, cby=-0.23, cbh=0.04, cbdw=0.005, \
                    lvl0=-5, lvl1=5, dlvl=1, \
                    xticklbls=["0","0.5","1"], dx=0.1): 
    
        F, th0 = self.dTWTT(direct=direct)
        F *= 1e9 # to ns
        lvls = np.arange(lvl0, lvl1+1, dlvl)
        h = ax.contourf(self.X, self.Z, F, levels=lvls, cmap=cmap, extend='both')
        self._set_map_axes(fig,ax,h, xticklbls,dx, clabel,cby,cbh,cbdw) 
        return h
        
    def plot_dTWTT_expr(self, fig,ax, X,Z,dTWTT, label=None, scale=1e9, lvls=np.arange(-5, 5+1, 1), \
            caxpos=None, lbl_bbox=(1.17,1.23), clbl='k'
        ): 

        dTWTT *= scale # to ns
        h = ax.contourf(X, Z, dTWTT, levels=lvls, cmap='coolwarm_r', extend='both')
        
        ax.set_xlabel(r'$L/H$')
        ax.set_xticks([0,0.5,1])
        ax.set_xticklabels(["0","0.5","1"])   
        ax.set_xticks(np.arange(0,1+1e-1,0.1), minor=True)
           
        if caxpos is not None:
            cax = fig.add_axes(caxpos)
            hcb = plt.colorbar(h, cax=cax, pad=0.15, orientation='horizontal')
            hcb.ax.xaxis.set_ticks(lvls[1::4])
            hcb.ax.xaxis.set_ticks(lvls[1::2], minor=True)
            hcb.set_label(r'$\Delta\tau_{xz,y}$ (ns)', labelpad=+1)
            
        if label is not None:
            kwt = dict(loc=1, frameon=False, prop=dict(size=FSLEG-0.75, color=clbl), bbox_to_anchor=lbl_bbox)
            ax.add_artist(AnchoredText(r'\textit{%s}'%(label), bbox_transform=ax.transAxes, **kwt))
            
        return h
        
    def plot_ODFcolumn(self, ax, fig, zi, WH=0.2, axposx=0.45, lvlmax=0.35):

            from specfabpy import specfab as sf
            from specfabpy import plotting as sfplt

            geo, prj = sfplt.getprojection(rotation=+30-90, inclination=50)
            lm, nlm_len = sf.init(2)

            def plotODF(ax, nlm, axpos0=(0,0)):
                trans = ax.transData.transform(axpos0)
                trans = fig.transFigure.inverted().transform(trans)
                W = H = WH
                axpos = [trans[0]-W/2, trans[1]-H/2, W,H]
                axin = plt.axes(axpos, projection=prj)
                axin.set_global()
                lvlset = [np.linspace(0.0, lvlmax, 5), lambda x,p:'%.1f'%x]
                sfplt.plotODF(nlm, lm, axin, lvlset=lvlset, cmap='Greys', showcb=False, nchunk=None)
                sfplt.plotcoordaxes(axin, geo, negaxes=True, color=sfplt.c_dred, axislabels='vuxi')
                return axin

            for z in zi: # plot at these depth intevals
                I = np.nanargmin(np.abs(self.z - z))
                nlm = sf.a2_to_nlm(np.diag(self.lami[:,I]))
                axin = plotODF(ax, nlm, axpos0=(axposx, z))

   
if __name__ == '__main__':

    print('*** Dielectric constants used ***')
    
    print('epsavg = %.3f'%(epsavg))
    print('deps/epsavg = %.3f'%(deps/epsavg))
    epsrel_max = (1+deps/epsavg*(1-1/3))
    epsrel_min = (1+deps/epsavg*(0-1/3))
    print('max/min of epsi/epsavg = %.3f / %.3f'%(epsrel_max,epsrel_min))
    Vi_max, Vi_min = 1/np.sqrt(epsrel_max), 1/np.sqrt(epsrel_min)
    print('max/min of V_i = %.3f / %.3f'%(Vi_max,Vi_min))
    print('max/min of S_i = %.3f / %.3f'%(1/Vi_max,1/Vi_min))
    
