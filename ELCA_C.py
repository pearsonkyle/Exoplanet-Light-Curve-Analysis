# Optimized transit model with C, must create a new environment variable "ELCA_PATH"
import os
import copy
import ctypes
import numpy as np
import matplotlib.pyplot as plt

import dynesty
from dynesty import plotting
from dynesty.utils import resample_equal

from scipy.ndimage import gaussian_filter as norm_kde
from scipy.stats import gaussian_kde
########################################################
# LOAD IN TRANSIT FUNCTION FROM C

# define 1d array pointer in python
array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

# load library
lib_trans = np.ctypeslib.load_library('lib_transit.so',
    os.path.join(os.path.dirname(os.path.realpath(__file__)),'C_sharedobject')
)

# load fn from library and define inputs
occultquadC = lib_trans.occultquad

# inputs
occultquadC.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, array_1d_double ]

# no outputs, last *double input is saved over in C
occultquadC.restype = None
########################################################

def time_bin(time, flux, dt=1./(60*24)):
    bins = int(np.floor((max(time) - min(time))/dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    for i in range(bins):
        mask = (time >= (min(time)+i*dt)) & (time < (min(time)+(i+1)*dt))
        if mask.sum() > 0:
            bflux[i] = np.nanmean(flux[mask])
            btime[i] = np.nanmean(time[mask])
    zmask = (bflux==0) | (btime==0) | np.isnan(bflux) | np.isnan(btime)
    return btime[~zmask], bflux[~zmask]

def transit(t, values):
    time = np.require(t,dtype=ctypes.c_double,requirements='C')
    model = np.zeros(len(t),dtype=ctypes.c_double)
    model = np.require(model,dtype=ctypes.c_double,requirements='C')
    keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
    vals = [values[k] for k in keys]
    occultquadC( time, *vals, len(time), model )
    return model

class lc_fitter(object):

    def __init__(self, time, data, dataerr, prior, bounds):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.fit_nested()

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        model = np.zeros(len(self.time),dtype=ctypes.c_double)
        model = np.require(model,dtype=ctypes.c_double,requirements='C')

        def loglike(pars):
            # chi-squared
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            
            # explicit function call: occultquadC( t,rp,ar,per,inc,u1,u2,ecc,ome,tm, n,model )
            keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            occultquadC( time, *vals, len(time), model )
            # saves lightcurve to model

            return -0.5 * np.sum( ((self.data-model)/self.dataerr)**2 )
        
        def prior_transform(upars):
            # transform unit cube to prior volume
            return (boundarray[:,0] + bounddiff*upars)
        
        # TODO try 
        dsampler = dynesty.DynamicNestedSampler(
            loglike, prior_transform,
            ndim=len(freekeys), bound='multi', sample='unif', 
            maxiter_init=5000, dlogz_init=1, dlogz=0.05,
            maxiter_batch=100, maxbatch=10, nlive_batch=100
        )
        dsampler.run_nested()
        self.results = dsampler.results

        # alloc data for best fit + error
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        tests = [copy.deepcopy(self.prior) for i in range(6)]

        # Derive kernel density estimate for best fit
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        samples = self.results['samples']
        logvol = self.results['logvol']
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], 1000)  # resample
        wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
        self.weights = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate

        # errors + final values
        mean, cov = dynesty.utils.mean_and_cov(self.results.samples, weights)
        mean2, cov2 = dynesty.utils.mean_and_cov(self.results.samples, self.weights)
        for i in range(len(freekeys)):
            self.errors[freekeys[i]] = cov[i,i]**0.5
            tests[0][freekeys[i]] = mean[i]
            tests[1][freekeys[i]] = mean2[i]

            counts, bins = np.histogram(samples[:,i], bins=100, weights=weights)
            mi = np.argmax(counts)
            tests[5][freekeys[i]] = bins[mi] + 0.5*np.mean(np.diff(bins))

            # finds median and +- 2sigma, will vary from mode if non-gaussian
            self.quantiles[freekeys[i]] = dynesty.utils.quantile(self.results.samples[:,i], [0.025, 0.5, 0.975], weights=weights)
            tests[2][freekeys[i]] = self.quantiles[freekeys[i]][1]

        # find minimum near weighted mean
        mask = (samples[:,0] < self.parameters[freekeys[0]]+2*self.errors[freekeys[0]]) & (samples[:,0] > self.parameters[freekeys[0]]-2*self.errors[freekeys[0]])
        bi = np.argmin(self.weights[mask])

        for i in range(len(freekeys)):
            tests[3][freekeys[i]] = samples[mask][bi,i]
            tests[4][freekeys[i]] = np.average(samples[mask][:,i],weights=self.weights[mask],axis=0)

        # find best fit
        chis = []
        res = []
        for i in range(len(tests)):
            lightcurve = transit(self.time, tests[i])
            residuals = self.data - lightcurve
            res.append(residuals)
            btime, br = time_bin(self.time, residuals)
            blc = transit(btime, tests[i])
            mask = blc < 1
            duration = btime[mask].max() - btime[mask].min()
            tmask = ((btime - tests[i]['tmid']) < duration) & ((btime - tests[i]['tmid']) > -1*duration)
            chis.append( np.mean(br[tmask]**2) )

        mi = np.argmin(chis)
        self.parameters = copy.deepcopy(tests[mi])
        # plt.scatter(samples[mask,0], samples[mask,1], c=weights[mask]); plt.show()

        # final model
        self.model = transit(self.time, self.parameters)
        self.residuals = self.data - self.model

    def plot_bestfit(self):
        f = plt.figure( figsize=(12,7) )
        #f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid( (4,5), (0,0), colspan=5,rowspan=3 )
        ax_res = plt.subplot2grid( (4,5), (3,0), colspan=5, rowspan=1 )
        axs = [ax_lc, ax_res]

        axs[0].errorbar(self.time, self.data, yerr=self.dataerr, ls='none', marker='o', color='black', zorder=1, alpha=0.5)
        axs[0].plot(self.time, self.model, 'r-', zorder=2)
        axs[0].set_xlabel("Time [day]")
        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')

        axs[1].plot(self.time, self.residuals*1e6, 'ko')
        axs[1].set_xlabel("Time [day]")
        axs[1].set_ylabel("Residuals [ppm]")
        plt.tight_layout()

        return f,axs

if __name__ == "__main__":

    prior = { 
        'rprs':0.013,        # Rp/Rs
        'ars':14.25,        # a/Rs
        'per':3.336817,     # Period [day]
        'inc':87.5,        # Inclination [deg]
        'u1': 0.3, 'u2': 0.01, # limb darkening (linear, quadratic)
        'ecc':0,            # Eccentricity
        'omega':0,          # Arg of periastron
        'tmid':0.75         # time of mid transit [day]
    } 

    # GENERATE NOISY DATA
    time = np.linspace(0.65,0.85,500) # [day]
    data = transit(time, prior) + np.random.normal(0, 2e-4, len(time))
    dataerr = np.random.normal(300e-6, 50e-6, len(time))

    #dude()

    mybounds = {
        'rprs':[0,2*prior['rprs']],
        'tmid':[min(time),max(time)],
        #'ars':[13,15], 
        #'inc':[87,88]
    }

    myfit = lc_fitter(time, data, dataerr, prior, mybounds)
    
    for k in myfit.bounds.keys():
        print("{:.6f} +- {}".format( myfit.parameters[k], myfit.errors[k]))

    fig,axs = myfit.plot_bestfit()

    # triangle plot
    fig,axs = dynesty.plotting.cornerplot(myfit.results, labels=list(myfit.bounds.keys()), quantiles_2d=[0.4,0.85], smooth=0.015, show_titles=True,use_math_text=True, title_fmt='.2e',hist2d_kwargs={'alpha':1,'zorder':2,'fill_contours':False})
    dynesty.plotting.cornerpoints(myfit.results, labels=list(myfit.bounds.keys()), fig=[fig,axs[1:,:-1]],plot_kwargs={'alpha':0.1,'zorder':1,} )
    plt.tight_layout()
    plt.show()