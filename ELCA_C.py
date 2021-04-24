# Optimized transit model with C, must create a new environment variable "ELCA_PATH"
import os
import copy
import ctypes
import numpy as np
import matplotlib.pyplot as plt

import ultranest
from plotting import corner

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

def getPhase(curTime, pPeriod, tMid):
    phase = (curTime - tMid) / pPeriod
    return phase - int(np.nanmin(phase))

# Function that bins an array
def binner(arr, n, err=''):
    if len(err) == 0:
        ecks = np.pad(arr.astype(float), (0, ((n - arr.size % n) % n)), mode='constant', constant_values=np.NaN).reshape(-1, n)
        arr = np.nanmean(ecks, axis=1)
        return arr
    else:
        ecks = np.pad(arr.astype(float), (0, ((n - arr.size % n) % n)), mode='constant', constant_values=np.NaN).reshape(-1, n)
        why = np.pad(err.astype(float), (0, ((n - err.size % n) % n)), mode='constant', constant_values=np.NaN).reshape(-1, n)
        weights = 1./(why**2.)
        # Calculate the weighted average
        arr = np.nansum(ecks * weights, axis=1) / np.nansum(weights, axis=1)
        err = np.array([np.sqrt(1. / np.nansum(1. / (np.array(i) ** 2.))) for i in why])
        return arr, err



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
        
        self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run()
        
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)
        
        for i, key in enumerate(freekeys):

            self.parameters[key] = self.results['maximum_likelihood']['point'][i]
            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]
        
        # self.results['maximum_likelihood']
        self.create_fit_variables()

    def plot_triangle(self):
        ranges = []
        mask1 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        mask2 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        mask3 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        titles = []
        labels= []
        flabels = {
            'rprs':r'R$_{p}$/R$_{s}$',
            'tmid':r'T$_{mid}$',
            'ars':r'a/R$_{s}$',
            'inc':r'I',
            'u1':r'u$_1$',
            'fpfs':r'F$_{p}$/F$_{s}$', 
            'omega':r'$\omega$',
            'ecc':r'$e$',
            'c0':r'$c_0$',
            'c1':r'$c_1$',
            'c2':r'$c_2$',
            'c3':r'$c_3$',
            'c4':r'$c_4$',
            'a0':r'$a_0$',
            'a1':r'$a_1$',
            'a2':r'$a_2$'
        }
        for i, key in enumerate(self.quantiles):
            titles.append(f"{self.parameters[key]:.5f} +- {self.errors[key]:.5f}")
            ranges.append([
                self.parameters[key] - 4*self.errors[key],
                self.parameters[key] + 4*self.errors[key]
            ])

            mask3 = mask3 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 3*self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 3*self.errors[key]) )

            mask1 = mask1 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + self.errors[key]) )

            mask2 = mask2 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 2*self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 2*self.errors[key]) )

            labels.append(flabels.get(key, key))

        chi2 = self.results['weighted_samples']['logl']*-2
        fig = corner(self.results['weighted_samples']['points'], 
            labels= labels,
            bins=int(np.sqrt(self.results['samples'].shape[0])), 
            range= ranges,
            #quantiles=(0.1, 0.84),
            plot_contours=True,
            levels=[chi2[mask1].max(), chi2[mask2].max(), chi2[mask3].max()],
            plot_density=False,
            titles=titles,
            data_kwargs={
                'c':chi2,
                'vmin':np.percentile(chi2[mask3],1),
                'vmax':np.percentile(chi2[mask3],99),
                'cmap':'viridis'
            },
            label_kwargs={
                'labelpad':15,
            },
            hist_kwargs={
                'color':'black',
            }
        )
        return fig

    def create_fit_variables(self):
        self.phase = getPhase(self.time, self.parameters['per'], self.parameters['tmid'])
        self.transit = transit(self.time, self.parameters)
        self.model = self.transit
        self.detrended = self.data 
        self.detrendederr = self.dataerr 
        self.residuals = self.data - self.model
        self.chi2 = np.sum(self.residuals**2/self.dataerr**2)
        self.bic = len(self.bounds) * np.log(len(self.time)) - 2*np.log(self.chi2)

    def plot_bestfit(self, nbins=10, phase=True, title=""):
        # import pdb; pdb.set_trace()

        f, (ax_lc, ax_res) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

        if phase:
            ax_res.set_xlabel('Phase')

            ecks = self.phase

        else:
            ax_res.set_xlabel('Time [day]')

            ecks = self.time

        # clip plot to get rid of white space
        ax_res.set_xlim([min(ecks), max(ecks)])
        ax_lc.set_xlim([min(ecks), max(ecks)])

        # making borders and tick labels black
        ax_lc.spines['bottom'].set_color('black')
        ax_lc.spines['top'].set_color('black')
        ax_lc.spines['right'].set_color('black')
        ax_lc.spines['left'].set_color('black')
        ax_lc.tick_params(axis='x', colors='black')
        ax_lc.tick_params(axis='y', colors='black')

        ax_res.spines['bottom'].set_color('black')
        ax_res.spines['top'].set_color('black')
        ax_res.spines['right'].set_color('black')
        ax_res.spines['left'].set_color('black')
        ax_res.tick_params(axis='x', colors='black')
        ax_res.tick_params(axis='y', colors='black')

        # residual plot
        ax_res.errorbar(ecks, self.residuals / np.median(self.data), yerr=self.detrendederr, color='gray',
                        marker='o', markersize=5, linestyle='None', mec='None', alpha=0.75)
        ax_res.plot(ecks, np.zeros(len(ecks)), 'r-', lw=2, alpha=1, zorder=100)
        ax_res.set_ylabel('Residuals')
        ax_res.set_ylim([-3 * np.nanstd(self.residuals / np.median(self.data)),
                         3 * np.nanstd(self.residuals / np.median(self.data))])

        correctedSTD = np.std(self.residuals / np.median(self.data))
        ax_lc.errorbar(ecks, self.detrended, yerr=self.detrendederr, ls='none',
                       marker='o', color='gray', markersize=5, mec='None', alpha=0.75)
        ax_lc.plot(ecks, self.transit, 'r', zorder=1000, lw=2)

        ax_lc.set_ylabel('Relative Flux')
        ax_lc.get_xaxis().set_visible(False)

        ax_res.errorbar(binner(ecks, len(self.residuals) // 10),
                        binner(self.residuals / np.median(self.data), len(self.residuals) // 10),
                        yerr=
                        binner(self.residuals / np.median(self.data), len(self.residuals) // 10, self.detrendederr)[
                            1],
                        fmt='s', ms=5, mfc='b', mec='None', ecolor='b', zorder=10)
        ax_lc.errorbar(binner(ecks, len(ecks) // 10),
                       binner(self.detrended, len(self.detrended) // 10),
                       yerr=
                       binner(self.residuals / np.median(self.data), len(self.residuals) // 10, self.detrendederr)[
                           1],
                       fmt='s', ms=5, mfc='b', mec='None', ecolor='b', zorder=10)

        # remove vertical whitespace
        f.subplots_adjust(hspace=0)

        return f,(ax_lc, ax_res)

if __name__ == "__main__":

    prior = { 
        'rprs':0.03,        # Rp/Rs
        'ars':14.25,        # a/Rs
        'per':3.336817,     # Period [day]
        'inc':86.5,        # Inclination [deg]
        'u1': 0.3, 'u2': 0.1, # limb darkening (linear, quadratic)
        'ecc':0,            # Eccentricity
        'omega':0,          # Arg of periastron
        'tmid':0.75         # time of mid transit [day]
    } 

    # GENERATE NOISY DATA
    time = np.linspace(0.65,0.85,500) # [day]
    data = transit(time, prior) + np.random.normal(0, 2e-4, len(time))
    dataerr = np.random.normal(300e-6, 50e-6, len(time))

    mybounds = {
        'rprs':[0,2*prior['rprs']],
        'tmid':[min(time),max(time)],
        'ars':[13,15], 
        #'inc':[85,87]
    }

    myfit = lc_fitter(time, data, dataerr, prior, mybounds)
    
    for k in myfit.bounds.keys():
        print("{:.6f} +- {}".format( myfit.parameters[k], myfit.errors[k]))

    fig,axs = myfit.plot_bestfit()
    plt.show()

    myfit.plot_triangle()
    plt.show()