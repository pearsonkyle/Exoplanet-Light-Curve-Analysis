# Uses Gaussian Kernel Regression to handle errors correlated to centroid position
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
from scipy import spatial

import requests
import re
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

phaseCurve = lib_trans.phasecurve
phaseCurve.argtypes = [array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, array_1d_double ]
phaseCurve.restype = None  
########################################################

def phasecurve(t, values):
    time = np.require(t,dtype=ctypes.c_double,requirements='C')
    model = np.zeros(len(t),dtype=ctypes.c_double)
    model = np.require(model,dtype=ctypes.c_double,requirements='C')
    keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
    vals = [values[k] for k in keys]
    cvals = [values[k] for k in ['c0','c1','c2','c3','c4']]
    phaseCurve( time, *cvals, *vals, len(time), model)
    return model

def transit(t, values):
    time = np.require(t,dtype=ctypes.c_double,requirements='C')
    model = np.zeros(len(t),dtype=ctypes.c_double)
    model = np.require(model,dtype=ctypes.c_double,requirements='C')
    keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
    vals = [values[k] for k in keys]
    occultquadC( time, *vals, len(time), model )
    return model

def weightedflux(flux,gw,nearest):
    return np.sum(flux[nearest]*gw,axis=-1)

def gaussian_weights(X, w=None, neighbors=50, feature_scale=1000):
    if isinstance(w, type(None)): w = np.ones(X.shape[1])
    Xm = (X - np.median(X,0))*w
    kdtree = spatial.cKDTree(Xm*feature_scale)
    nearest = np.zeros((X.shape[0],neighbors))
    gw = np.zeros((X.shape[0],neighbors),dtype=float)
    for point in range(X.shape[0]):
        ind = kdtree.query(kdtree.data[point],neighbors+1)[1][1:]
        dX = Xm[ind] - Xm[point]
        Xstd = np.std(dX,0)
        gX = np.exp(-dX**2/(2*Xstd**2))
        gwX = np.product(gX,1)
        gw[point,:] = gwX/gwX.sum()
        nearest[point,:] = ind
    gw[np.isnan(gw)] = 0.01
    return gw, nearest.astype(int)

class lc_fitter(object):

    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=100, eclipse=False):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.syspars = syspars
        self.gw, self.nearest = gaussian_weights(syspars, neighbors=neighbors)
        self.eclipse = eclipse  # offset model such that minimum is at 1
        self.fit_nested()

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        # alloc arrays for C
        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        self.lightcurve = np.zeros(len(self.time),dtype=ctypes.c_double)
        self.lightcurve = np.require(self.lightcurve,dtype=ctypes.c_double,requirements='C')

        def loglike(pars):
            # update free parameters
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]

            # call C function
            keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            occultquadC(time, *vals, len(time), self.lightcurve)
            self.lightcurve += self.eclipse*(1-np.min(self.lightcurve))
            detrended = self.data/self.lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = self.lightcurve*wf
            return -0.5 * np.sum(((self.data-model)**2/self.dataerr**2))
        
        def prior_transform(upars):
            # transform unit cube to prior volume
            return (boundarray[:,0] + bounddiff*upars)

        #dsampler = dynesty.NestedSampler(loglike, prior_transform, len(freekeys), sample='unif', bound='multi', nlive=1000)

        dsampler = dynesty.DynamicNestedSampler(
            loglike, prior_transform,
            ndim=len(freekeys), bound='multi', sample='unif', 
            maxiter_init=5000, dlogz_init=1, dlogz=0.05, 
            maxiter_batch=1000, maxbatch=10, nlive_batch=100
        )

        dsampler.run_nested(maxiter=2e6,maxcall=2e6)
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
            lightcurve += self.eclipse*(1-np.min(lightcurve))
            detrended = self.data / lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            residuals = self.data - model
            res.append(residuals)
            btime, br = time_bin(self.time, residuals)
            blc = transit(btime, tests[i])
            mask = blc < 1
            if mask.shape[0] == 0:
                mask = np.ones(blc.shape,dtype=bool)
            if mask.sum() == 0:
                mask = np.ones(blc.shape,dtype=bool)
            duration = btime[mask].max() - btime[mask].min()
            tmask = ((btime - tests[i]['tmid']) < duration) & ((btime - tests[i]['tmid']) > -1*duration)
            chis.append(np.mean(br[tmask]**2))

        mi = np.argmin(chis)
        self.parameters = copy.deepcopy(tests[mi])
        # plt.scatter(samples[mask,0], samples[mask,1], c=weights[mask]); plt.show()

        # best fit model
        self.transit = transit(self.time, self.parameters)
        self.transit += self.eclipse*(1-np.min(self.transit))
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf

    def plot_bestfit(self):
        f = plt.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]
        bt, bf = time_bin(self.time, self.detrended,1./(24*60))
        axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
        axs[0].plot(bt,bf,'c.',alpha=0.5,zorder=2)
        axs[0].plot(self.time, self.transit, 'r-', zorder=3)
        axs[0].set_xlabel("Time [day]")
        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')
        
        axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=r'$\sigma$ = {:.0f} ppm'.format( np.std(self.residuals/np.median(self.data)*1e6)))
        bt, br = time_bin(self.time, self.residuals/np.median(self.data)*1e6,1./(24*60))
        axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format( np.std(br)))

        axs[1].legend(loc='best')
        axs[1].set_xlabel("Time [day]")
        axs[1].set_ylabel("Residuals [ppm]")
        axs[1].grid(True,ls='--')
        plt.tight_layout()

        return f,axs

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

def get_ld(priors, band='Spit36'):
    '''
    Query the web for limb darkening coefficients in the Spitzer bandpass
    Problem with LDTK + Spitzer: https://github.com/hpparvi/ldtk/issues/11
    '''
    url = 'http://astroutils.astronomy.ohio-state.edu/exofast/quadld.php'

    form = {
        'action':url,
        'pname':'Select Planet',
        'bname':band,
        'teff':priors['T*'],
        'feh':priors['FEH*'],
        'logg':priors['LOGG*']
    }
    session = requests.Session()
    res = session.post(url,data=form)
    lin,quad = re.findall(r"\d+\.\d+",res.text)
    return float(lin), float(quad)

if __name__ == "__main__":
    import json
    import pickle 

    rsun = 6.955e8 # m
    rjup = 7.1492e7 # m
    au=1.496e11 # m 

    # Gaussian kernel regression to handle Spitzer systematics
    priors = json.load(open('Spitzer/WASP-19_prior.json','r'))
    
    u1,u2 = get_ld(priors, band='Spit36') # (0.078093363, 0.18576002)
    # u1,u2 = get_ld(priors, band='Spit45') # (0.069588612, 0.14764559)

    prior = { 
        'rprs': priors['b']['rp']*rjup / (priors['R*']*rsun) ,
        'ars': priors['b']['sma']*au/(priors['R*']*rsun),
        'per': priors['b']['period'],
        'inc': priors['b']['inc'],
        'u1': u1, 'u2': u2, # limb darkening (linear, quadratic)
        'ecc': priors['b']['ecc'],
        'omega': priors['b'].get('omega',0), 
        'tmid':0.75 
    } 

    #pipeline_data = pickle.load(open('Spitzer/WASP-19_data.pkl','rb'))
    pipeline_data = pickle.load(open('Spitzer/WASP-19_data.pkl','rb'))

    # time = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_time']
    # btime, data = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_flux'], dt=0.5/(60*24))
    # btime, dataerr = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_err'], dt=0.5/(60*24))
    # btime, wx = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_xcent'], dt=0.5/(60*24))
    # btime, wy = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_ycent'], dt=0.5/(60*24))
    # btime, npp = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_npp'], dt=0.5/(60*24))

    time = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_time']
    data = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_flux']
    dataerr = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_err']
    wx = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_xcent']
    wy = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_ycent']
    npp = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_npp']

    syspars = np.array([wx,wy,npp]).T

    mybounds = {
        'rprs':[0,1.25*prior['rprs']],
        'tmid':[min(time),max(time)],
        'ars':[prior['ars']*0.9,prior['ars']*1.1]
    }

    print(np.median(time))
    print(time.shape)

    # native resolution ~9500 datapoints, ~52 minutes
    # native resolution, C-optimized, ~4 minutes
    myfit = lc_fitter(time, data, dataerr, prior, mybounds, syspars)
    
    for k in myfit.bounds.keys():
        print("{:.6f} +- {}".format( myfit.parameters[k], myfit.errors[k]))

    fig,axs = myfit.plot_bestfit()

    # triangle plot
    fig,axs = dynesty.plotting.cornerplot(myfit.results, labels=['Rp/Rs','Tmid','a/Rs'], quantiles_2d=[0.4,0.85], smooth=0.015, show_titles=True,use_math_text=True, title_fmt='.2e',hist2d_kwargs={'alpha':1,'zorder':2,'fill_contours':False})
    dynesty.plotting.cornerpoints(myfit.results, labels=['Rp/Rs','Tmid','a/Rs'], fig=[fig,axs[1:,:-1]],plot_kwargs={'alpha':0.1,'zorder':1,} )
    plt.tight_layout()
    plt.show()

    # pixel map 
    # plt.scatter(wx, wy, c=myfit.wf/np.median(myfit.wf), vmin=0.99, vmax=1.01,cmap='jet'); plt.show()