import os
import copy
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

import dynesty
from dynesty import plotting
from dynesty.utils import resample_equal

from scipy.optimize import least_squares
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

# transit, orbital radius, anomaly
input_type1 = [array_1d_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, array_1d_double ]

# phasecurve, brightness, eclipse
input_type2 = [array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, array_1d_double ]

# transit
occultquadC = lib_trans.occultquad
occultquadC.argtypes = input_type1
occultquadC.restype = None

# orbital radius
orbitalRadius = lib_trans.orbitalradius
orbitalRadius.argtypes = input_type1
orbitalRadius.restype = None

# true anomaloy
orbitalAnomaly = lib_trans.orbitalanomaly
orbitalAnomaly.argtypes = input_type1
orbitalAnomaly.restype = None

# phase curve
phaseCurve = lib_trans.phasecurve
phaseCurve.argtypes = input_type2
phaseCurve.restype = None

# phase curve without eclipse
brightnessCurve = lib_trans.brightness
brightnessCurve.argtypes = input_type2
brightnessCurve.restype = None

# eclipse
eclipseC = lib_trans.eclipse
eclipseC.argtypes = input_type2
eclipseC.restype = None


# cast arrays into C compatible format with pythonic magic
def format_args(f):    
    @wraps(f)
    def wrapper(*args,**kwargs):
        if len(args) == 2:
            t = args[0]
            values = args[1]
        data = {}
        data['time'] = np.require(t,dtype=ctypes.c_double,requirements='C')
        data['model'] = np.require(np.ones(len(t)),dtype=ctypes.c_double,requirements='C')
        data['cvals'] = np.require(np.zeros(5),dtype=ctypes.c_double,requirements='C')
        if 'transit' in f.__name__ or 'orbit' in f.__name__ or 'anomaly' in f.__name__:
            keys=['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
        else:
            keys=['fpfs','rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
        data['vals'] = [values[k] for k in keys]
        for i,k in enumerate(['c0','c1','c2','c3','c4']): data['cvals'][i] = values[k]
        return f(*args, **data)
    return wrapper

@format_args
def brightness(t, values, **kwargs):
    brightnessCurve( kwargs['time'], kwargs['cvals'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def phasecurve(t, values, **kwargs):
    phaseCurve(kwargs['time'], kwargs['cvals'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def eclipse(t, values, **kwargs):
    eclipseC( kwargs['time'], kwargs['cvals'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def orbitalradius(t, values, **kwargs):
    orbitalRadius( kwargs['time'], *kwargs['vals'], len(kwargs['time']), kwargs['model'] )
    return kwargs['model'] 

@format_args
def trueanomaly(t, values, **kwargs):
    orbitalAnomaly( kwargs['time'], *kwargs['vals'], len(kwargs['time']), kwargs['model'] )
    return kwargs['model']

@format_args
def transit(t, values, **kwargs):
    occultquadC( kwargs['time'], *kwargs['vals'], len(kwargs['time']), kwargs['model'] )
    return kwargs['model']
########################################################

def fourierseries(t, C, P):
    return (1 + C[0] + C[1]*np.cos(2*np.pi*t/P) + C[2]*np.sin(2*np.pi*t/P) + C[3]*np.cos(4*np.pi*t/P) + C[4]*np.sin(4*np.pi*t/P))

class map_fitter(object):

    def __init__(self, time, prior):
        self.time = time
        self.prior = prior
        self.fit_lm()
        
    def fit_lm(self): 

        # transit starts at -pi/2    
        ta = trueanomaly(self.time, self.prior)
        longitude = np.rad2deg(ta)+90

        # TODO adjust for inclination
        latitude = np.linspace(-90,90,91) # break in 2 deg chunks

        # fractional area emitting light in direction of observer
        coslat = np.cos(np.deg2rad(latitude)) 

        # reproduce amplitude from observations
        phase_amp = brightness(self.time, self.prior)
        brightmap = np.zeros(phase_amp.shape)

        def map2min(pars):
            # approx longitudinal brightness map with fourier series
            map1d = fourierseries(self.time, pars, self.prior['per'])
            # TODO convert to temperature map instead? 

            # assumes planet is tidally locked
            lmask = (longitude%360 > (longitude[0]-90)%360 ) & (longitude%360 < (longitude[0]+90)%360)

            # lamberts law of cosines
            dlong = longitude - longitude[0]
            coslong = np.cos(np.deg2rad(dlong)) * lmask

            # compute disk integrated brightness for each orbital phase
            for i in range(map1d.shape[0]):
                brightmap[i] = np.sum(map1d*np.roll(coslong,i))
            
            # TODO figure out how to normalize
            # compare to observations
            return (phase_amp - brightmap/brightmap.min())**2

        freekeys = ['c0','c1','c2','c3','c4']
        res = least_squares(map2min, x0=[self.prior[k] for k in freekeys], jac='3-point', loss='linear')
        print(res.x)


class lc_fitter(object):

    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=100, mode='ns'):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.syspars = syspars
        self.neighbors = neighbors
        
        if mode == 'ns':
            self.fit_nested()
        else:
            self.fit_lm()

    def fit_lm(self):

        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])

        # trim data around predicted transit/eclipse time
        self.gw, self.nearest = gaussian_weights(self.syspars, neighbors=self.neighbors)

        # alloc arrays for C
        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        lightcurve = np.require(np.zeros(len(time)),dtype=ctypes.c_double,requirements='C')
        cvals = np.require(np.zeros(5),dtype=ctypes.c_double,requirements='C')

        def lc2min(pars):
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]

            # call C function
            keys = ['fpfs', 'rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            for i,k in enumerate(['c0','c1','c2','c3','c4']): cvals[i] = self.prior[k]
            phaseCurve( self.time, cvals, *vals, len(self.time), lightcurve)

            detrended = self.data/lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return ((self.data-model)/self.dataerr)**2 

        res = least_squares(lc2min, x0=[self.prior[k] for k in freekeys], 
            bounds=[boundarray[:,0], boundarray[:,1]], jac='3-point', 
            loss='linear', method='dogbox', xtol=None, ftol=1e-4, tr_options='exact')

        self.parameters = copy.deepcopy(self.prior)
        self.errors = {}

        for i,k in enumerate(freekeys):
            self.parameters[k] = res.x[i]
            self.errors[k] = 0

        # best fit model
        self.transit = phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf
        self.phase = (self.time-self.parameters['tmid'])/self.parameters['per']

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)
        
        # trim data around predicted transit/eclipse time
        self.gw, self.nearest = gaussian_weights(self.syspars, neighbors=self.neighbors)

        # alloc arrays for C
        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        lightcurve = np.require(np.zeros(len(time)),dtype=ctypes.c_double,requirements='C')
        cvals = np.require(np.zeros(5),dtype=ctypes.c_double,requirements='C')

        def loglike(pars):
            # update free parameters
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]

            # call C function
            keys = ['fpfs', 'rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            for i,k in enumerate(['c0','c1','c2','c3','c4']): cvals[i] = self.prior[k]
            phaseCurve(time, cvals, *vals, len(time), lightcurve)

            detrended = self.data/lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return -0.5 * np.sum(((self.data-model)**2/self.dataerr**2))
        
        def prior_transform(upars):
            freekeys = list(self.bounds.keys())
            boundarray = np.array([self.bounds[k] for k in freekeys])
            bounddiff = np.diff(boundarray,1).reshape(-1)
            vals = (boundarray[:,0] + bounddiff*upars)

            # set limits of phase amplitude to be less than eclipse depth
            edepth = vals[freekeys.index('rprs')]**2 * vals[freekeys.index('fpfs')]
            for k in ['c1','c2','c3','c4']:
                ki = freekeys.index(k)
                vals[ki] = upars[ki] * edepth - 0.5*edepth

            return vals

        dsampler = dynesty.NestedSampler(loglike, prior_transform, len(freekeys), sample='unif', bound='multi', nlive=1000)
        
        # DynamicNestedSampler(
        #     loglike, prior_transform,
        #     ndim=len(freekeys), bound='multi',
        #     maxiter_init=5000, dlogz_init=1, dlogz=0.05,
        #     maxiter_batch=1000, maxbatch=10, nlive_batch=100
        # )
        
        
        dsampler.run_nested()
        self.results = dsampler.results
        del(self.results['bound'])

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
            
            lightcurve = phasecurve(self.time, tests[i])
            detrended = self.data / lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            residuals = self.data - model
            res.append(residuals)
            btime, br = time_bin(self.time, residuals)
            blc = transit(btime, tests[i])
            mask = np.ones(blc.shape,dtype=bool)
            # TODO add more ephemesis on in transit fits
            duration = btime[mask].max() - btime[mask].min()
            tmask = ((btime - tests[i]['tmid']) < duration) & ((btime - tests[i]['tmid']) > -1*duration)
            chis.append(np.mean(br[tmask]**2))

        mi = np.argmin(chis)
        self.parameters = copy.deepcopy(tests[mi])
        # plt.scatter(samples[mask,0], samples[mask,1], c=weights[mask]); plt.show()

        # best fit model
        self.transit = phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf
        self.phase = (self.time-self.parameters['tmid'])/self.parameters['per']

    def plot_bestfit(self, bin_dt=10./(60*24), zoom=False, phase=True):
        f = plt.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        #print("in elca_phasecurve, prototype phase calculation")
        #import pdb; pdb.set_trace()

        bt, bf = time_bin(self.time, self.detrended, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']

        if phase:
            axs[0].plot(bp,bf,'co',alpha=0.5,zorder=2)
            axs[0].plot(self.phase, self.transit, 'r-', zorder=3)
            axs[0].set_xlim([min(self.phase), max(self.phase)])
            axs[0].set_xlabel("Phase ")
        else:
            axs[0].plot(bt,bf,'co',alpha=0.5,zorder=2)
            axs[0].plot(self.time, self.transit, 'r-', zorder=3)
            axs[0].set_xlim([min(self.time), max(self.time)])
            axs[0].set_xlabel("Time [day]")
       
            
        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')

        if zoom:
            axs[0].set_ylim([1-1.25*self.parameters['rprs']**2, 1+0.5*self.parameters['rprs']**2])
        else:
            if phase:
                axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.01)
            else:
                axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.01)

        bt, br = time_bin(self.time, self.residuals/np.median(self.data)*1e6, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']

        if phase:
            axs[1].plot(self.phase, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=r'$\sigma$ = {:.0f} ppm'.format( np.std(self.residuals/np.median(self.data)*1e6)))
            axs[1].plot(bp,br,'c.',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format( np.std(br)))
            axs[1].set_xlim([min(self.phase), max(self.phase)])
            axs[1].set_xlabel("Phase")

        else:
            axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=r'$\sigma$ = {:.0f} ppm'.format( np.std(self.residuals/np.median(self.data)*1e6)))
            axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format( np.std(br)))
            axs[1].set_xlim([min(self.time), max(self.time)])
            axs[1].set_xlabel("Time [day]")
        
        
        axs[1].legend(loc='best')
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

    

if __name__ == "__main__":
    import json
    import pickle 

    rsun = 6.955e8 # m
    rjup = 7.1492e7 # m
    au=1.496e11 # m 

    # Gaussian kernel regression to handle Spitzer systematics
    priors = json.load(open('Spitzer/WASP-103_prior.json','r'))
    
    #u1,u2 = get_ld(priors, band='Spit36') 
    u1,u2 = get_ld(priors, band='Spit45') 

    prior = { 
        # transit 
        'rprs': priors['b']['rp']*rjup / (priors['R*']*rsun) ,
        'ars': priors['b']['sma']*au/(priors['R*']*rsun),
        'per': priors['b']['period'],
        'inc': priors['b']['inc'],
        'tmid':0.25, 

        # eclipse 
        'fpfs': 0.01,
        'omega': priors['b'].get('omega',0), 
        'ecc': priors['b']['ecc'],

        # limb darkening (linear, quadratic)
        'u1': u1, 'u2': u2, 
    
        # phase curve amplitudes
        'c0':5e-4, 'c1':0, 'c2':-5e-4, 'c3':0., 'c4':0
    }

    print("edepth:",prior['rprs']**2 * prior['fpfs'])
    #pipeline_data = pickle.load(open('Spitzer/WASP-19_data.pkl','rb'))
    pipeline_data = pickle.load(open('Spitzer/WASP-103_data.pkl','rb'))

    # time = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_time']
    # data = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_flux']
    # dataerr = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_err']
    # wx = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_xcent']
    # wy = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_ycent']
    # npp = pipeline_data['Spitzer-IRAC-IR-45-SUB']['b'][1]['aper_npp']
    # syspars = np.array([wx,wy,npp]).T

    print((priors['b']['rp']*rjup / (priors['R*']*rsun))**2)

    time = np.linspace(0, priors['b']['period'], 10000)

    mapfit = map_fitter(time, prior)
    

    f,ax = plt.subplots(2)
    phase = (time - prior['tmid'])/prior['per']
    ax[0].plot(phase, phasecurve(time, prior),'k.', alpha=0.5)
    ax[0].plot(phase, transit(time,prior), 'r.',alpha=0.5)
    ax[0].plot(phase, brightness(time, prior), 'c.', alpha=0.5)
    #ax[0].plot(time, eclipse(time, prior), 'm.', alpha=0.5)
    
    ta = trueanomaly(time, prior)
    longitude = np.rad2deg(ta)+90
    ax[1].plot(ta, brightness(time, prior), 'k.', alpha=0.5)
    ax[1].set_xlabel("True Anomaly")
    ax[1].set_ylabel("Relative Flux")
    plt.show()


    dude()
    
    mybounds = {
        'rprs':[0,1.25*prior['rprs']],
        'tmid':[min(time),max(time)],
        'ars':[prior['ars']*0.9,prior['ars']*1.1],
        
        'fpfs':[0,0.5],
        'omega': [prior['omega']-25,prior['omega']+25],
        'ecc': [0,0.05],

        'c0':[-1,1],
        'c1':[-1,1],
        'c2':[-1,1],
        'c3':[-1,1],
        'c4':[-1,1]
    }

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