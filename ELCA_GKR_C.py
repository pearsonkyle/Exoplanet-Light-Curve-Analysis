# Uses Gaussian Kernel Regression to handle errors correlated to centroid position
import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt

import dynesty
from dynesty import plotting

from scipy import spatial

########################################################
# LOAD IN TRANSIT FUNCTION FROM C

# define 1d array pointer in python
array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

# load library
lib_trans = np.ctypeslib.load_library('lib_transit.so',
    os.path.join(os.environ.get('ELCA_PATH'),'C_sharedobject')
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

    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=50):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.syspars = syspars
        self.gw, self.nearest = gaussian_weights(syspars, neighbors=neighbors)
        self.fit_nested()

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        lightcurve = np.zeros(len(self.time),dtype=ctypes.c_double)
        lightcurve = np.require(lightcurve,dtype=ctypes.c_double,requirements='C')

        def loglike(pars):
            # update free parameters
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            # lightcurve = transit(self.time, self.prior)

            # call C function
            keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            occultquadC( time, *vals, len(time), lightcurve )
            
            detrended = self.data/lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
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

        # alloc data
        self.errors = {}
        self.parameters = {}
        for k in self.prior:
            self.parameters[k] = self.prior[k]
            
        # errors + final values
        self.weights = np.exp(self.results['logwt'] - self.results['logz'][-1])
        for i in range(len(freekeys)):
            lo,me,up = dynesty.utils.quantile(self.results.samples[:,i], [0.025, 0.5, 0.975], weights=self.weights)
            self.errors[freekeys[i]] = [lo-me,up-me]
            self.parameters[freekeys[i]] = me
        
        # best fit model
        self.transit = transit(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf

    def plot_bestfit(self):
        f = plt.figure( figsize=(12,7) )
        #f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid( (4,5), (0,0), colspan=5,rowspan=3 )
        ax_res = plt.subplot2grid( (4,5), (3,0), colspan=5, rowspan=1 )
        axs = [ax_lc, ax_res]

        axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
        axs[0].plot(self.time, self.transit, 'r-', zorder=2)
        axs[0].set_xlabel("Time [day]")
        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')

        axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.5)
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
    priors = json.load(open('Spitzer/prior.json','r'))
    
    # u1,u2 = get_ld(priors, band='Spit36') # (0.078093363, 0.18576002)
    # u1,u2 = get_ld(priors, band='Spit45') # (0.069588612, 0.14764559)

    prior = { 
        'rprs': priors['b']['rp']*rjup / (priors['R*']*rsun) ,
        'ars': priors['b']['ars'],
        'per': priors['b']['period'],
        'inc': priors['b']['inc'],
        'u1': 0.078, 'u2': 0.1857, # limb darkening (linear, quadratic)
        'ecc': priors['b']['ecc'],
        'omega': priors['b']['omega'], 
        'tmid':0.75 
    } 

    pipeline_data = pickle.load(open('Spitzer/data.pkl','rb'))

    # time = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_time']
    # btime, data = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_flux'], dt=0.5/(60*24))
    # btime, dataerr = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_err'], dt=0.5/(60*24))
    # btime, wx = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_xcent'], dt=0.5/(60*24))
    # btime, wy = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_ycent'], dt=0.5/(60*24))
    # btime, npp = time_bin(time, pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_npp'], dt=0.5/(60*24))

    time = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_time']
    data = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_flux']
    dataerr = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_err']
    wx = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_xcent']
    wy = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_ycent']
    npp = pipeline_data['Spitzer-IRAC-IR-36-SUB']['b'][0]['aper_npp']

    syspars = np.array([wx,wy,npp]).T

    mybounds = {
        'rprs':[0,0.2],
        'tmid':[min(time),max(time)],
        'ars':[7,9]
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