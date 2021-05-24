import os
import glob
import copy
import ctypes
import logging
import requests
import numpy as np
from io import StringIO
from pandas import read_csv
from functools import wraps
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from scipy import spatial

import ultranest

########################################################
# LOAD IN TRANSIT FUNCTION FROM C

# define 1d array pointer in python
array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

# load library
c_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'C')
c_files = glob.glob(os.path.join(c_dir,'*.so'))

if len(c_files) == 0:
    assert("No shared object file: lib_transit")
else:
    lib_trans = np.ctypeslib.load_library(c_files[0], c_dir)

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

input_moon = [array_1d_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, \
                        array_1d_double, ctypes.c_double, array_1d_double ]

# transit
occultquadC = lib_trans.occultquad
occultquadC.argtypes = input_type1
occultquadC.restype = None

# transit with variable tmid
occultquadC_tmid = lib_trans.occultquad_tmid
occultquadC_tmid.argtypes = input_moon
occultquadC_tmid.restype = None

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
        else: # order matters
            keys=['fpfs','rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
        data['vals'] = [values.get(k,0) for k in keys]
        for i,k in enumerate(['c0','c1','c2','c3','c4']): data['cvals'][i] = values.get(k,0)
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

def transit_tmid(t, tmid, values):
    # variable mid-transit time transit function
    time = np.require(t,dtype=ctypes.c_double,requirements='C')
    tmid = np.require(tmid,dtype=ctypes.c_double,requirements='C')
    model = np.zeros(len(t),dtype=ctypes.c_double)
    model = np.require(model,dtype=ctypes.c_double,requirements='C')
    keys = ['rprs','ars','per','inc','u1','u2','ecc','omega']
    vals = [values[k] for k in keys]
    occultquadC_tmid( time, *vals, tmid, len(time), model )
    return model

########################################################
# NONLINEAR LD TRANSIT

def tldlc(z, rprs, g1=0, g2=0, g3=0, g4=0, nint=int(2**3)):
    '''
    G. ROUDIER: Light curve model
    '''
    ldlc = np.zeros(z.size)
    xin = z.copy() - rprs
    xin[xin < 0e0] = 0e0
    xout = z.copy() + rprs
    xout[xout > 1e0] = 1e0
    select = xin > 1e0
    if True in select: ldlc[select] = 1e0
    inldlc = []
    xint = np.linspace(1e0, 0e0, nint)
    znot = z.copy()[~select]
    xinnot = np.arccos(xin[~select])
    xoutnot = np.arccos(xout[~select])
    xrs = np.array([xint]).T*(xinnot - xoutnot) + xoutnot
    xrs = np.cos(xrs)
    diffxrs = np.diff(xrs, axis=0)
    extxrs = np.zeros((xrs.shape[0]+1, xrs.shape[1]))
    extxrs[1:-1, :] = xrs[1:,:] - diffxrs/2.
    extxrs[0, :] = xrs[0, :] - diffxrs[0]/2.
    extxrs[-1, :] = xrs[-1, :] + diffxrs[-1]/2.
    occulted = vecoccs(znot, extxrs, rprs)
    diffocc = np.diff(occulted, axis=0)
    si = vecistar(xrs, g1, g2, g3, g4)
    drop = np.sum(diffocc*si, axis=0)
    inldlc = 1. - drop
    ldlc[~select] = np.array(inldlc)
    return ldlc

def vecistar(xrs, g1, g2, g3, g4):
    '''
    G. ROUDIER: Stellar surface extinction model
    '''
    ldnorm = (-g1/10e0 - g2/6e0 - 3e0*g3/14e0 - g4/4e0 + 5e-1)*2e0*np.pi
    select = xrs < 1e0
    mu = np.zeros(xrs.shape)
    mu[select] = (1e0 - xrs[select]**2)**(1e0/4e0)
    s1 = g1*(1e0 - mu)
    s2 = g2*(1e0 - mu**2)
    s3 = g3*(1e0 - mu**3)
    s4 = g4*(1e0 - mu**4)
    outld = (1e0 - (s1+s2+s3+s4))/ldnorm
    return outld

def vecoccs(z, xrs, rprs):
    '''
    G. ROUDIER: Stellar surface occulation model
    '''
    out = np.zeros(xrs.shape)
    vecxrs = xrs.copy()
    selx = vecxrs > 0e0
    veczsel = np.array([z.copy()]*xrs.shape[0])
    veczsel[veczsel < 0e0] = 0e0
    select1 = (vecxrs <= rprs - veczsel) & selx
    select2 = (vecxrs >= rprs + veczsel) & selx
    select = (~select1) & (~select2) & selx
    zzero = veczsel == 0e0
    if True in select1 & zzero:
        out[select1 & zzero] = np.pi*(np.square(vecxrs[select1 & zzero]))
        pass
    if True in select2 & zzero: out[select2 & zzero] = np.pi*(rprs**2)
    if True in select & zzero: out[select & zzero] = np.pi*(rprs**2)
    if True in select1 & ~zzero:
        out[select1 & ~zzero] = np.pi*(np.square(vecxrs[select1 & ~zzero]))
        pass
    if True in select2: out[select2 & ~zzero] = np.pi*(rprs**2)
    if True in select & ~zzero:
        redxrs = vecxrs[select & ~zzero]
        redz = veczsel[select & ~zzero]
        s1 = (np.square(redz) + np.square(redxrs) - rprs**2)/(2e0*redz*redxrs)
        s1[s1 > 1e0] = 1e0
        s2 = (np.square(redz) + rprs**2 - np.square(redxrs))/(2e0*redz*(rprs+0.0001))
        s2[s2 > 1e0] = 1e0
        fs3 = -redz + redxrs + rprs
        ss3 = redz + redxrs - rprs
        ts3 = redz - redxrs + rprs
        os3 = redz + redxrs + rprs
        s3 = fs3*ss3*ts3*os3
        zselect = s3 < 0e0
        if True in zselect: s3[zselect] = 0e0
        out[select & ~zzero] = (np.square(redxrs)*np.arccos(s1) +
                                (rprs**2)*np.arccos(s2) - (5e-1)*np.sqrt(s3))
        pass
    return out

def time2z(time, ipct, tknot, sma, orbperiod, ecc, tperi=None, epsilon=1e-5):
    '''
    G. ROUDIER: Time samples in [Days] to separation in [R*]
    '''
    if tperi is not None:
        ft0 = (tperi - tknot) % orbperiod
        ft0 /= orbperiod
        if ft0 > 0.5: ft0 += -1e0
        M0 = 2e0*np.pi*ft0
        E0 = solveme(M0, ecc, epsilon)
        realf = np.sqrt(1e0 - ecc)*np.cos(E0/2e0)
        imagf = np.sqrt(1e0 + ecc)*np.sin(E0/2e0)
        w = np.angle(np.complex(realf, imagf))
        if abs(ft0) < epsilon:
            w = np.pi/2e0
            tperi = tknot
            pass
        pass
    else:
        w = np.pi/2e0
        tperi = tknot
        pass
    ft = (time - tperi) % orbperiod
    ft /= orbperiod
    sft = np.copy(ft)
    sft[(sft > 0.5)] += -1e0
    M = 2e0*np.pi*ft
    E = solveme(M, ecc, epsilon)
    realf = np.sqrt(1. - ecc)*np.cos(E/2e0)
    imagf = np.sqrt(1. + ecc)*np.sin(E/2e0)
    f = []
    for r, i in zip(realf, imagf):
        cn = np.complex(r, i)
        f.append(2e0*np.angle(cn))
        pass
    f = np.array(f)
    r = sma*(1e0 - ecc**2)/(1e0 + ecc*np.cos(f))
    z = r*np.sqrt(1e0**2 - (np.sin(w+f)**2)*(np.sin(ipct*np.pi/180e0))**2)
    z[sft < 0] *= -1e0
    return z, sft

def solveme(M, e, eps):
    '''
    G. ROUDIER: Newton Raphson solver for true anomaly
    M is a numpy array
    '''
    E = np.copy(M)
    for i in np.arange(M.shape[0]):
        while abs(E[i] - e*np.sin(E[i]) - M[i]) > eps:
            num = E[i] - e*np.sin(E[i]) - M[i]
            den = 1. - e*np.cos(E[i])
            E[i] = E[i] - num/den
            pass
        pass
    return E

def transit_nl(time, values):
    sep,phase = time2z(time, values['inc'], values['tmid'], values['ars'], values['per'], values['ecc'])
    model = tldlc(abs(sep), values['rprs'], values['u0'], values['u1'], values['u2'], values['u3'])
    return model

#########################################################
# NASA Exoplanet Archive Scraper

def tap_query(base_url, query, dataframe=True):
    """query a url with TAP protocol

    Parameters
    ----------
    base_url : str
        Table Name

    query : dict
        TAP keys as a dictionary
        (select, from, where, order by, format) 

    Returns
    -------
    Pandas Dataframe or text
    """
    # build url
    uri_full = base_url
    for k in query:
        if k != "format":
            uri_full += f"{k} {query[k]} "

    uri_full = f"{uri_full[:-1]} &format={query.get('format', 'csv')}"
    uri_full = uri_full.replace(' ', '+')
    print(uri_full)

    response = requests.get(uri_full, timeout=300)
    # TODO check status_code?

    if dataframe:
        return read_csv(StringIO(response.text))
    else:
        return response.text

def nea_scrape(target=None):
    """ download parameters from nasa exoplanet archive

    Parameters
    ----------
    target : str
        Table Name

    Returns
    -------
    Pandas Dataframe
    """
    uri_ipac_base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    uri_ipac_query = {
        # Table columns: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
        "select": "pl_name,hostname,pl_radj,pl_radjerr1,ra,dec,"
                    "pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,"
                    "pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbeccen,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,"
                    "pl_orblper,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,"
                    "pl_ratror,pl_ratrorerr1,pl_ratrorerr2,"
                    "st_teff,st_tefferr1,st_tefferr2,st_met,st_meterr1,st_meterr2,"
                    "st_logg,st_loggerr1,st_loggerr2,st_mass,st_rad,st_raderr1",
        "from": "pscomppars",  # Table name
        "where": "tran_flag = 1",
       #"order by": "pl_pubdate desc",
        "format": "csv"
    }

    if target:
        uri_ipac_query["where"] += f" and pl_name = '{target}'"

    return tap_query(uri_ipac_base, uri_ipac_query)

#########################################################
# Fitting algorithm for detrended data
class lc_fitter():

    def __init__(self, time, data, dataerr, prior, bounds, verbose=False):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.verbose = verbose
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
        
        if self.verbose:
            self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=5e5)
        else:
            self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=5e5, show_status=self.verbose, viz_callback=self.verbose)

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
            labels.append(flabels.get(key, key))
            titles.append(f"{self.parameters[key]:.5f} +- {self.errors[key]:.5f}")
            ranges.append([
                self.parameters[key] - 5*self.errors[key],
                self.parameters[key] + 5*self.errors[key]
            ])

            if key == 'a2' or key == 'a1': 
                continue

            mask3 = mask3 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 3*self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 3*self.errors[key]) )

            mask1 = mask1 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + self.errors[key]) )

            mask2 = mask2 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 2*self.errors[key]) ) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 2*self.errors[key]) )

        chi2 = self.results['weighted_samples']['logl']*-2
        fig = corner(self.results['weighted_samples']['points'], 
            labels= labels,
            bins=int(np.sqrt(self.results['samples'].shape[0])), 
            range= ranges,
            #quantiles=(0.1, 0.84),
            plot_contours=True,
            levels=[ np.percentile(chi2[mask1],99), np.percentile(chi2[mask2],99), np.percentile(chi2[mask3],99)],
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
        self.phase = (self.time - self.parameters['tmid']+0.25*self.parameters['per']) / self.parameters['per'] % 1 - 0.25
        self.transit = transit(self.time, self.parameters)
        self.model = self.transit
        self.detrended = self.data 
        self.detrendederr = self.dataerr 
        self.residuals = self.data - self.model
        self.chi2 = np.sum(self.residuals**2/self.dataerr**2)
        self.bic = len(self.bounds) * np.log(len(self.time)) - 2*np.log(self.chi2)

        # compare fit chi2 to smoothed data chi2
        dt = np.diff(np.sort(self.time)).mean()
        si = np.argsort(self.time)
        try:
            self.sdata = savgol_filter(self.data[si], 1+2*int(0.5/24/dt), 2)
        except:
            self.sdata = savgol_filter(self.data[si], 1+2*int(1.5/24/dt), 2)

        schi2 = np.sum((self.data[si] - self.sdata)**2/self.dataerr[si]**2)
        self.quality = schi2/self.chi2

        # measured duration
        tdur = (self.transit < 1).sum() * np.median(np.diff(np.sort(self.time)))

        # test for partial transit
        newtime = np.linspace(self.parameters['tmid']-0.2,self.parameters['tmid']+0.2,10000)
        newtran = transit(newtime, self.parameters)
        masktran = newtran < 1
        newdur = np.diff(newtime).mean()*masktran.sum()

        self.duration_measured = tdur
        self.duration_expected = newdur

    def plot_bestfit(self, title="", bin_dt=15./(60*24), zoom=False, phase=True):
        f = pl.figure(figsize=(9,6))
        f.subplots_adjust(top=0.92,bottom=0.09,left=0.1,right=0.98, hspace=0)
        ax_lc = pl.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = pl.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        axs[0].set_title(title)
        axs[0].set_ylabel("Relative Flux", fontsize=14)
        axs[0].grid(True,ls='--')

        rprs2 = self.parameters['rprs']**2
        rprs2err = 2*self.parameters['rprs']*self.errors['rprs']
        lclabel= r"$R^{2}_{p}/R^{2}_{s}$ = %6f $\pm$ %6f, $T_{mid}$ = %5f $\pm$ %5f" %(rprs2, rprs2err,
            self.parameters['tmid'], 
            self.errors['tmid'])

        if zoom:
            axs[0].set_ylim([1-1.25*self.parameters['rprs']**2, 1+0.5*self.parameters['rprs']**2])
        else:
            if phase:
                axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.2)
            else:
                axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.2)

        if phase:
            si = np.argsort(self.phase)
            bt2, br2, _ = time_bin(self.phase[si]*self.parameters['per'], self.residuals[si]/np.median(self.data)*1e6, bin_dt)
            axs[1].plot(self.phase, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.2, label=r'$\sigma$ = {:.0f} ppm'.format( np.std(self.residuals/np.median(self.data)*1e6)))
            axs[1].plot(bt2/self.parameters['per'],br2,'bo',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format( np.std(br2)))
            axs[1].set_xlim([min(self.phase), max(self.phase)])
            axs[1].set_xlabel("Phase", fontsize=14)

            si = np.argsort(self.phase)
            bt2, bf2, bs = time_bin(self.phase[si]*self.parameters['per'], self.detrended[si], bin_dt)
            axs[0].errorbar(bt2/self.parameters['per'],bf2,yerr=bs,alpha=0.5,zorder=2,color='blue',ls='none',marker='o')
            axs[0].plot(self.phase[si], self.transit[si], 'r-', zorder=3, label=lclabel)
            axs[0].set_xlim([min(self.phase), max(self.phase)])
            axs[0].set_xlabel("Phase ", fontsize=14)
        else:
            bt, br, _ = time_bin(self.time, self.residuals/np.median(self.data)*1e6, bin_dt)
            axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.2, label=r'$\sigma$ = {:.0f} ppm'.format( np.std(self.residuals/np.median(self.data)*1e6)))
            axs[1].plot(bt,br,'bo',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format( np.std(br)))
            axs[1].set_xlim([min(self.time), max(self.time)])
            axs[1].set_xlabel("Time [day]", fontsize=14)

            bt, bf, bs = time_bin(self.time, self.detrended, bin_dt)
            si = np.argsort(self.time)
            axs[0].errorbar(bt,bf,yerr=bs,alpha=0.5,zorder=2,color='blue',ls='none',marker='o')
            axs[0].plot(self.time[si], self.transit[si], 'r-', zorder=3, label=lclabel)
            axs[0].set_xlim([min(self.time), max(self.time)])
            axs[0].set_xlabel("Time [day]", fontsize=14)

        axs[0].get_xaxis().set_visible(False)
        axs[1].legend(loc='best')
        axs[0].legend(loc='best')
        axs[1].set_ylabel("Residuals [ppm]", fontsize=14)
        axs[1].grid(True,ls='--',axis='y')
        return f,axs

#########################################################
# Fitting algorithm for ground based observations
# supports exponential airmass model or quadratic in time
class lc_fitter_detrend(lc_fitter):
    # TODO add quadratic detrend if no airmass
    def __init__(self, time, data, dataerr, airmass, prior, bounds, mode='ns', verbose=False):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.airmass = airmass
        self.prior = prior
        self.bounds = bounds
        self.verbose = verbose

         # set transit function depending on quad/non-linear limb darkening
        if 'u3' in prior.keys():
            self.transit_fn = transit_nl
        else:
            self.transit_fn = transit

        if mode == "lm":
            self.fit_LM()
        elif mode == "ns":
            self.fit_nested()

    def fit_LM(self):

        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])

        def lc2min(pars):
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            model = self.transit_fn(self.time, self.prior)
            model *= self.prior['a1']*np.exp(self.prior['a2']*self.airmass)
            return ((self.data-model)/self.dataerr)**2

        try:
            res = least_squares(lc2min, x0=[self.prior[k] for k in freekeys],
                bounds=[boundarray[:,0], boundarray[:,1]], jac='3-point', loss='linear')
        except Exception as e:
            print(e)
            print("bounded light curve fitting failed...check priors (e.g. estimated mid-transit time + orbital period)")

            for i,k in enumerate(freekeys):
                if not boundarray[i,0] < self.prior[k] < boundarray[i,1]:
                    print(f"bound: [{boundarray[i,0]}, {boundarray[i,1]}] prior: {self.prior[k]}")

            print("removing bounds and trying again...")
            res = least_squares(lc2min, x0=[self.prior[k] for k in freekeys], method='lm', jac='3-point', loss='linear')

        self.parameters = copy.deepcopy(self.prior)
        self.errors = {}

        for i,k in enumerate(freekeys):
            self.parameters[k] = res.x[i]
            self.errors[k] = 0

        self.create_fit_variables()

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        def loglike(pars):
            # chi-squared
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            model = self.transit_fn(self.time, self.prior)
            model *= np.exp(self.prior['a2']*self.airmass)
            detrend = self.data/model # used to estimate a1
            model *= np.median(detrend)
            return -0.5 * np.sum( ((self.data-model)/self.dataerr)**2 )

        def prior_transform(upars):
            return (boundarray[:,0] + bounddiff*upars)

        if self.verbose:
            self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=5e5)
        else:
            self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=5e5, show_status=self.verbose, viz_callback=self.verbose)

        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)
        
        for i, key in enumerate(freekeys):

            self.parameters[key] = self.results['maximum_likelihood']['point'][i]
            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]
        
        self.create_fit_variables()

    def create_fit_variables(self):
        self.phase = (self.time - self.parameters['tmid']+0.25*self.parameters['per']) / self.parameters['per'] % 1 - 0.25
        self.transit = self.transit_fn(self.time, self.parameters)
        # TODO monte carlo the error prop for a1
        model = self.transit*np.exp(self.parameters['a2']*self.airmass)
        detrend = self.data/model
        self.parameters['a1'] = np.median(detrend)
        self.errors['a1'] = detrend.std()
        self.airmass_model = self.parameters['a1']*np.exp(self.parameters['a2']*self.airmass)
        self.model = self.transit * self.airmass_model
        self.detrended = self.data / self.airmass_model
        self.detrendederr = self.dataerr / self.airmass_model
        self.residuals = self.data - self.model
        self.chi2 = np.sum(self.residuals**2/self.dataerr**2)
        self.bic = len(self.bounds) * np.log(len(self.time)) - 2*np.log(self.chi2)

        # compare fit chi2 to smoothed data chi2
        dt = np.diff(np.sort(self.time)).mean()
        si = np.argsort(self.time)
        try:
            self.sdata = savgol_filter(self.data[si], 1+2*int(0.5/24/dt), 2)
        except:
            self.sdata = savgol_filter(self.data[si], 1+2*int(1.5/24/dt), 2)

        schi2 = np.sum((self.data[si] - self.sdata)**2/self.dataerr[si]**2)
        self.quality = schi2/self.chi2

        # measured duration
        tdur = (self.transit < 1).sum() * np.median(np.diff(np.sort(self.time)))

        # test for partial transit
        newtime = np.linspace(self.parameters['tmid']-0.2,self.parameters['tmid']+0.2,10000)
        newtran = self.transit_fn(newtime, self.parameters)
        masktran = newtran < 1
        newdur = np.diff(newtime).mean()*masktran.sum()

        self.duration_measured = tdur
        self.duration_expected = newdur


#########################################################
# Multiple light curve fitter
class glc_fitter(lc_fitter):
    def __init__(self, input_data, global_bounds, local_bounds, mode='ns', individual_fit=False):
        # keys for input_data: time, flux, ferr, airmass, priors all numpy arrays
        self.data = input_data
        self.global_bounds = global_bounds
        self.local_bounds = local_bounds
        self.individual_fit = individual_fit
        self.fit_nested()

    def fit_nested(self):

        # create bound arrays for generating samples
        nobs = len(self.data)
        gfreekeys = list(self.global_bounds.keys())

        if isinstance(self.local_bounds, dict):
            lfreekeys = list(self.local_bounds.keys())
            boundarray = np.vstack([ [self.global_bounds[k] for k in gfreekeys], [self.local_bounds[k] for k in lfreekeys]*nobs ])
        else:
            # if list type
            lfreekeys = list(self.local_bounds[0].keys())
            boundarray = [self.global_bounds[k] for k in gfreekeys]
            for i in range(nobs): 
                boundarray.extend([self.local_bounds[i][k] for k in lfreekeys])
            boundarray = np.array(boundarray)

        # fit individual light curves to constrain priors
        if self.individual_fit:
            for i in range(nobs):

                print(f"Fitting individual light curve {i+1}/{nobs}")
                mybounds = dict(**self.local_bounds, **self.global_bounds)
                if 'per' in mybounds: del(mybounds['per'])

                myfit = lc_fitter_detrend(
                    self.data[i]['time'],
                    self.data[i]['flux'],
                    self.data[i]['ferr'],
                    self.data[i]['airmass'],
                    self.data[i]['priors'],
                    mybounds
                )

                self.data[i]['individual'] = myfit.parameters.copy()
                
                # update local priors
                for j, key in enumerate(self.local_bounds.keys()):

                    boundarray[j+i*len(self.local_bounds)+len(gfreekeys),0] = myfit.parameters[key] - 5*myfit.errors[key]
                    boundarray[j+i*len(self.local_bounds)+len(gfreekeys),1] = myfit.parameters[key] + 5*myfit.errors[key]
                    if key == 'rprs':
                        boundarray[j+i*len(self.local_bounds)+len(gfreekeys),0] = max(0,myfit.parameters[key] - 5*myfit.errors[key])

                del(myfit)

        # transform unit cube to prior volume
        bounddiff = np.diff(boundarray,1).reshape(-1)
        def prior_transform(upars):
            return (boundarray[:,0] + bounddiff*upars)

        def loglike(pars):
            chi2 = 0

            # for each light curve
            for i in range(nobs):

                # global keys
                for j, key in enumerate(gfreekeys):
                    self.data[i]['priors'][key] = pars[j]

                # local keys
                for j, key in enumerate(lfreekeys):
                    self.data[i]['priors'][key] = pars[j+i*len(lfreekeys)+len(gfreekeys)]

                # compute model
                model = transit(self.data[i]['time'], self.data[i]['priors'])
                model *= np.exp(self.data[i]['priors']['a2']*self.data[i]['airmass'])
                detrend = self.data[i]['flux']/model
                model *= np.median(detrend)

                chi2 += np.sum( ((self.data[i]['flux']-model)/self.data[i]['ferr'])**2 )

            # maximization metric for nested sampling
            return -0.5*chi2

        freekeys = []+gfreekeys
        for n in range(nobs):
            for k in lfreekeys:
                freekeys.append(f"local_{n}_{k}")

        self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run()
        self.errors = {}
        self.quantiles = {}
        self.parameters = {}

        for i, key in enumerate(freekeys):

            self.parameters[key] = self.results['maximum_likelihood']['point'][i]
            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]

        for n in range(nobs):
            for k in lfreekeys:
                pkey = f"local_{n}_{k}"
                self.data[n]['priors'][k] = self.parameters[pkey]

            # solve for a1
            model = transit(self.data[n]['time'], self.data[n]['priors'])
            airmass = np.exp(self.data[n]['airmass']*self.data[n]['priors']['a2'])
            detrend = self.data[n]['flux']/(model*airmass)
            self.data[n]['priors']['a1'] = np.median(detrend)
            # save errors to each dataset here?

    def plot_bestfit(self):
        # TODO mosaic
        nobs = len(self.data)
        nrows = nobs//4+1
        fig,ax = pl.subplots(nrows, 4, figsize=(12,4*nrows))

        # turn off all axes
        for i in range(nrows*4):
            ri = int(i/4)
            ci = i%4
            if ax.ndim == 1:
                ax[i].axis('off')
            else:
                ax[ri,ci].axis('off')

        # plot observations
        for i in range(nobs):
            ri = int(i/4)
            ci = i%4
            model = transit(self.data[i]['time'], self.data[i]['priors'])
            airmass = np.exp(self.data[i]['airmass']*self.data[i]['priors']['a2'])
            detrend = self.data[i]['flux']/(model*airmass)

            if ax.ndim == 1:
                ax[i].axis('on')
                ax[i].errorbar(self.data[i]['time'], self.data[i]['flux'], yerr=self.data[i]['ferr'], ls='none', marker='o', color='black', alpha=0.5)
                ax[i].plot(self.data[i]['time'], model*airmass*detrend.mean(), 'r-')
                ax[i].set_xlabel("Time")

            else:
                ax[ri,ci].axis('on')
                ax[ri,ci].errorbar(self.data[i]['time'], self.data[i]['flux'], yerr=self.data[i]['ferr'], ls='none', marker='o', color='black', alpha=0.5)
                ax[ri,ci].plot(self.data[i]['time'], model*airmass*detrend.mean(), 'r-')
                ax[ri,ci].set_xlabel("Time")
        pl.tight_layout()
        return fig            



#########################################################
# Fitting algorithm for phase curve measurements with GKR nearest neighbor detrending

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

class pc_fitter():

    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=100, mode='ns', verbose=False):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.syspars = syspars
        self.neighbors = neighbors
        self.verbose = verbose
        
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

        self.create_fit_variables()

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
            # boundarray = np.array([self.bounds[k] for k in freekeys])
            # bounddiff = np.diff(boundarray,1).reshape(-1)
            vals = (boundarray[:,0] + bounddiff*upars)

            # set limits of phase amplitude to be less than eclipse depth or user bound
            edepth = vals[freekeys.index('rprs')]**2 * vals[freekeys.index('fpfs')]
            for k in ['c1','c2','c3','c4']:
                if k in freekeys:
                    if k == 'c1':
                        ki = freekeys.index(k)
                        vals[ki] = upars[ki]*0.4*edepth+0.1*edepth
                        #vals[ki] = boundarray[ki,0] + upars[ki]*(0.5*edepth-boundarray[ki,0])
                    if k == 'c2':
                        ki = freekeys.index(k)
                        vals[ki] = upars[ki]*0.25*edepth - 0.125*edepth
                        #vals[ki] = boundarray[ki,0] + upars[ki]*(0.1*edepth-boundarray[ki,0])

            return vals

        if self.verbose:
            self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=5e5)
        else:
            self.results = ultranest.ReactiveNestedSampler(freekeys, loglike, prior_transform).run(max_ncalls=5e5, show_status=self.verbose, viz_callback=self.verbose)
        
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

    def create_fit_variables(self):
        self.phase = (self.time - self.parameters['tmid']) / self.parameters['per']
        self.transit = phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.detrended = self.data/self.wf
        self.detrendederr = self.dataerr 
        self.residuals = self.data - self.model
        self.chi2 = np.sum(self.residuals**2/self.dataerr**2)
        self.bic = len(self.bounds) * np.log(len(self.time)) - 2*np.log(self.chi2)

        # compare fit chi2 to smoothed data chi2
        dt = np.diff(np.sort(self.time)).mean()
        si = np.argsort(self.time)
        try:
            self.sdata = savgol_filter(self.data[si], 1+2*int(0.5/24/dt), 2)
        except:
            self.sdata = savgol_filter(self.data[si], 1+2*int(1.5/24/dt), 2)

        schi2 = np.sum((self.data[si] - self.sdata)**2/self.dataerr[si]**2)
        self.quality = schi2/self.chi2

        # measured duration
        tdur = (self.transit < 1).sum() * np.median(np.diff(np.sort(self.time)))

        # test for partial transit
        newtime = np.linspace(self.parameters['tmid']-0.2,self.parameters['tmid']+0.2,10000)
        newtran = transit(newtime, self.parameters)
        masktran = newtran < 1
        newdur = np.diff(newtime).mean()*masktran.sum()

        self.duration_measured = tdur
        self.duration_expected = newdur

    def plot_bestfit(self, bin_dt=10./(60*24), zoom=False, phase=True):
        f = pl.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = pl.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = pl.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        bt, bf, _ = time_bin(self.time, self.detrended, bin_dt)
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
                axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.025)
            else:
                axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.025)

        bt, br, _ = time_bin(self.time, self.residuals/np.median(self.data)*1e6, bin_dt)
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
        pl.tight_layout()
        return f,axs

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
        # constrain plots to +/- 4 sigma and estimate sigma levels
        for i, key in enumerate(self.quantiles):
            titles.append(f"{self.parameters[key]:.5f} +- {self.errors[key]:.5f}")

            if key == 'fpfs':
                ranges.append([
                    self.parameters[key] - 3*self.errors[key],
                    self.parameters[key] + 3*self.errors[key]
                ])
            else:
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


#########################################################
# UTILITY FUNCTIONS + PLOTTING
def time_bin(time, flux, dt=1./(60*24)):
    bins = int(np.floor((max(time) - min(time))/dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    bstds = np.zeros(bins)
    for i in range(bins):
        mask = (time >= (min(time)+i*dt)) & (time < (min(time)+(i+1)*dt))
        if mask.sum() > 0:
            bflux[i] = np.nanmean(flux[mask])
            btime[i] = np.nanmean(time[mask])
            bstds[i] = np.nanstd(flux[mask])/(1+mask.sum())**0.5
    zmask = (bflux==0) | (btime==0) | np.isnan(bflux) | np.isnan(btime)
    return btime[~zmask], bflux[~zmask], bstds[~zmask]

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

# rip off of corner.py
def corner(xs, bins=20, range=None, weights=None, color="k", hist_bin_factor=1,
           smooth=None, smooth1d=None, levels=[1],
           labels=None, label_kwargs=None,
           titles=[], title_fmt=".2f", title_kwargs=None,
           truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=None, verbose=False, fig=None,
           max_n_ticks=5, top_ticks=False, use_math_text=False, reverse=False,
           hist_kwargs=None, **hist2d_kwargs):

    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()

    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Parse the parameter ranges.
    if range is None:
        if "extents" in hist2d_kwargs:
            logging.warn("Deprecated keyword argument 'extents'. "
                         "Use 'range' instead.")
            range = hist2d_kwargs.pop("extents")
        else:
            range = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in range], dtype=bool)
            if np.any(m):
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic range. "
                                  "Please provide a `range` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))

    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range = list(range)
        for i, _ in enumerate(range):
            try:
                emin, emax = range[i]
            except TypeError:
                q = [0.5 - 0.5*range[i], 0.5 + 0.5*range[i]]
                range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")
    try:
        hist_bin_factor = [float(hist_bin_factor) for _ in range]
    except TypeError:
        if len(hist_bin_factor) != len(range):
            raise ValueError("Dimension mismatch between hist_bin_factor and "
                             "range")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor   # size of left/bottom margin
        trdim = 0.5 * factor   # size of top/right margin
    else:
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            if reverse:
                ax = axes[K-i-1, K-i-1]
            else:
                ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            bins_1d = int(max(1, np.round(hist_bin_factor[i] * bins[i])))
            n, _, _ = ax.hist(x, bins=bins_1d, weights=weights,
                              range=np.sort(range[i]), **hist_kwargs)
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(x, bins=bins[i], weights=weights,
                                range=np.sort(range[i]))
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if len(titles):
            title = None
            ax.set_title(titles[i], **title_kwargs)

        # Set up the axes.
        ax.set_xlim(range[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            if reverse:
                ax.xaxis.tick_top()
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                if reverse:
                    ax.set_title(labels[i], y=1.25, **label_kwargs)
                else:
                    ax.set_xlabel(labels[i], **label_kwargs)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K-i-1, K-j-1]
                else:
                    ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(y, x, ax=ax, range=[range[j], range[i]], weights=weights,
                    smooth=smooth, bins=[bins[j], bins[i]], levels=levels,
                    **hist2d_kwargs)

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

            if j > 0:
                ax.set_yticklabels([])
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

    return fig

def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

def hist2d(x, y, bins=20, range=None, levels=[2],
           ax=None, plot_datapoints=True, plot_contours=True, 
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
            **kwargs):
    if ax is None:
        ax = pl.gca()

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["s"] = data_kwargs.get("s", 2.0)
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.2)
        ax.scatter(x, y, marker="o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()

        # mask data in range + chi2
        maskx = (x > range[0][0]) & (x < range[0][1])
        masky = (y > range[1][0]) & (y < range[1][1])
        mask = maskx & masky & (data_kwargs['c'] < data_kwargs['vmax']*1.2)

        # approx posterior + smooth
        xg, yg = np.meshgrid( np.linspace(x[mask].min(),x[mask].max(),256), np.linspace(y[mask].min(),y[mask].max(),256) )
        cg = griddata(np.vstack([x[mask],y[mask]]).T, data_kwargs['c'][mask], (xg,yg), method='nearest', rescale=True)
        scg = gaussian_filter(cg,sigma=15)
        
        # plot
        try:
            ax.contour(xg, yg, scg*np.nanmin(cg)/np.nanmin(scg), np.sort(levels), **contour_kwargs, vmin=data_kwargs['vmin'], vmax=data_kwargs['vmax'])        
        except:
            pass
    
    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
