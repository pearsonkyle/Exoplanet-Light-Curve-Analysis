import copy
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from astropy import units as u
from astropy import constants as const
from scipy.signal import savgol_filter
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from scipy import spatial

import ultranest

from elca.tools import transit, transit_tmid, corner, hist2d

def moon_transit(time, prior):
    # planet parameters
    a_planet = (((prior['per']*u.day)**2 * const.G*(prior['mass_star']*u.M_sun)/(4*np.pi))**(1./3)).to(u.AU)
    v_planet = 2*np.pi*a_planet.to(u.AU)/(prior['per']*u.day)
    r_planet = (prior['planet_rprs']*prior['radius_star']*u.R_sun).to(u.R_jup)

    # compute moon tmid for transit model
    moon_phase = (time - prior['moon_tmid'])/prior['moon_per'] % 1 # mean anomaly or something like that
    a_moon = (((prior['moon_per']*u.day)**2 * const.G*(prior['mass_planet']*u.M_jup)/(4*np.pi))**(1./3)).to(u.AU)
    ax = a_moon * np.cos(2*np.pi*moon_phase) # sky projected orbit of moon

    # tmid offset for lightcurve
    toffset = (ax/v_planet).to(u.day)
    # adjust mid-transit times of planet by this much when adding transit

    # occultation of planet and moon
    occulted = np.abs(ax.value) < r_planet.to(u.AU).value

    # prior['rprs'] is for moon 
    moondata = transit_tmid(time, prior['tmid']+toffset.value, prior)

    # mask out moon's transit when behind/infront of planet
    moondata[occulted] = 1

    return moondata, toffset, occulted

def time_bin(time, flux, dt=1./(60*24)):
    bins = int(np.floor((max(time) - min(time))/dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    bstd = np.zeros(bins)

    for i in range(bins):
        mask = (time >= (min(time)+i*dt)) & (time < (min(time)+(i+1)*dt))
        if mask.sum() > 0:
            bflux[i] = np.nanmean(flux[mask])
            btime[i] = np.nanmean(time[mask])
            bstd[i] = np.nanstd(time[mask])/(np.sqrt(1+mask.sum()))
    zmask = (bflux==0) | (btime==0) | np.isnan(bflux) | np.isnan(btime)
    return btime[~zmask], bflux[~zmask], bstd[~zmask]


class moon_fitter():

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

        def loglike(pars):
            for i in range(len(pars)):
                self.prior[freekeys[i]] = pars[i]
            model, _, _ = moon_transit(self.time, self.prior)
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
            'rprs':r'Moon R$_{p}$/R$_{s}$',
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
            'a2':r'$a_2$',
            'planet_rprs':r'R$_{p}$/R$_{s}$',
            'moon_per':r'Period',
            'moon_tmid':r'Moon T$_{mid}$'
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


if __name__ == "__main__":

    # classic transit parameters
    prior = {
        'rprs':0.05,       # Rp/Rs
        'ars':14.25,       # a/Rs
        'per':3.333,       # Period [day]
        'inc':86.5,        # Inclination [deg]
        'u1': 0.3, 'u2': 0.1, # limb darkening (linear, quadratic)
        'ecc':0,           # Eccentricity
        'omega':0,         # Arg of periastron
        'tmid':0.5,        # time of mid transit [day]
    }

    # copy and add some extra parameters for the moon
    moon_prior = copy.deepcopy(prior)

    moon_prior['rprs'] = 0.0075
    moon_prior['radius_star'] = 0.75 # sun radius
    moon_prior['mass_star'] = 1      # sun mass
    moon_prior['mass_planet'] = 0.5  # jupiter mass
    moon_prior['moon_per'] = 7       # day
    moon_prior['moon_tmid'] = 1.5
    moon_prior['planet_rprs'] = prior['rprs']

    print(f"Moon radius: {(moon_prior['rprs']*moon_prior['radius_star']*u.R_sun).to(u.km)}")

    # create data
    time = np.linspace(0, 100, 100000)

    # phase mask data around transits
    phase = (time-prior['tmid'] - 0.5*prior['per'])/prior['per'] % 1 - 0.5
    pmask = (phase>-0.15) & (phase<0.15)
    time = time[pmask]

    # compute transits
    planetdata = transit(time, prior)
    moondata, toffset, occulted = moon_transit(time, moon_prior)
    alldata = planetdata - np.abs(1-moondata)
    
    # mid-transit like phase for moon
    phase = (time - (prior['tmid']+toffset.value - 0.5*prior['per'])) / prior['per'] % 1 - 0.5

    noise = np.random.normal(0,0.1e-4,len(time))

    si = np.argsort(phase)

    bt, bf, bs  = time_bin(phase[si]*prior['per'], (moondata+noise)[si], dt=15/(60*24) )

    '''
    fig, ax = plt.subplots(1)
    ax.plot(phase, -1e6*(1-moondata + noise),'k.', label="Data", alpha=0.1)
    ax.plot(phase[si], -1e6*(1-moondata[si]),'g-', label="Truth")
    ax.plot(bt/prior['per'], -1e6*(1-bf), 'co', alpha=0.75, label="Binned") 
    ax.plot(phase[si], -1e6*(1-savgol_filter((moondata+noise)[si], int(time.max())*20+1, 2)),'m-', alpha=0.25, label="Smoothed")
    ax.set_xlabel("Moon Phase")
    ax.set_ylabel("Relative Flux [ppm]")
    ax.set_xlim([-0.06, 0.06])
    ax.set_ylim([-40, 25])
    ax.legend(loc='best')
    plt.show()
    '''

    # add noise
    alldata += noise

    bounds = {
        'rprs':[0,0.01], # moon
        'moon_per':[6.9,7.1],
        'moon_tmid':[1.49,1.51],
        'planet_rprs':[0.049,0.051],
        'tmid':[0.49,0.51],
    }

    mf = moon_fitter(time, alldata, 1e-4+noise, moon_prior, bounds, verbose=True)
    mf.plot_triangle()
    plt.show()

    # for per in np.linspace(prior['per']*1.5,prior['per']*3):
    #     for tmid in np.linspace(0,1,100):
    #         t0 = tmid*per
    #         moondata, toffset, occulted = moon_transit(time, moon_prior)
    #         #alldata = planetdata - np.abs(1-moondata)