import copy
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy import constants as const

from elca.tools import transit, transit_tmid


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

if __name__ == "__main__":

    # classic transit parameters
    prior = {
        'rprs':0.03,       # Rp/Rs
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

    moon_prior['rprs'] = 0.00333
    moon_prior['radius_star'] = 1    # sun radius
    moon_prior['mass_star'] = 1      # sun mass
    moon_prior['mass_planet'] = 0.5  # jupiter mass
    moon_prior['moon_per'] = 9       # day
    moon_prior['moon_tmid'] = 1.5
    moon_prior['planet_rprs'] = prior['rprs']

    # create data
    time = np.linspace(0, 27, 100000)

    # compute transits
    planetdata = transit(time, prior)
    moondata, toffset, occulted = moon_transit(time, moon_prior)
    
    # mid-transit like phase for moon
    phase = (time - (prior['tmid']+toffset.value - 0.5*prior['per'])) / prior['per'] % 1 - 0.5

    fig, axx = plt.subplots(2, figsize=(12,5))
    axx[0].plot(time, -1e6*(1-moondata), 'b-', label='moon')
    axx[0].plot(time[occulted], -1e6*(1-moondata[occulted]), 'g.', label='moon-planet occultation')

    axx[0].plot(time, -1e6*(1-planetdata), 'r-', label='planet',alpha=0.5)
    axx[0].set_ylim([-20,1])
    axx[0].legend(loc='best')
    axx[0].set_xlabel("Time [day]")
    axx[0].set_ylabel("Relative Flux [ppm]")
    axx[0].grid(True,ls='--')

    axx[1].plot(phase, -1e6*(1-moondata),'bo')
    axx[1].plot(phase[occulted], -1e6*(1-moondata[occulted]),'go')
    axx[1].plot(phase, -1e6*(1-planetdata),'r-',alpha=0.5)
    axx[1].set_xlabel("Moon Phase")
    axx[1].set_ylabel("Relative Flux [ppm]")
    axx[1].set_ylim([-20,1])

    axx[1].set_xlim([-0.1,0.1])
    plt.tight_layout()
    plt.show()

