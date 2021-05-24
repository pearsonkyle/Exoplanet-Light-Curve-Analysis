import numpy as np
import matplotlib.pyplot as plt

from elca.tools import transit_nl, lc_fitter_detrend

if __name__ == "__main__":

    prior = {
        'rprs':0.03,        # Rp/Rs
        'ars':14.25,        # a/Rs
        'per':3.336817,     # Period [day]
        'inc':87.5,        # Inclination [deg]
        'u0': 1.8, 'u1': -3.3, 'u2': 3.9, 'u3': -1.5,  # limb darkening (nonlinear)
        'ecc':0,            # Eccentricity
        'omega':0,          # Arg of periastron
        'tmid':0.75,         # time of mid transit [day],

        'a1':50,            # airmass coeffcients
        'a2':0.25
    }

    time = np.linspace(0.65,0.85,200) # [day]

    # simulate extinction from airmass
    stime = time-time[0]
    alt = 90* np.cos(4*stime-np.pi/6)
    airmass = 1./np.cos( np.deg2rad(90-alt))

    # GENERATE NOISY DATA
    data = transit_nl(time, prior)*np.exp(prior['a2']*airmass)*prior['a1']
    data += np.random.normal(0, prior['a1']*250e-6, len(time))
    dataerr = np.random.normal(300e-6, 50e-6, len(time))

    # add bounds for free parameters only
    mybounds = {
        'rprs':[0,0.1],
        'tmid':[prior['tmid']-0.01,prior['tmid']+0.01],
        'ars':[13,15],

        #'a1':automatically solved for since it's correlated to a2
        'a2':[0,1]
    }

    myfit = lc_fitter_detrend(time, data, dataerr, airmass, prior, mybounds, mode='ns', verbose=True)

    for k in myfit.bounds.keys():
        print("{:.6f} +- {}".format( myfit.parameters[k], myfit.errors[k]))

    fig,axs = myfit.plot_bestfit()
    plt.tight_layout()
    plt.show()

    myfit.plot_triangle()
    plt.show()
