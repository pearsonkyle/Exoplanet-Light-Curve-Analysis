import numpy as np
import matplotlib.pyplot as plt

from elca.tools import transit, lc_fitter

if __name__ == "__main__":

    prior = { 
        'rprs':0.02,        # Rp/Rs
        'ars':5.344,        # a/Rs
        'per':2.15,     # Period [day]
        'inc':88.98,        # Inclination [deg]
        'u1': 0.3, 'u2': 0.1, # limb darkening (linear, quadratic)
        'ecc':0.159,            # Eccentricity
        'omega':0,          # Arg of periastron
        'tmid':0.75       # time of mid transit [day]
    }

    # GENERATE NOISY DATA
    time = np.linspace(0.5,1,500) # [day]
    data = transit(time, prior) + np.random.normal(0, 2e-4, len(time))
    dataerr = np.random.normal(300e-6, 50e-6, len(time))

    mybounds = {
        'rprs':[0,2*prior['rprs']],
        'tmid':[min(time),max(time)],
        'ars':[prior['ars']*0.9, prior['ars']*1.1 ], 
        #'inc':[85,87]
    }

    myfit = lc_fitter(time, data, dataerr, prior, mybounds, verbose=True)

    for k in myfit.bounds.keys():
        print("{:.6f} +- {}".format( myfit.parameters[k], myfit.errors[k]))

    fig,axs = myfit.plot_bestfit()

    myfit.plot_triangle()
    plt.show()