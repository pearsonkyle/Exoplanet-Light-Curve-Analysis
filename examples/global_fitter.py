import copy
import numpy as np
import matplotlib.pyplot as plt

from elca.tools import transit, glc_fitter

if __name__ == "__main__":

    import time as timer
    t1 = timer.time()

    # simulate input data
    epochs = np.random.choice(np.arange(100), 15, replace=False)
    input_data = []

    for i, epoch in enumerate(epochs):

        nobs = np.random.randint(50) + 100
        phase = np.linspace(-0.02-0.01*np.random.random(), 0.02+0.01*np.random.random(), nobs)
        
        prior = {
            'rprs':0.1, # Rp/Rs
            'ars':14.25,        # a/Rs
            'per':3.5,          # Period [day]
            'inc':87.5,         # Inclination [deg]
            'u1': np.random.random()*0.1, 'u2': np.random.random()*0.3+0.1, 
            #'u2': 3.9, 'u3': -1.5,  # limb darkening (nonlinear)
            'ecc':0,            # Eccentricity
            'omega':0,          # Arg of periastron
            'tmid':1,        # time of mid transit [day],

            'a1': 1000+1000*np.random.random(),
            'a2':-0.25 + 0.1*np.random.random()
        }

        time = prior['tmid'] + prior['per']*(phase+epoch)
        stime = time-time[0]
        alt = 90* np.cos(4*stime-np.pi/6)
        airmass = 1./np.cos( np.deg2rad(90-alt))
        model = transit(time, prior)*np.exp(prior['a2']*airmass)*prior['a1']

        flux = model*np.random.normal(1, np.mean(np.sqrt(model)/model)*0.1, model.shape)
        ferr = 0.1*flux**0.5

        input_data.append({
            'time':time,
            'flux':flux,
            'ferr':ferr,
            'airmass':airmass,
            'priors':prior
        })

    # shared properties between light curves
    global_bounds = {
        'per':[3.5-0.001,3.5+0.001],
        'tmid':[1-0.01,1+0.01],
        'ars':[13,15],
    }

    # individual properties, dict or list of dicts
    local_bounds = {
        'rprs':[0,0.2],
        #'a1': automagically solved for since it's perfectly correlated to a2
        'a2':[-0.5,0]
    }

    print('epochs:',epochs)
    myfit = glc_fitter(input_data, global_bounds, local_bounds, individual_fit=True)

    t2 = timer.time()
    print(f"Time elapsed: {(t2-t1)/60} min")

    fig = myfit.plot_triangle()
    plt.savefig("global_posterior.png")
    plt.close()

    fig = myfit.plot_bestfit()
    plt.savefig("global_bestfit.png")
    plt.close()

    # lc - min
    # 3 - 5
    # 9 - 24
    # 15 - 
    # 21 - ?
    # convergence gets tough with lots of light curves
    # better to detrend first