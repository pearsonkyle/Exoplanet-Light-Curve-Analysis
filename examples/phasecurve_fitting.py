import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from elca.tools import pc_fitter

# load data for wasp19b in irac4.5 using nearest neighbor regression
alldata = np.loadtxt("spitzer/alldata.txt")
data = {
    'aper_time':alldata[:,0],
    'aper_flux':alldata[:,1],
    'aper_err':alldata[:,2],
    'aper_xcent':alldata[:,3],
    'aper_ycent':alldata[:,4],
    'aper_npp':alldata[:,5]
}
pars = json.load(open("spitzer/prior.json","r"))

# prior
mybounds = {
    'rprs':[0,pars['rprs']*1.25],
    'tmid':[pars['tmid']-0.01, pars['tmid']+0.01],
    'ars':[pars['ars']-1, pars['ars']+1],

    'fpfs':[0.1*pars['fpfs'],2*pars['fpfs']],
    #'omega': [prior['omega']-25,prior['omega']+25],
    #'ecc': [0,0.05],

    #'c0':[-1,1],
    'c1':[-1,1], # set automatically between 0.5*eclipse depth (Day-night amplitude)
    'c2':[-1,1], # set automatically between 0.2*eclipse depth - 0.1*ed (offset)
    #'c3':[-1,1],
    #'c4':[-1,1]
}

syspars = np.array([
    data['aper_xcent'],
    data['aper_ycent'],
    data['aper_npp']]).T

myfit = pc_fitter(
    data['aper_time'], 
    data['aper_flux'], 
    data['aper_err'], 
    pars, mybounds, syspars, mode='ns',
    verbose=True)

myfit.plot_triangle()
plt.show()

myfit.plot_bestfit()
plt.show()