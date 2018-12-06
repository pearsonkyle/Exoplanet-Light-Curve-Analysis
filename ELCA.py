import ctypes
import corner
import numpy as np
from itertools import chain
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
from os import environ, path, mkdir
from shutil import rmtree
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.optimize import minimize, least_squares, curve_fit

# make sure you compile the C library and add it to LD_LIBRARY_PATH
# then install the pymultinest package
# follow instructions on the github page to do so
import pymultinest
# TODO add libmultinest.so to LD_LIBRARY_PATH
# add bashrc export path

########################################################
# LOAD IN TRANSIT FUNCTION FROM C

# define 1d array pointer in python
array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

# load library
lib_trans = np.ctypeslib.load_library('lib_transit.so',environ['ELCA_PATH'])

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


# also take *args for multinest?
def transit(**kwargs):
    '''
        Analytic expression for a transiting extrasolar planet.
        Polynomial + Exponential equation included for modeling
        the systematic trends due to airmass extinction

        CITATION: http://adsabs.harvard.edu/abs/2002ApJ...580L.171M

        FUNCTION ARGUMENTS (KEYWORD - DESCRIPTION)
            time - list/array of time values in days
            values - dictionary with keys:{rp,ar,per,inc,u1,u2,ecc,ome,tm,a0,a1,a2}
            airmass - values for exponential airmass function (if neccessary)

        EXAMPLE:
            from ELCA import transit
            import numpy as np

            t = np.linspace(0.85,1.05,200)

            init = { 'rp':0.06, 'ar':14.07, 'per':3.336817,
                     'inc':88.75, 'u1': 0.3, 'u2': 0, 'ecc':0,
                     'ome':0, 'a0':1, 'a1':0, 'a2':0, 'tm':0.95 }

            data = transit(time=t, values=init)


        PARAMETER ESTIMATION & MODEL FITTING
            freevals - list/array of parameter values that will be changed in minimization function    ex: (0.05,0.95)
            freekeys - tuple/list of keys that correspond to which params/freevals are what parameter  ex: ('rp','tm')
            (see lc_fitter class for how to use these)
    '''
    time = np.require(kwargs['time'],dtype=ctypes.c_double,requirements='C')
    vd = kwargs['values']

    keys = ['rp','ar','per','inc','u1','u2','ecc','ome','tm']


    #  removed due to error with len(params from NS,since c_double)
    #if len( kwargs.get('freevals',[]) )  == len( kwargs.get('freekeys',[1]) ):
    try:
        # add free vals to vals
        if 'freekeys' in kwargs.keys():
            for i in range(len(kwargs['freekeys'])):
                vd[ kwargs['freekeys'][i] ] = kwargs['freevals'][i]
        else:
            pass
    except:
        print('freekeys and freevals not the same length, check kwargs.keys()')
        import pdb; pdb.set_trace()


    # create list of parameters, order matters
    vals =  [vd.get(keys[i],0) for i in range(len(keys)) ]

    # alloc data for model
    model = np.zeros(len(time),dtype=ctypes.c_double)
    model = np.require(model,dtype=ctypes.c_double,requirements='C')

    # explicit function call: occultquadC( t,rp,ar,per,inc,u1,u2,ecc,ome,tm, n,model )
    occultquadC( time, *vals, len(time),model ) # saves transitcurve to model

    # airmass function
    a0 = vd.get('a0',1)
    a1 = vd.get('a1',0)
    a2 = vd.get('a2',0)

    # exponential airmass function (change to accept list as well?)
    if isinstance(kwargs.get('airmass',0),np.ndarray):
        model *= (a0 * np.exp(kwargs['airmass']*a1))
    else:     # polynomial funcion
        model *= (a0 + time*a1 + time*time*a2)

    return model

class lc_fitter(object):
    def __init__(self,t,data,dataerr=None,init=None,bounds=None,airmass=False,nested=False,plot=False,loss='linear',
                    live_points=250, evidence_tol=0.5):

        self.t = np.array(t)
        self.y = np.array(data)

        self.init = init
        self.bounds = bounds

        self.loss = loss

        # add airmass and exponential function if available
        if isinstance(airmass,list) or isinstance(airmass,np.ndarray):
            self.airmass = np.array(airmass)
        else:
            self.airmass = False

        if type(dataerr) == type(None):
            self.yerr = np.ones(len(t))
        else:
            self.yerr = dataerr

        self.data = { 'LS':{},'NS':{} }

        self.fit_ls()
        if nested:
            # set up directory for nested sampler results
            if not path.exists("chains"): 
                mkdir("chains")
            else:
                rmtree("chains")
                mkdir("chains")

            self.live_points = live_points
            self.ee = 0.1 # evidence tolerance
            self.fit_ns()
        # Nested sampling coming soon

    def fit_ls(self):
        # prep data for **kwargs
        fixeddict = {};
        freekeys = [];

        # hopefully each iteration of for loop yields the same order (issue fixed in python 3.6)
        initvals = [ self.init[key] for key in self.bounds.keys() ]
        freekeys = tuple([ key for key in self.bounds.keys() ])
        lo,up = zip(*[ self.bounds[key] for key in self.bounds.keys() ])
        # TODO double check the order of up and lo to leastsq input

        # add fixed values to dictonary for transit function
        for i in self.init.keys():
            if i in freekeys: pass
            else: fixeddict[i] = self.init[i]

        # assemble function input for transit()
        kargs = {'freekeys':freekeys, 'values':fixeddict}
        if isinstance(self.airmass,np.ndarray): kargs['airmass'] = self.airmass

        def fcn2min(params,**kwargs):
            # define objective function: returns the array to be minimized
            # minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
            model = transit(time=self.t,freevals=params,**kwargs)
            return (model - self.y)/self.yerr


        # params -> list of free parameters
        # kwargs -> keys for params, values of fixed parameters
        res = least_squares(fcn2min,x0=initvals,kwargs=kargs,bounds=[lo,up],loss=self.loss)  #method='lm' does not support bounds
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        self.data['LS']['res'] = res
        self.data['LS']['finalmodel'] = transit(time=self.t,freevals=res.x,**kargs)
        self.data['LS']['residuals'] = self.y - self.data['LS']['finalmodel']

        try:
            # compute uncertainties from covariance matrix from jacobian squared
            perr = np.sqrt( np.diag( np.linalg.inv( np.dot( res.jac.T, res.jac) ) )) # diag[ (J.T*J)^-1 ]^0.5
            #perr *= (self.data['LS']['residuals']**2).sum()/( len(self.y) - len(freekeys ) ) # scale by variance, variance is very small?
        except:
            perr = np.array(initvals)*0.01
        
        perr[perr==0] = 0.01
        perr[np.isnan(perr)] = 0.01

        # add the best fit parameters to the fixed dictionary
        errordict = {}
        for i in range(len(res.x)):
            fixeddict[ freekeys[i] ] = res.x[i]
            errordict[ freekeys[i] ] = perr[i]

        # save data
        self.data['LS']['parameters'] = fixeddict
        self.data['LS']['errors'] = errordict
        self.data['freekeys'] = freekeys

        # compute transit and airmass model separately
        keys = ['rp','ar','per','inc','u1','u2','ecc','ome','tm']
        vals = {};
        for k in keys:
            vals[k] = fixeddict[k]
        self.data['LS']['transit'] = transit(time=self.t,values=vals)


        amkeys = ['a0','a1','a2']
        vals = {};
        for k in amkeys:
            vals[k] = fixeddict[k]

        if isinstance(self.airmass,np.ndarray):
            self.data['LS']['airmass'] = transit(time=self.t,values=vals,airmass=self.airmass)
        else:
            self.data['LS']['airmass'] = transit(time=self.t,values=vals)

        # COMPUTE PHASE
        self.data['LS']['phase'] = (self.t - self.data['LS']['parameters']['tm']) / self.data['LS']['parameters']['per']

        # TODO COMPUTE CHI2

    def fit_ns(self):
        SIGMA_TOL = 25 # boundary for hypercube parameter space

        # hopefully each iteration of for loop yields the same order (issue fixed in python 3.6)
        freekeys = tuple([ key for key in self.bounds.keys() ])
        lo,up = zip(*[ self.bounds[key] for key in self.bounds.keys() ])

        # adjust bounds based on uncertainties of LS if unbounded
        paramlims = []
        for i in range(len(freekeys)):
            
            # uncertainties are too unreliable from LS to use this idea
            #nbound = self.data['LS']['errors'][freekeys[i]]*SIGMA_TOL

            if np.isinf(lo[i]):
                paramlims.append( min(0, self.data['LS']['parameters'][freekeys[i]]-self.data['LS']['parameters'][freekeys[i]]) )
            else:
                paramlims.append( lo[i] )
                
            if np.isinf(up[i]):
                paramlims.append( max( self.data['LS']['parameters'][freekeys[i]]+self.data['LS']['parameters'][freekeys[i]],1) )
            else:
                paramlims.append( up[i] )

        print(freekeys)
        print(paramlims)
        # add fixed values to dictonary for transit function
        fixeddict = {};
        for i in self.init.keys():
            if i in freekeys: pass
            else: fixeddict[i] = self.init[i]

        # assemble function kwargs for transit()
        kargs = {'freekeys':freekeys, 'values':fixeddict}

        def myprior_transit(cube, ndim, n_params,paramlimits=paramlims):
            '''This transforms a unit cube into the dimensions of your prior
            space to search. Make sure you do this right!'''
            for i in range(len(freekeys)): # for only the free params
                cube[i] = (paramlimits[2*i+1] - paramlimits[2*i])*cube[i]+paramlimits[2*i]

        # loss functions to diminish outlier influence
        loss = {
            'linear': lambda z: z,
            'soft_l1' : lambda z : 2 * ((1 + z)**0.5 - 1),
        }
        def myloglike(cube, ndim, n_params):
            '''The most important function. What is your likelihood function?
            I have chosen a simple chi2 gaussian errors likelihood here.'''
            model = transit(time=self.t,freevals=cube,**kargs)
            chi2 = ((self.y-model)/self.yerr)**2
            loglike = -np.sum( loss[self.loss](chi2) )
            return loglike


        '''----------------------------------------------------
        Now we set up the multinest routine
        ----------------------------------------------------'''
        # number of dimensions our problem has
        ndim = len(freekeys)
        n_params = len(freekeys) #oddly, this needs to be specified

        pymultinest.run(myloglike, myprior_transit, n_params, evidence_tolerance=self.ee, 
            resume = False, verbose = False, sampling_efficiency=0.5, n_live_points=self.live_points, max_iter=150000)

        # lets analyse the results
        a = pymultinest.Analyzer(n_params = n_params) #retrieves the data that has been written to hard drive
        self.data['NS']['analyzer'] = a

        # information from MultiNest
        self.data['NS']['stats'] = self.data['NS']['analyzer'].get_stats()
        self.data['NS']['posteriors'] = self.data['NS']['analyzer'].get_data()[:,2:]
        values = self.data['NS']['stats']['marginals'] # gets the marginalized posterior probability distributions

        # add the best fit parameters to the fixed dictionary
        errordict = {}
        for i in range(ndim):
            fixeddict[ freekeys[i] ] = values[i]['median']
            errordict[ freekeys[i] ] = values[i]['sigma']

        # save data
        self.data['NS']['parameters'] = fixeddict
        self.data['NS']['errors'] = errordict
        self.data['NS']['finalmodel'] = transit(time=self.t,values=fixeddict)
        self.data['NS']['residuals'] = self.y - self.data['NS']['finalmodel']

        # compute transit and airmass model separately
        keys = ['rp','ar','per','inc','u1','u2','ecc','ome','tm']
        vals = {};
        for k in keys:
            vals[k] = fixeddict[k]
        self.data['NS']['transit'] = transit(time=self.t,values=vals)

        amkeys = ['a0','a1','a2']
        vals = {};
        for k in amkeys:
            vals[k] = fixeddict[k]

        if isinstance(self.airmass,np.ndarray):
            self.data['NS']['airmass'] = transit(time=self.t,values=vals,airmass=self.airmass)
        else:
            self.data['NS']['airmass'] = transit(time=self.t,values=vals)

        # COMPUTE PHASE
        self.data['NS']['phase'] = (self.t - self.data['NS']['parameters']['tm']) / self.data['NS']['parameters']['per']

    def plot_results(self, detrend=False, phase=False, t='LS',show=False,title='Lightcurve Fit',savefile=None):
        '''
            Detrend - Removes airmass function
            Phase - plot in phase units
            t - Type LS for least squares, NS for nested sampling
            show - show plot at the end
            title - plot title (also the saved file name)
            save - save output or not
            output - type of file to save as
        '''
        from mpl_toolkits.axes_grid.inset_locator import inset_axes
        f = plt.figure( figsize=(12,7) )
        f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid( (4,5), (0,0), colspan=5,rowspan=3 )
        ax_res = plt.subplot2grid( (4,5), (3,0), colspan=5, rowspan=1 )
        inset_axes = inset_axes(ax_lc,
                    width=2, # width = "30%" of parent_bbox
                    height=2, # height : 1 inch
                    loc=4)

        f.suptitle(title,fontsize=20)

        if phase:
            x = self.data[t]['phase']
            ax_res.set_xlabel('Phase')

            # make symmetric about 0 phase
            maxdist = max( np.abs( self.data[t]['phase'][0]),  self.data[t]['phase'][-1] )
            ax_res.set_xlim([-maxdist, maxdist])
            ax_lc.set_xlim([-maxdist, maxdist])
        else:
            x = self.t
            ax_res.set_xlabel('Time (RJD)')

            # make symetric about mid point
            d1 = np.abs( self.t[0] - self.data[t]['parameters']['tm'] )
            d2 = np.abs( self.t[-1] - self.data[t]['parameters']['tm'] )
            low = self.data[t]['parameters']['tm'] - max(d1,d2)
            up = self.data[t]['parameters']['tm'] + max(d1,d2)
            ax_res.set_xlim([low,up])
            ax_lc.set_xlim([low,up])

        # residual histogram
        # bins up to 3 std of Residuals
        maxbs = np.round(3*np.std(self.data['LS']['residuals'])*1e6,-2)
        bins = np.linspace(-maxbs,maxbs, np.sqrt(x.shape[0]) )
        inset_axes.hist( self.data[t]['residuals']*1e6,bins=bins, orientation="horizontal",color="black" )
        inset_axes.get_xaxis().set_visible(False)
        inset_axes.set_title('Residuals (PPM)')

        # residual plot
        ax_res.plot(x,self.data[t]['residuals']*1e6,'ko')
        ax_res.plot(x,np.zeros(len(self.t)),'r-',lw=2,alpha=0.85 )
        ax_res.set_ylabel('Rel. Flux (PPM)')
        ax_res.set_ylim([-maxbs,maxbs])

        # light curve plot
        if detrend:
            ax_lc.errorbar( x, self.y/self.data[t]['airmass'], yerr=self.yerr/self.data[t]['airmass'], ls='none', marker='o', color='black')
            ax_lc.plot( x, self.data[t]['transit'],'r-', lw=2)
            # TODO propogate uncertainties
        else:
            ax_lc.errorbar( x, self.y, yerr=self.yerr, ls='none', marker='o', color='black')
            ax_lc.plot( x, self.data[t]['finalmodel'],'r-', lw=2)
            # TODO show both fits

        ax_lc.set_ylabel('Relative Flux')
        ax_lc.get_xaxis().set_visible(False)

        if show:
            plt.show()

        if savefile:
            plt.savefig(savefile)
            plt.close()

    def plot_posteriors(self,show=False,title='Lightcurve Posterior',savefile=None):
        f = corner.corner(self.data['NS']['posteriors'], labels=tuple([ key for key in self.bounds.keys() ]),bins=int(np.sqrt(self.data['NS']['posteriors'].shape[0])), plot_contours=False, plot_density=False)
        # TODO use math text 
        f.suptitle(title)
        if show:
            plt.show()

        if savefile:
            plt.savefig(savefile )
            plt.close()


if __name__ == "__main__":

    t = np.linspace(0.9,1.1, 0.2*(1./2)*24*60 )

    init = { 'rp':0.05, 'ar':14.0,       # Rp/Rs, a/Rs
             'per':3.5, 'inc':87.5,  # Period (days), Inclination
             'u1': 0.3, 'u2': 0,          # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,            # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,              # Airmass extinction terms
             'a2':0, 'tm':1 }          # tm = Mid Transit time (Days)

    # only report params with bounds, all others will be fixed to initial value
    mybounds = {
              'rp':[0,1],
              'tm':[min(t),max(t)],
              'a0':[-np.inf,np.inf],
              'a1':[-np.inf,np.inf]
              }

    # GENERATE NOISY DATA
    data = transit(time=t, values=init) + np.random.normal(0, 4e-4, len(t))
    dataerr = np.random.normal(400e-6, 50e-6, len(t))

    myfit = lc_fitter(t,data,
                        dataerr=dataerr,
                        init= init,
                        bounds= mybounds,
                        nested=True,
                        loss='soft_l1'
                        )
    for k in myfit.data['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit.data['NS']['parameters'][k],myfit.data['NS']['errors'][k]) )

    myfit.plot_results(show=True,t='NS')
    myfit.plot_posteriors(show=True)



    dude()
    
    f,ax = plt.subplots(2)
    # for various signal to noise ratios
    # various number of live points
    snrs = np.logspace(0,1.25,8)

    rp = []
    err = []
    tmid = []
    terr = []
    for i in snrs:
        
        # GENERATE NOISY DATA
        data = transit(time=t, values=init) + np.random.normal(0, (init['rp']**2)/i, len(t))
        dataerr = np.random.normal( 0.5*(init['rp']**2)/i, 50e-6, len(t))
        
        myfit = lc_fitter(t,data,
                            dataerr=dataerr,
                            init= init,
                            bounds= mybounds,
                            #nested=True
                            )

        rp.append( myfit.data['LS']['parameters']['rp'] ) 
        err.append( myfit.data['LS']['errors']['rp'] ) 
        tmid.append( myfit.data['LS']['parameters']['tm'] )
        terr.append( myfit.data['LS']['errors']['tm'] )
        del myfit 

    ax[0].errorbar(snrs, rp,yerr=err,ls=None, label='SNR: {:.1f}'.format(i))
    ax[1].errorbar(snrs, (np.array(tmid)-init['tm'])*24*60 ,yerr=np.array(terr)*24*60,ls=None, label='SNR: {:.1f}'.format(i))
        
    ax[0].axhline( init['rp'], color='black',ls='--', label='True')
    ax[0].set_xlabel('SNR')
    ax[0].legend(loc='best')
    ax[0].set_ylabel('Rp/Rs')

    ax[1].axhline( 0, color='black',ls='--', label='True')
    ax[1].set_xlabel('SNR')
    ax[1].legend(loc='best')
    ax[1].set_ylabel('T_mid - True (min)')
    plt.show()
    
