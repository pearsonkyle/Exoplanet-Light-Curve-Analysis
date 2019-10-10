import ctypes
import numpy as np
from os import environ
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.optimize import minimize, least_squares, curve_fit


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

    # add free vals to vals
    if len( kwargs.get('freevals',[]) )  == len( kwargs.get('freekeys',[1]) ):
        for i in range(len(kwargs['freevals'])):
            vd[ kwargs['freekeys'][i] ] = kwargs['freevals'][i]
    else:
        pass


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


    if isinstance(kwargs.get('airmass',0),np.ndarray): # exponential airmass function
        model *= (a0 * np.exp(kwargs['airmass']*a1))
    else:     # polynomial funcion
        model *= (a0 + time*a1 + time*time*a2)

    return model


# RAINBOW SPECTRUM!
def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0, 360, 360. / num_colors):
        hue = i/360.
        lightness = (50)/100.
        saturation = (100)/100.
        colors.append(hls_to_rgb(hue, lightness, saturation))
    return colors




class param:
    rp = 0
    ar = 1
    per = 2
    inc = 3
    u1 = 4
    u2 = 5
    ecc = 6
    ome = 7
    tm = 8



class lc_fitter(object):
    def __init__(self,t,data,dataerr=None,init=None,bounds=None,airmass=False,ls=True,nested=False,plot=False,loss='huber'):

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

        if ls == True: self.fit_lm()

        # Nested sampling coming soon

    def fit_lm(self):
        # prep data for **kwargs
        fixeddict = {};
        freekeys = [];

        # hopefully each iteration of for loop yields the same order (issue fixed in python 3.6)
        initvals = [ self.init[key] for key in self.bounds.keys() ]
        freekeys = tuple([ key for key in self.bounds.keys() ])
        up,lo = zip(*[ self.bounds[key] for key in self.bounds.keys() ])

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
        res = least_squares(fcn2min,x0=initvals,kwargs=kargs,bounds=[up,lo],loss=self.loss)  #method='lm' does not support bounds
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        self.data['LS']['res'] = res
        self.data['LS']['finalmodel'] = transit(time=self.t,freevals=res.x,**kargs)
        self.data['LS']['residuals'] = self.y - self.data['LS']['finalmodel']

        # compute uncertainties from covariance matrix from jacobian squared
        perr = np.sqrt( np.diag( np.linalg.inv( np.dot( res.jac.T, res.jac) ) )) # diag[ (J.T*J)^-1 ]^0.5
        #perr *= (self.data['LS']['residuals']**2).sum()/( len(self.y) - len(freekeys ) ) # scale by variance, variance is very small ~1e-8
        # TODO check the order of this, same as order of res.x?


        # add the best fit parameters to the fixed dictionary
        errordict = {}
        for i in range(len(res.x)):
            fixeddict[ freekeys[i] ] = res.x[i]
            errordict[ freekeys[i] ] = perr[i]

        # save data
        self.data['LS']['parameters'] = fixeddict
        self.data['LS']['errors'] = errordict
        self.data['LS']['freekeys'] = freekeys
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


    def plot_results(self, detrend=False, phase=False, t='LS',show=False,title='Lightcurve Fit',save=False,output='png'):
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
        bins = np.linspace(-maxbs,maxbs,7)
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

        ax_lc.set_ylabel('Relative Flux')
        ax_lc.get_xaxis().set_visible(False)

        if show:
            plt.show()

        if save:
            f.savefig(title+'.'+output)
            plt.close(f)


if __name__ == "__main__":

    t = np.linspace(0.85,1.05,200)

    init = { 'rp':0.06, 'ar':14.07,       # Rp/Rs, a/Rs
             'per':3.336817, 'inc':88.75, # Period (days), Inclination
             'u1': 0.3, 'u2': 0,          # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,            # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,              # Airmass extinction terms
             'a2':0, 'tm':0.95 }          # tm = Mid Transit time (Days)

    # only report params with bounds, all others will be fixed to initial value
    mybounds = {
              'rp':[0,1],
              'tm':[min(t),max(t)],
              'a0':[-np.inf,np.inf],
              'a1':[-np.inf,np.inf]
              }


    # GENERATE NOISY DATA
    data = transit(time=t, values=init) + np.random.normal(0, 2e-4, len(t))
    dataerr = np.random.normal(300e-6, 50e-6, len(t))

    myfit = lc_fitter(t,data,
                        dataerr=dataerr,
                        init= init,
                        bounds= mybounds,
                        )

    for k in myfit.data['LS']['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit.data['LS']['parameters'][k],myfit.data['LS']['errors'][k]) )

    myfit.plot_results(show=True,phase=True)

    
