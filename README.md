# Exoplanet Light Curve Analysis with Nested Sampler

A python package for modeling exoplanet light curves. The transit function is based on the analytic expressions of Mandel and Agol 2002 and is re-written in C for microsecond execution speeds. This branch uses the mutlimodal nested sampling algorithm (https://arxiv.org/abs/1306.2144) to find a global solution. 

Check out the "nested" branch for a global solver using the Multinest library for nested sampling. 

- Simple transit generator
- Easily create noisy datasets
- Parameter optimization and uncertainty estimation (powered by Scipy)
    - For posterior parameter distributions check out the "nested" branch

![alt text](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis/raw/nested/lightcurve_fit.png "Light Curve Modeling")
 
## Running the package
```python

if __name__ == "__main__":

    t = np.linspace(0.85,1.05,400)

    init = { 'rp':0.06, 'ar':14.07,       # Rp/Rs, a/Rs
             'per':3.336817, 'inc':87.5,  # Period (days), Inclination
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
    data = transit(time=t, values=init) + np.random.normal(0, 4e-4, len(t))
    dataerr = np.random.normal(400e-6, 50e-6, len(t))

    myfit = lc_fitter(t,data,
                        dataerr=dataerr,
                        init= init,
                        bounds= mybounds,
                        nested=True
                        )


    for k in myfit.data['LS']['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit.data['NS']['parameters'][k],myfit.data['NS']['errors'][k]) )

    myfit.plot_results(show=True,t='NS')
    myfit.plot_posteriors(show=True)
```

## Output

```python 
myfit = {
    'freekeys': list,           # parameter dictionary keys that correspond to the parameters being solved for
                                # obeys the same format as 'parameters' below
    'LS': {
        'res': ndarray,         # Optimize Result from scipy.optimize.least_squares fit
        'finalmodel': ndarray,  # best fit model of light curve (transit+detrending model)
        'residuals': ndarray,   # residual from light curve fit (data-finalmodel)
        'transit': ndarray,     # just the transit model with no system trend
        'phase': ndarray,       # lightcurve phase calculation based on fit mid transit
        'parameters':{             
            'rp': float, 'ar': float,   # Rp/Rs, a/Rs
            'per': float, 'inc': float, # Period (days), Inclination
            'u1': float, 'u2':  float,  # limb darkening (linear, quadratic)
            'ecc': float, 'ome': float, # Eccentricity, Arg of periastron
            'tm': float                 # tm = Mid Transit time (Days)
            },
        'errors':{
            # same format as parameters
            # uncertainty estimate on parameters from posterior distributions
        }               
    },

    'NS': {
        'res': ndarray,         # Optimize Result from scipy.optimize.least_squares fit
        'finalmodel': ndarray,  # best fit model of light curve (transit+detrending model)
        'residuals': ndarray,   # residual from light curve fit (data-finalmodel)
        'transit': ndarray,     # just the transit model with no system trend
        'phase': ndarray,       # lightcurve phase calculation based on fit mid transit
        'parameters':{     
            'rp','ar','per','inc','u1','u2','ecc','ome','tm' : float 
            },
        'errors':{
            # same format as parameters
            # uncertainty estimate on parameters from posterior distributions
        },
        'posteriors': ndarray,   # parameter space evaluations [N,#evals] (N=# free parameters)
        'stats': {               # Output from get_stats_mode() in PyMultiNest 
            'modes',             # see: https://johannesbuchner.github.io/PyMultiNest/pymultinest_analyse.html 
            'marginals',
            'nested sampling global log-evidence', 
            'nested sampling global log-evidence error', 
            'global evidence', 'global evidence error', 
            'nested importance sampling global log-evidence', 
            'nested importance sampling global log-evidence error',
        },                       
                                 
}
```

## Parameter Uncertainty Estimation 
The nested sampling algorithm enables uncertainty estimations for each parameter from their posterior distribution. 
![alt text](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis/raw/nested/lightcurve_posterior.png "Posterior Distribution")


## Set up and install from scratch

Clone the git repo
```
cd $HOME
git clone https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis.git
```
Rename for simplicity later
```
mv Exoplanet-Light-Curve-Analysis ELCA
```
Compile the C code. Python will link to this file later
```
cd ELCA/util_lib
chmod +x compile
./compile
```
Create two PATH variables so that this code can be accessed from anywhere on your computer. This keeps all of the codes in one location for easy updating and referencing.
```
cd $HOME
gedit .bashrc
```
**add the two lines below to your .bash_profile or .bashrc**
```
export PYTHONPATH=$HOME/ELCA:$PYTHONPATH
export ELCA_PATH=$HOME/ELCA/util_lib
```
Update your .bashrc file after adding those two lines
```
source .bashrc
```
DISCLAIMER:
If you did not download the python package to your $HOME directory then you will need to make changes to where the PATHS point. If you do not have a .bashrc file or .bash_profile file then you may need to create one. If you're installing this on windows you may have some difficulty and I would reccomend looking at this: http://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/ Email me for more windows instructions if you're on a windows system < v10.
