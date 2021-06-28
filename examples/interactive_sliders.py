''' 
 Use: bokeh serve interactive_sliders.py
'''
import json
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

from elca.tools import phasecurve

rsun = 6.955e8 # m
rjup = 7.1492e7 # m
au = 1.496e11 # m 

priors = json.load(open('spitzer/prior.json','r'))

prior = { 
    # transit 
    'rprs': priors['rprs'],
    'ars': priors['ars'],
    'per': priors['per'],
    'inc': priors['inc'],
    'tmid':0.25-priors['per'], 

    # eclipse 
    'fpfs': 0.1,
    'omega': priors.get('omega',0), 
    'ecc': priors['ecc'],

    # limb darkening (linear, quadratic)
    'u1': 0, 'u2': 0, 

    # phase curve amplitudes
    'c1':0, 'c2':0.,
}

time = np.linspace(0,priors['per']*0.95, 10000) # [day]
data = phasecurve(time, prior)
source = ColumnDataSource(data=dict(x=time, y=data))

data += np.random.normal(0, 100e-6, len(time))
source_noisy = ColumnDataSource(data=dict(x=time, y=data))

# Set up plot
plot = figure(plot_height=420, plot_width=666, title="Interactive Phase Curve",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[time.min(), time.max()], y_range=[0.98, 1.01])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
#plot.circle('x', 'y', source=source_noisy, color='black')
plot.xaxis.axis_label = "Time [day]"
plot.yaxis.axis_label = "Relative Flux"

rprs = Slider(title="Transit Depth (Rp/Rs)", value=prior['rprs'], start=0.01, end=0.15, step=0.001)
fpfs = Slider(title="Flux Ratio (Fp/Fs)", value=0.1, start=0.001, end=0.25, step=0.001)
tmid = Slider(title="Mid-Transit (Tmid)", value=prior['tmid'], start=0, end=0.5, step=0.01)
per = Slider(title="Period", value=prior['per'], start=prior['per']*0.8, end=prior['per']*1.2, step=prior['per']*0.01)
ecc = Slider(title="Eccentricity", value=prior['ecc'], start=0, end=0.15, step=0.01)
ome = Slider(title="Omega", value=prior['omega'], start=0, end=360, step=0.1)
amp = Slider(title="Day-Night Amplitude (c1)", value=prior['c1'], start=0, end=0.5*prior['fpfs']*prior['rprs']**2, step=0.0001)
#offset = Slider(title="Offset Parameter (c2)", value=prior['c2'], start=0, end=0.5*prior['fpfs']*prior['rprs']**2, step=0.0001)
width = Slider(title="Day-Night Width (c2)", value=prior['c2'], start=-0.05*prior['fpfs']*prior['rprs']**2, end=0.05*prior['fpfs']*prior['rprs']**2, step=0.00001)
#offset2 = Slider(title="Offset 2 (c4)", value=prior['c4'], start=-0.05*prior['fpfs']*prior['rprs']**2, end=0.05*prior['fpfs']*prior['rprs']**2, step=0.00001)

def update_data(attrname, old, new):
    prior['rprs'] = rprs.value
    prior['fpfs'] = fpfs.value
    amp.end = 0.5*prior['fpfs']*prior['rprs']**2
    prior['tmid'] = tmid.value
    prior['per'] = per.value
    prior['ecc'] = ecc.value
    prior['omega'] = ome.value
    prior['c1'] = amp.value
    prior['c2'] = width.value
    source.data = dict(x=time, y=phasecurve(time, prior))

for w in [rprs, fpfs, tmid, per, ome, ecc, amp, width]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(rprs, tmid, per, fpfs, ome, ecc, amp, width)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"