
import os
import subprocess
from setuptools import find_packages, setup
from distutils.core import Extension

# Get external packages required to be installed
dir_path = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(dir_path, 'requirements.txt')

with open(req_file, "r") as fh:
      install_requires = [line for line in map(str.lstrip, fh.read().splitlines()) if len(line) > 0
                          and not line.startswith('#')]

# Do setup
setup(
    name='elca',
    version='1.0.0',
    description='A package for analyzing light curves of transiting exoplanets',
    url='https://github.jpl.nasa.gov/kpearson/Exoplanet-Light-Curve-Analysis',
    author='Kyle Pearson',
    author_email='kpearso@jpl.nasa.gov',
    license='MIT',
    packages=find_packages(include=['elca', 'elca.*']),
    install_requires=install_requires,
    ext_modules=[ Extension(
        'elca/C/lib_transit', 
        ["elca/C/MandelTransit.c"],
        extra_compile_args=["-Ofast"])
        ],
)
