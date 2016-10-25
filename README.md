# Exoplanet-Light-Curve-Analysis

A python package for modeling exoplanet light curves. The transit function is based on the analytic expressions of Mandel and Agol 2002 and is re-written in C for microsecond execution speeds.

Functionality:
- Simple transit generator
- Easily create noisy datasets
- Parameter optimization with a simple uncertainty derivation powered by Scipy 

![ELCA](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis/blob/master/Lightcurve%20Fit.png "Light Curve Modeling")




### Set up and install from scratch

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
export LD_LIBRARY_PATH=$HOME/ELCA/util_lib:$LD_LIBRARY_PATH
```

Update your .bashrc file after adding those two lines
```
source .bashrc
```
DISCLAIMER: 
If you did not download the python package to your $HOME directory then you will need to make changes to where the PATHS point. If you do not have a .bashrc file or .bash_profile file then you may need to create one. If you're installing this on windows you may have some difficulty and I would reccomend looking at this: http://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/ Email me for more windows instructions if you're on a windows system < v10. 
