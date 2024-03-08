# specabundie #

* August 12, 2022
* I. Escala (iescala@carnegiescience.edu; Princeton/Carnegie)
* Published in Escala et al. 2019, 2020a
* Based on IDL software by E. Kirby (Kirby et al. 2008, 2009)
* Spectral resolution determination software by K. McKinnon (UCSC)

## files ##

* `spec_abund_ie.py` (and variations) contain capabilties for measuring Teff, [Fe/H], [alpha/Fe], and the Gaussian spectral resolution (Dlam), from spectral synthesis.
* `spec_abund_ie_dlamfix.py` is to be used when the spectral resolution is known (either a constant or an array as a function of wavelength). 
* `spec_abund_ie_resscale.py` provides an estimate of the resolution scale (a multiplicative factor with Dlam that accounts for the fact that the spectral resolution is higher than that estimated from the sky/arc lines, which fill the slit). The stellar parameters (Teff, [Fe/H], [alpha/Fe]) output by this code should not be used.
* `call_spec_abund_mpi.py` is a program that can be directly modified by the user according to their specifications in order to run `specabundie` on a given slitmask. MPI is used to run `specabundie` on multiple individual stars on the mask at once according to the number of processors indicated in the call sequence.
* `data_redux.py` contains the helper functions necessary to perform the measurement, including the continuum normalization routines.
* `read_synth.py` includes functions to assist in the reading in and interpolation of the grid of synthetic spectra produced by I.Escala and E.Kirby. 
* `smooth_gauss.f` is a Fortran implementation of the smoothing and interpolation of the synthetic spectrum onto the wavelength range and resolution of a given observed spectrum.
* `pywrap.sh` is a bash script to wrap `smooth_gauss.f` in Python. It is recommended that you re-wrap `smooth_gauss` on each new OS.
* `from_phot.py` is a Python impelemtation and modification of `feh.pro` by E. Kirby to measure stellar parameters (Teff, Logg, etc) from grids of [PARSEC](http://stev.oapd.inaf.it/cgi-bin/cmd) isochrones.
* `pickle_to_fits.py` is a post-processing code to combine all files output by `specabundie` into a single FITS file for a given slitmask.
* `resolution/` is a directory that contains optional programs to measure the spectral resolution as a function of wavelength based on sky and/or arc lines.
* `resolution/call_resolution.py` is a program to interface with the main code to measure the spectral resolution and enforce the correct file structure.
* `resolution/resolution.py` performs the spectral resolution measurement.
* `specres_moogify.py` is a post-processing code to take the output of the `resolution` functions and convert it to a format usable in `specabundie`.

## requirements ##

* Python 3
* [astropy](www.astropy.org) for FITS file handling and tables
* scipy
* numpy
* [mpi4py](https://pypi.org/project/mpi4py/)
* [emcee](https://emcee.readthedocs.io/en/stable/) (only for `fit_type='mcmc'` in `call_resolution`, which I don't recommend)

* Access to `gridie` and `grid7` synthetic spectral grids by I. Escala and E. Kirby

## set up ##

Clone this repository and maintain the indicated file structure. `specabundie` should be executed from the root directory containing the Python codes.

###### file structure

* A directory that contains the spec1d files for your slitmasks. The spec1d files should be organized within folders specified by the slitmask name, e.g. `slitmask_path/slitmask_name/`.
* A directory that contains the `moogify` files for your slitmasks. These are FITS files that contain the data required to perform the chemical abundance measurements for the slitmask (more information below). Each slitmask should have a corresponding `moogify.fits.gz` file. E.g., `moogify_path/slitmask_name/moogify.fits.gz`. 
* Directories that contain the grids of synthetic spectra, `synth_path_blue = synth_path/gridie/` and `synth_path_red = synth_path/grid7/`.
* A directory to save the output dictionaries containing the measured stellar parameters, `save_path`. Sub-directories will be automatically generated when savings the output according to `slitmask_name`.
* All other directory paths are the following, `telluric_path = ./telluric/`, `mask_path = ./mask/`.

######  special instructions for the spectral resolution determination

OPTIONAL: There are additional steps involved to properly set up the file structure for running the spectral resolution determination software.

Suppose you have a slitmask, `slitmask_name`, that was observed across multiple nights `night1` and `night2`. Create a directory, `resolution_data_path`. Within this directory, create **two** directories for the slitmask: `slitmask_name` and `slitmask_name` + `_stack`. Put the final stacked spec1d files in `slitmask_name_stack`.  

Then, create sub-directories in `slitmask_name`: the names of the slitmasks observed on each night (e.g., `slitmask_name_a` and `slitmask_name_b` if there are two masks covering the same targets designed at different parallactic angles, or just `slitmask_name` if the same mask name is shared between nights.) Within these sub-directories, create new folders for each night, e.g. `slitmask_name_a/night1` and `slitmask_name_b/night2`. Then, dump all the required **unstacked** files output by the spec1d software for the slitmask *from that night* in there: the `calibSlit` files (both red and blue), the `slit` files (both red and blue), and the spec1d files. The resolution determination program sorts everything as it runs.

NOTE that you only need this complicated file structure if you want to use the arc lines to determine the spectral resolution (recommended for the 600ZD grating). If you only want to use skylines, then you only need to create the `resoulution_data_path/slitmask_name` directory and use the appropriate keywords when running the code.

######  wrapping smooth_gauss

Wrap the Fortran code `smooth_gauss.f` in Python:
```
bash pywrap.sh
```

###### acessing your desired set of isochrones

You may want to use a different filter set or stellar age than those currently provided in the `isochrones` folder. If so, go to the [PARSEC](http://stev.oapd.inaf.it/cgi-bin/cmd) website and download your desired set of isochrones *for a fixed stellar age* but variable metallicity. I recommend spanning the full metallicity range of the isochrones in steps no smaller than 0.001. Then, rename your downloaded files according to the following format: `feh_filter_age_parsec.dat` where `filter` is the filter set and `age` is the assumed stellar age in Gyr (integer). Dump the renamed isochrone files in the `isochrones/rgb/` folder.

Then, go into the `from_phot` code and add a few lines to make sure it can read in the new filter set. Add `filter` to the list of acceptable filters on `line 97`. Then, under the `if` statement starting on `line 170`, add a block to read the isochrone set. Specifiy the columns that correspond the following parameters: MH, age, logL, logT, logg, label, mag_red, and mag_blue, where mag_red and mag_blue are the specific passbands to be used (e.g. V and I for Johnson-Cousins photometry).

###### create your moogify file

The moogify file does **not** need to originate from E. Kirby's IDL software. One day people will wonder why the input FITS files have these names, just as we wonder why we still use the magnitude system. I recommend creating this file for the stars in your slitmask using Astropy tables and then gzipping it.
```
from astropy.table import Table

t = Table()
t['objname'] = object names 
t['slit'] = slit number
t['zrest'] = redshift measured from a software like zspec (not provided)
t['good'] = boolean array indicating whether the star is good (1) or bad (0)
t['ra'] = right acension
t['dec'] = declination
t['mag_red'] = red magnitudes (from your photometry catalog)
t['mag_blue'] = blue magnitudes (from your photometry catalog)
t['mag_red_err'] = red magnitude errors (from your photometry catalog)
t['mag_blue_err'] = blue magnitude errors (from your photometry catalog)
t['teffphot'] = photometric effective temperature (from from_phot, for your adopted age)
t['loggphot'] = photometric surface gravity (from from_phot, for your adopted age)
t['teffphoterr'] = uncertainty on the photometric effective temperature (from from_phot, for your adopted age)
t['loggphoterr'] = uncertainty on the photometric surface gravity (from from_phot, for your adopted age)

t.write('moogify.fits', format='fits', overwrite=True)
```

Note that if you plan on using `call_resolution` to measure the spectral resolution, running `specres_moogify` will automatically add it (under the keyword `dlam`) to your moogify file.

## running specabundie ##

**Step 1**: Measure photometric stellar parameters using `from_phot` and save the output (for adding to your moogify file above).
```
from from_phot import from_phot
import numpy as np

from_phot(t['mag_red'], t['mag_blue']-t['mag_red'], err_mag_in=t['mag_red_error'], err_color_in=np.sqrt(t['mag_red_error']**2. + t['mag_blue_err']**2.),
          age=12, filter='vi', dm=24.45, ddm=0.05)
```

**OPTIONAL Step 2**: Measure spectral resolution using `call_resolution`. This is an example for whether observations were taken across multiple nights for a given slitmask and arc lines are being used (i.e. for the 600ZD grating).
```
from call_resolution import call_resolution

call_resolution(slitmask_name, stacked=True, fit_sky_only=False, spectra_path=resolution_data_path, 
                out_path=resolution_output_path)
```

Then, convert the output to `dlam` and add it to your `moogify.fits.gz` file (that should already be located in its proper file structure) by running `specres_moogify`
```
from specres_moogify import specres_moogify

specres_moogify(slitmask_name, stacked=True, slitmask_path=slitmask_path, moogify_path=moogify_path, 
                data_path=resolution_output_path, dlam_def=1.2)
```

**OPTIONAL Step 3**: Measure the resolution scale parameter by running `spec_abund_ie_resscale` using `call_spec_abund_mpi`. This is to account for the fact that the spectral resolution for the star will be higher than that measured from the sky/arc lines, which fill the slit.

Go into `call_spec_abund_mpi` and modify the beginning of the file, specifying path names, and setting `dlamfix=False` and `resscale = True`. Then specify
`slitmask_name` and `grating` (1200G or 600ZD). Execute the code given your number of processors (here, I picked 8):
```
mpiexec -n 8 call_spec_abund_mpi.py
```

When `specabund` is finished running, execute `pickle_to_fits` to collect the output into a single FITS file.
```
from pickle_to_fits import pickle_to_fits

pickle_to_fits(slitmask_name, save_path=save_path, dlamfix=False, resscale=True)
```

You can then open this FITS file (`specabundie_slitmask_name_resscale.fits`) and compute the mean resolution scale for the mask:
```
resscale = np.mean(specabund['resscale'])
```

**Step 4** Measure stellar parameters using `call_spec_abund_mpi`. 

Go into `call_spec_abund_mpi` and modify the beginning of the file, specifying path names. Then specify `slitmask_name` and `grating` (1200G or 600ZD). Even if you did not optionally measure the resolution scale, I recommend setting `dlamfix = True` with a reasonable value according to your grating configuration. If you set `dlamfix = False`, the code will fit for `dlam`, but this is not really necessary (see the Appendix of Escala et al. 2019) or preferable. If you measured `resscale`, then add an entry to `resscale_dict` for your `slitmask_name`. 

Finally, execute the code given your number of processors (here, I picked 8):
```
mpiexec -n 8 call_spec_abund_mpi.py
```

And construct your final FITS file when `specabund` is finished running.
```
from pickle_to_fits import pickle_to_fits

pickle_to_fits(slitmask_name, save_path=save_path, dlamfix=False, resscale=True)
```

## licensing ##

* Copyright 2017-2018 by Ivanna Escala.
* In summary, you are free to use and edit this software with my permission (this software is currently private). But please keep me informed. Have fun!
