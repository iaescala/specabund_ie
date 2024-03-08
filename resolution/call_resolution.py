"""
@author:iescala

Ivanna Escala 2019-2020
"""

import resolution
from shutil import copy2
import glob2 as glob
import re
import os
import sys

def call_resolution(slitmask0, stacked=False, fit_sky_only=False, site='keck',
fit_type='ml', spectra_path='', out_path=''):

    """
    Program to call spectral resolution fitting code by K. McKinnon (UCSC)
    with some structual modifications by I. Escala

    Parameters
    ----------
    slitmask0: str: name of the slitmask
    stacked: optional, boolean, default False. Whether the observations
    were taken across multiple nights.
    fit_sky_only: optional, boolean, default False. Whether to fit only
    the sky lines (and not the arc lines).
    site: optional, string, default 'keck'. Observing site.
    fit_type: optional, string, default 'ml' or maximum likelihood. Fitting
    algorithm to use. Other option is 'mcmc', but takes much longer.
    spectra_path: optional, string. Path to the location of the observed
    spectral files.
    out_path: optional, string. Output path of the spectral resolution
    measurement.

    Returns
    -------
    Files containing the measured spectral resolution for each
    individual star observed on a slitmask. 
    """

if stacked: slitmask = slitmask0+'_stack'
else: slitmask = slitmask0

filenames = glob.glob(spectra_path+slitmask+'/**/spec1d.*.fits.gz')

object_ids = [re.search('\.\d\d\d\.', file).group()[1:-1] for file in filenames]

slitmask = slitmask0

for i in range(len(object_ids)):

    if stacked:

        dirs = os.listdir(spectra_path+slitmask)

        for dir in dirs:
            subdirs = os.listdir(spectra_path+slitmask+'/'+dir)

            for subdir in subdirs:

                if os.path.exists(spectra_path+slitmask+'/'+dir+'/'+subdir+'/'+object_ids[i]+'/'):
                    pass
                else:
                    os.mkdir(spectra_path+slitmask+'/'+dir+'/'+subdir+'/'+object_ids[i]+'/')

                other_files = glob.glob(spectra_path+slitmask+'/'+dir+'/'+subdir+'/*.'+dir+'.'+object_ids[i]+'*.fits.gz')

                if len(other_files) > 0:
                    for file in other_files:
                        copy2(file, spectra_path+slitmask+'/'+dir+'/'+subdir+'/'+object_ids[i])
                        os.remove(file)
    else:

        if os.path.exists(spectra_path+slitmask+'/'+object_ids[i]+'/'):
            pass
        else:
            os.mkdir(spectra_path+slitmask+'/'+object_ids[i]+'/')
            copy2(filenames[i], spectra_path+slitmask+'/'+object_ids[i])

        other_files = glob.glob(spectra_path+slitmask+'/*.'+slitmask+'.'+object_ids[i]+'*.fits.gz')
        if len(other_files) > 0:
            for file in other_files:
                copy2(file, spectra_path+slitmask+'/'+object_ids[i])
                os.remove(file)

for object_id in object_ids:

    print(object_id)

    try: resolution.singleStarResolution(object_id, slitmask, spectra_path=spectra_path,
        run_fit_only=False, fittype=fit_type, out=out_path, stacked=stacked,
        fit_sky_only=fit_sky_only, site=site)
    except:
        print('FAILURE')
        continue
