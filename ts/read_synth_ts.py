"""
@author: Ivanna Escala 2018-2022
"""

from os.path import isfile, join
from functools import reduce
import scipy.ndimage
import numpy as np
import itertools
import gzip
import glob
import sys
import os


def read_synth(Teff = np.nan, Logg = np.nan, Feh = np.nan, Alphafe = np.nan, data_path=''):

    """
    Read the ".bin.gz" file containing the synthetic spectrum information
    that it output by running MOOGIE

    Parameters:
    ------------
    Teff: float: effective temperature (K) of synthetic spectrum
    Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
    Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
    Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
    data_path: string: the path leading to the parent directory containing the synthetic
                       spectrum data

    Returns:
    ---------
    wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                whether it is full resolution or binned, and on stop and start wavelengths
    flux: array: the normalized flux of the synthetic spectrum

    """

    data_path = f"{data_path}synths"

    if Logg <= 3.0:
        model_vt = 2.
        geo = 'sph'
    else:
        model_vt = 1.
        geo = 'pp'

    #Redefine the parameters according to the file naming convention
    modelfn = construct_model_filename(Teff, Logg, Feh, Alphafe,
    path=data_path, grid_type='apogee', micro_turb_vel=model_vt, geometry=geo)

    #Open and read the contents of the gzipped binary file without directly
    #unzipping, for enhanced performance

    #wave, norm = [], []

    with gzip.open(modelfn, 'rb') as f:
        bstring = f.read()
        result = np.fromstring(bstring, dtype=float, sep='\n')
        wave, flux = result[0::3], result[1::3]

        #lines = f.readlines()
        #for line in lines:
        #    wavei, normi, _ = line.split()
        #    wave.append(float(wavei))
        #    norm.append(float(normi))

    #wave = np.array(wave)
    #norm = np.array(norm)

    return wave, flux

def construct_model_filename(teffi, loggi, fehi, alphai, geometry='sph',
model_mass=1.0, micro_turb_vel=2., path='.', grid_type='default'):

        #The user is allowed to define the geometry to some extent,
        #but there are limits as described above
        if loggi < 3.0: geometry = 'sph' #override user input if necessary
        if loggi > 3.5: geometry = 'pp' #only plane parallel models

        if geometry == 'sph':
            geostr = 's'
        if geometry == 'pp':
            geostr = 'p'
            model_mass = 0.0 #overwrite default parameters

        #if its the default MARCS grid, then check the zclass
        if grid_type == 'default':
            try:
                zclass = check_zclass(fehi, alphai)
            except:
                if alphai >= 0.4: zclass = 'ae'
                elif alphai <= -0.4: zclass = 'an'
                else: zclass = 'ap'

        #if using the MARCS-APOGEE grid, then just use 'x3'
        if grid_type == 'apogee':
            zclass = 'x3'

        if np.sign(loggi) == 1 or np.sign(loggi) == 0: gsign = '+'
        else: gsign = '-'

        if np.sign(fehi) == 1 or np.sign(fehi) == 0: fsign = '+'
        else: fsign = '-'

        if np.sign(alphai) == 1 or np.sign(alphai) == 0: asign = '+'
        else: asign = '-'

        feh_str = str(np.abs(fehi))
        if len(feh_str) != 4: feh_str += '0'

        alpha_str = str(np.abs(alphai))
        if len(alpha_str) != 4: alpha_str += '0'

        vt_str = f'0{int(micro_turb_vel)}'

        gstr = str(int(loggi*10.))
        if len(gstr) != 2: gstr = '0'+gstr

        modelfn = (f"{path}/t{int(teffi)}/g_{gstr}/{geostr}{int(teffi)}_g{gsign}{np.abs(loggi)}"
                   f"_m{model_mass}_t{vt_str}_{zclass}_z{fsign}{feh_str}"
                   f"_a{asign}{alpha_str}_c+0.00_n+0.00_o{asign}{alpha_str}"
                   f"_r+0.00_s+0.00.bin.gz")

        return modelfn

def read_interp_synth(teff=np.nan, logg=np.nan, feh=np.nan, alphafe=np.nan,
data_path='', npar=4, hash=None):

    """
    Construct a synthetic spectrum in between grid points based on linear interpolation
    of synthetic spectra in the MARCS+TS grid

    Parameters:
    -----------
    Teff: float: effective temperature (K) of synthetic spectrum
    Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
    Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
    Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
    data_path: string: the path leading to the parent directory containing the synthetic
                       spectrum data
    npar: integer: number of parameters used to describe a synthetic spectrum
    hash: dict, optional: a dictionary to use to store memory concerning which synthetic
          spectra have been read in. Should be initliazed externally as an empty dict.

    Returns:
    --------
    wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                whether it is full resolution or binned, and on stop and start wavelengths
    flux: array: the normalized flux of the synthetic spectrum
    """

    #Define the points of the 4D grid
    teff_arr = np.concatenate((np.arange(3000., 4000., 100.), np.arange(4000., 6750., 250.))).astype('int')
    logg_arr = np.round(np.arange(0., 5.5, 0.5), decimals=1)

    #interp grid
    feh_arr = np.round(np.arange(-2.5, 1.1, 0.1), decimals=2)
    alphafe_arr = np.round(np.arange(-1.0, 1.1, 0.1), decimals=2)

    #No interp grid
    #feh_arr = np.round(np.arange(-2.5, 1.25, 0.25), decimals=2)
    #alphafe_arr = np.round(np.arange(-1.0, 1.25, 0.25), decimals=2)

    params = np.array([teff, logg, feh, alphafe])
    params_grid = np.array([teff_arr, logg_arr, feh_arr, alphafe_arr])

    #in_grid,_ = enforce_grid_check(teff, logg, feh, alphafe)
    #if not in_grid: return

    #Now identify the nearest grid points to the specified parameter values
    ds = []; nspecs = []; iparams = []
    for i in range(npar):

        #The specified parameter value is a grid point
        w = np.digitize(params[i], params_grid[i])
        if params[i] in params_grid[i]:
            iparam = np.array([w-1, w-1])
            d = [1.]
            nspec = 1
            ds.append(d)

        #The specified parameter value is in between grid points
        else:
            if w == (len(params_grid[i])): w -= 1
            iparam = np.array([w-1, w])
            d = params_grid[i][iparam] - params[i]
            d_rev = np.abs(d[::-1])
            nspec = 2
            ds.append(d_rev)

        nspecs.append(nspec)
        iparams.append(iparam)

    #Now, based on the nearest grid points, construct the linearly interpolated
    #synthetic spectrum

    #Calculate the number of pixels in the synthetic spectrum, and initialize the
    #interpolated synthetic spectrum array
    start = 4100; sstop = 9100; step = 0.14
    npixels = len(np.arange(start, sstop, step))
    synth_interp = np.zeros(npixels)

    #Function for loading a specified synthetic spectrum
    def load_synth(p):

        teffi, loggi, fehi, alphai = p

        if logg_arr[loggi] <= 3.0:
            model_vt = 2.
            geo = 'sph'
        else:
            model_vt = 1.
            geo = 'pp'

        if hash is not None:

            #First construct the filename corresponding to the parameters, to use for
            #testing whether we should read in the specified synthetic spectrum

            modelfn = construct_model_filename(teff_arr[teffi], logg_arr[loggi],
            feh_arr[fehi], alphafe_arr[alphai],path=data_path, micro_turb_vel=model_vt,
            grid_type='apogee', geometry=geo)

            key = modelfn.split('/')[-1][:-7]

            if key not in hash.keys():

                _,synthi = read_synth(Teff=teff_arr[teffi], Logg=logg_arr[loggi],
                Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphai], data_path=data_path)

                hash[key] = synthi

            #If the key is already present in the hash table, then find it and load the data
            else: synthi = hash[key]

        else:
            _,synthi= read_synth(Teff=teff_arr[teffi], Logg=logg_arr[loggi],
            Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei], data_path=data_path)

        return synthi

    #Load each nearby synthetic spectrum on the grid to linearly interpolate
    #to calculate the interpolated synthetic spectrum
    for i in range(nspecs[0]):
        for j in range(nspecs[1]):
            for k in range(nspecs[2]):
                for l in range(nspecs[3]):

                    p = [iparams[0][i], iparams[1][j], iparams[2][k], iparams[3][l]]
                    synthi = load_synth(p)

                    for m in range(npixels):
                        synth_interp[m] += ds[0][i]*ds[1][j]*ds[2][k]*ds[3][l]*synthi[m]

    facts = []
    for i in range(npar):
        if nspecs[i] > 1: fact = params_grid[i][iparams[i][1]] - params_grid[i][iparams[i][0]]
        else: fact = 1
        facts.append(fact)

    synth_interp /= reduce(lambda x, y: x*y, facts)
    wave = np.arange(start, sstop, step)

    return wave, synth_interp
