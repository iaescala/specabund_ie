"""
@author: Ivanna Escala 2018-2022
"""
from astropy.coordinates import SkyCoord, EarthLocation
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp1d
from smooth_gauss import smooth_gauss
from numpy.polynomial import Chebyshev
from astropy import constants as const
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits
import scipy.stats as stats
from astropy import units
import numpy as np
import read_synth_ts as read_synth
import pickle
import scipy
import copy
import glob
import sys
import os
import re

def check_pkl_file_exists(filename, file_path=''):
    """
    A helper function to check if a pickle file exists, and if so, to open
    it and return the data. Else, returns False.
    """
    if not os.path.exists(file_path+filename+'.pkl'):
        return None
    else:
        with open(file_path+filename+'.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

def construct_mask(wvlarr, fluxarr, weights, mask_ranges=None, ccd_mask=None,
fit_ranges=[[4100.,9100.]], zrest=0.):

    """
    Construct a wavelength mask (in Angstroms) for the given wavelength array, based on
    the desired wavelength regions to fit, the chip gap between the blue and red sides
    of the CCD, and other specified mask ranges

    Parameters
    ----------
    wvlarr: array-like: wavelength array to use for constructing the mask
    fluxarr: array-like: flux values to aid in the construction of the mask, ignoring
                         regions with nonsenical flux values
    weights: array-like: associated weights, again to ignore bad spectral regions
    mask_ranges: 2D array-like, optional: regions of the observed spectrum to mask
                 in comparison, in units of Angstroms
    fit_ranges: 2D array-like: wavelength range in Angstroms over which to perform the
                comparison
    ccd_mask: 2D array-like: wavelength range in Angstroms to mask, which corresponds to
              the gap between the blue and red sides of the CCD

    Returns
    -------
    mask: array-like: boolean mask of wavelength ranges in Angstroms, where masked regions
          are False
    """

    mask = np.full(len(wvlarr), False, dtype=bool)

    #Enforce the specified fit ranges
    for ranges in fit_ranges:
        mask[np.where(((wvlarr >= ranges[0]) & (wvlarr <= ranges[1])))] = True

    #Mask areas around the CCD
    nhalf = int(round(float(len(wvlarr))/2.))
    mask[0:4] = False; mask[-5:-1] = False
    mask[nhalf-5:nhalf-1] = False; mask[nhalf:nhalf+4] = False

    #Mask the specified wavelength ranges
    if mask_ranges is not None:
        for ranges in mask_ranges:
            mask[np.where(((wvlarr >= ranges[0]) & (wvlarr <= ranges[1])))] = False

    #Make sure that the standard masked regions are excluded
    standard_mask = np.array([[4856., 4866.],[5885.,5905.],[6270.,6300.],[6341., 6346.],[6356.,6365.],
    [6559.797, 6565.797], [7662., 7668.],[8113.,8123.], [8317.,8330.],[8488.023, 8508.023],
    [8525.091, 8561.091],[8645.141, 8679.141], [8804.756, 8809.756]])
    tell_mask = np.array([[6864.,7020.], [7162.,7320.],[7591.,7703.],[8128.,8352.],
    [8938.,9900.]])/(1. + zrest)
    standard_mask = np.append(standard_mask, tell_mask, axis=0)

    for ranges in standard_mask:
        mask[np.where(((wvlarr >= ranges[0]) & (wvlarr <= ranges[1])))] = False

    #Mask regions with nonsensical flux values
    mask[np.where((fluxarr < 0.)|(weights <= 0.)|np.isnan(weights)|np.isnan(fluxarr))] = False

    return mask

def compare_spectra(spec, model, stat_name='chisqr', weights=None, nvarys=4):

    """
    Compare two spectra (an observed spectrum and model spectrum)
    using a chi-squared statistic. Note: the observed and model synthetic
    spectrum must be interpolated onto the same wavelength array prior to
    calling this function.

    Parameters
    ----------
    spec: array-like: continuum normalized, radial velocity corrected
          flux for the observed spectrum
    model: array-like: relative flux for the model spectrum, interpolated onto
           the same wavelength array as the observed spectrum
    stat: string, optional: statistic to use to compare the spectra. Options include
          chi-squared (chisqr), reduced-chi squared (redchi), and residuals (residual).
    weights: array-like, optional: weights corresponding to the observed spectrum
    dof: integer, optional: number of degrees of freedom to be used, if calculated the
         reduced chi-squared statistc

    Returns
    -------
    stat: float: calculated test statistic
    """

    if not isinstance(spec, np.ndarray): spec = np.array(spec)
    if not isinstance(model, np.ndarray): model = np.array(model)

    if weights is not None:
        if not isinstance(weights, np.ndarray): weights = np.array(weights)

    stats = ['chisqr', 'redchi', 'residual']
    assert stat_name in stats

    mask = np.full(len(spec), True, dtype=bool)

    #Mask regions with nonsensical flux values
    mask[np.where(np.logical_or(spec < 0., model < 0., weights <= 0.))] = False

    if stat_name == 'chisqr':
        stat = np.nansum( (spec - model)**2. * weights)

    if stat_name == 'redchi':
        devs = (spec - model)**2. * weights
        stat = np.nansum(devs)
        stat /= (float(devs.size) - float(nvarys))

    if stat_name == 'residual':
        stat = (spec - model)*np.sqrt(weights)

    return stat

def check_fit_coadd(best_params, params_err, wvl, flux_obs, ivar, mask=None, nvarys=2,
hash=None, synth_path='', phot_dict=None, dlam_dict=None, dlamfix=True,
slit_list=None,teffphot_list=None,loggphot_list=None,ivar_list=None,grating=None):

    """
    Note: the information based on which values to perturb is contained in the
    params_err dict, which has the associated standard deviations. If the uncertainty
    is zero, than the parameter is not varied.
    """
    #Identify the number of varied parameters
    #w = np.where(np.array(params_err.values()) > 0.)[0]

    #Perturb the best fit values around the minimum
    ncontour = 41
    flex_factor_logg = 40.
    perturb = np.arange(0, ncontour, 1)*0.25 - 5.
    par_limits = {'teff': [3000., 6500.], 'logg': [0., 5.], 'feh': [-2.5, 1.],\
    'alphafe': [-1., 1.], 'dlam': [1., 1.4]}

    if grating == '600ZD':
        dlam_def = 1.2; dlamerr = 0.05; dlam_bounds = [1., 1.4]
        dlam_dict = {'dlam': dlam_def, 'dlamerr': dlamerr}

    if grating == '1200G':
        dlam_def = 0.45; dlamerr = 0.05
        dlam_dict = {'dlam': dlam_def, 'dlamerr': dlamerr}

    param_names = ['feh', 'alphafe']

    vals_dict = {}
    for name in param_names:

        #If parameter is at the edges of the grid, or if the statistical uncertainty
        #is zero, then construct the array to perturb the points by accordingly

        #if params_err[name] <= 0.
        if best_params[name] >= par_limits[name][1] or best_params[name] <= par_limits[name][0]:
            vals = np.array(range(ncontour))*(par_limits[name][1] - par_limits[name][0])\
            /(float(ncontour) - 1.) + par_limits[name][0]

        #Else, use the statistical uncertainty to construct the perturbation array
        else:
            step = params_err[name]
            right = best_params[name] - par_limits[name][0]
            left = par_limits[name][1] - best_params[name]

            #If the statistical uncertainty is greater than the maximum possible step size
            #that prevents the contour from going outside of the bounds of the grid, then
            #change the step size such that this does not occur
            step = min([step, max([right, left])/5.])

            #Construct the array
            vals = perturb * step + best_params[name]
            #If the array is outside of the bounds of the grid, then set the out of bounds
            #values to the grid boundaries
            w0 = vals < par_limits[name][0]; w1 = vals > par_limits[name][1]
            vals[w0] = par_limits[name][0]; vals[w1] = par_limits[name][1]

        #Identify the xaxis contour information and the parameter that is being varied
        if not list(vals).count(vals[0]) == len(vals):
            xvals = vals
            pvar = name

        vals_dict[name] = vals

    def get_synth(teffphot_list, loggphot_list, feh, alphafe, dlam, fit=True):

        relflux_synth_list = []

        for i in range(len(slit_list)):

            wvl_synth, relflux_synth = read_synth.read_interp_synth(teff=teffphot_list[i], logg=loggphot_list[i],
            feh=feh, alphafe=alphafe, data_path=synth_path, hash=hash)

            wsynth = np.where((wvl_synth > wvl[0])&(wvl_synth < wvl[-1]))[0]

            wvl_synth = wvl_synth[wsynth]
            relflux_synth = relflux_synth[wsynth]

            #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
            relflux_interp_smooth = smooth_gauss_wrapper(wvl_synth, relflux_synth, wvl, dlam)
            relflux_synth_list.append(relflux_interp_smooth)

        if fit==True:
            ivar_masked = [item[mask] for item in ivar_list]
            synth_masked = [item[mask] for item in relflux_synth_list]
        else:
            ivar_masked = ivar_list

        relflux_interp_smooth_stack = np.nansum(np.array(synth_masked)*np.array(ivar_masked),axis=0)/np.sum(np.array(ivar_masked),axis=0)

        return relflux_interp_smooth_stack

    #Calculate the reduced chi-squared statistic for each combination of perturbed
    #parameters
    chis = np.zeros(len(perturb)); synth_alpha = []
    for i in range(len(perturb)):

        if nvarys == 2:

            #If the x-values for the chi-squared contours are the same, then use the previously
             #determined value using the same x-value to save computation time
            if (vals_dict['feh'][i] == vals_dict['feh'][i-1]) and (vals_dict['alphafe'][i] == vals_dict['alphafe'][i-1])\
            and (i > 0):
                chis[i] = chis[i-1]

            #Otherwise, load in a new synthetic spectrum, and calculate the appropriate
            #chi-squared value
            else:
                synth = get_synth(teffphot_list, loggphot_list, vals_dict['feh'][i],
                vals_dict['alphafe'][i], dlam_def)

                #For effective temperature, include photometric constraints in the chi-squared
                #determination
                if not dlamfix:
                    if pvar == 'teff' or pvar == 'dlam':

                        teffinit = min([max([phot_dict['teffphot'], par_limits['teff'][0]]),
                        par_limits['teff'][1]])
                        dlaminit = min([max([dlam_dict['dlam'], par_limits['dlam'][0]]),
                        par_limits['dlam'][1]])

                        fluxp = np.insert(flux_obs[mask], 0, teffinit)
                        fluxp = np.insert(fluxp, 1, dlaminit)

                        ivarp = np.insert(ivar[mask], 0, float(len(ivar[mask]))/phot_dict['flex_factor']*\
                        phot_dict['teffphoterr']**(-2.))
                        ivarp = np.insert(ivarp, 1, dlam_dict['dlamerr']**(-2.))

                        synthp = np.insert(synth[mask], 0, vals_dict['teff'][i])
                        synthp = np.insert(synthp, 1, vals_dict['dlam'][i])

                        chi = compare_spectra(fluxp, synthp, stat_name='chisqr',
                        weights=ivarp)
                        chis[i] = chi

                    else:

                        chi = compare_spectra(flux_obs[mask], synth[mask], stat_name='chisqr',
                        weights=ivar[mask])
                        chis[i] = chi

                else:

                    if pvar == 'teff':

                        teffinit = min([max([phot_dict['teffphot'], par_limits['teff'][0]]),
                        par_limits['teff'][1]])

                        fluxp = np.insert(flux_obs[mask], 0, teffinit)
                        ivarp = np.insert(ivar[mask], 0, float(len(ivar[mask]))/phot_dict['flex_factor']*\
                        phot_dict['teffphoterr']**(-2.))
                        synthp = np.insert(synth[mask], 0, vals_dict['teff'][i])

                        chi = compare_spectra(fluxp, synthp, stat_name='chisqr',
                        weights=ivarp)
                        chis[i] = chi

                    else:

                        chi = compare_spectra(flux_obs[mask], synth, stat_name='chisqr',
                        weights=ivar[mask])
                        chis[i] = chi

        else:

            if (vals_dict['feh'][i] == vals_dict['feh'][i-1]) and (vals_dict['alphafe'][i] == vals_dict['alphafe'][i-1])\
            and (i > 0):
                chis[i] = chis[i-1]

            else:
                synth = get_synth(teffphot_list, loggphot_list, vals_dict['feh'][i],
                vals_dict['alphafe'][i], dlam_def)

                chi = compare_spectra(flux_obs[mask], synth[mask], stat_name='chisqr',
                weights=ivar[mask])
                chis[i] = chi

    return xvals, chis

def check_fit(best_params, params_err, wvl, flux_obs, ivar, mask=None,
hash=None, synth_path='', phot_dict=None,
dlam_dict=None, dlamfix=False):

    """
    Note: the information based on which values to perturb is contained in the
    params_err dict, which has the associated standard deviations. If the uncertainty
    is zero, than the parameter is not varied.
    """
    #Identify the number of varied parameters
    #w = np.where(np.array(params_err.values()) > 0.)[0]

    #Perturb the best fit values around the minimum
    ncontour = 41
    flex_factor_logg = 40.
    perturb = np.arange(0, ncontour, 1)*0.25 - 5.
    par_limits = {'teff': [3000., 6500.], 'logg': [0., 5.], 'feh': [-2.5, 1.],\
    'alphafe': [-1., 1.], 'dlam': [1., 1.4]}

    param_names = ['teff', 'logg', 'feh', 'alphafe']
    #if nvarys == 4: param_names.append('dlam')
    if not dlamfix: param_names.append('dlam')

    vals_dict = {}
    for name in param_names:

        #If parameter is at the edges of the grid, or if the statistical uncertainty
        #is zero, then construct the array to perturb the points by accordingly

        #if params_err[name] <= 0.
        if best_params[name] >= par_limits[name][1] or best_params[name] <= par_limits[name][0]:
            vals = np.array(range(ncontour))*(par_limits[name][1] - par_limits[name][0])\
            /(float(ncontour) - 1.) + par_limits[name][0]

        #Else, use the statistical uncertainty to construct the perturbation array
        else:
            step = params_err[name]
            right = best_params[name] - par_limits[name][0]
            left = par_limits[name][1] - best_params[name]

            #If the statistical uncertainty is greater than the maximum possible step size
            #that prevents the contour from going outside of the bounds of the grid, then
            #change the step size such that this does not occur
            step = min([step, max([right, left])/5.])

            #Construct the array
            vals = perturb * step + best_params[name]
            #If the array is outside of the bounds of the grid, then set the out of bounds
            #values to the grid boundaries
            w0 = vals < par_limits[name][0]; w1 = vals > par_limits[name][1]
            vals[w0] = par_limits[name][0]; vals[w1] = par_limits[name][1]

        #Identify the xaxis contour information and the parameter that is being varied
        if not list(vals).count(vals[0]) == len(vals):
            xvals = vals
            pvar = name

        vals_dict[name] = vals

    #Identify whether there is wavelength information in the blue and/or red
    #synthetic spectra ranges
    #wb = np.where((wvl >= 4100.)&(wvl < 6300.))[0]
    #wr = np.where((wvl >= 6300.)&(wvl < 9100.))[0]

    def get_synth(teff, logg, feh, alphafe, dlam):

        wvl_synth, relflux_synth = read_synth.read_interp_synth(teff=teff, logg=logg,
        feh=feh, alphafe=alphafe, data_path=synth_path, hash=hash)

        wsynth = np.where((wvl_synth > wvl[0])&(wvl_synth < wvl[-1]))[0]

        wvl_synth = wvl_synth[wsynth]
        relflux_synth = relflux_synth[wsynth]

        #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
        relflux_interp_smooth = smooth_gauss_wrapper(wvl_synth, relflux_synth, wvl, dlam)

        return relflux_interp_smooth

    #Calculate the reduced chi-squared statistic for each combination of perturbed
    #parameters
    chis = np.zeros(len(perturb)); synth_alpha = []
    for i in range(len(perturb)):

        #if nvarys == 4:
        if not dlamfix:

            #If the x-values for the chi-squared contours are the same, then use the previously
             #determined value using the same x-value to save computation time
            if (vals_dict['teff'][i] == vals_dict['teff'][i-1]) and (vals_dict['feh'][i] ==\
            vals_dict['feh'][i-1]) and (vals_dict['alphafe'][i] == vals_dict['alphafe'][i-1])\
            and (vals_dict['dlam'][i] == vals_dict['dlam'][i-1]) and (i > 0):
                chis[i] = chis[i-1]

            #Otherwise, load in a new synthetic spectrum, and calculate the appropriate
            #chi-squared value
            else:
                synth = get_synth(vals_dict['teff'][i], vals_dict['logg'][i], vals_dict['feh'][i],
                vals_dict['alphafe'][i], vals_dict['dlam'][i])

                if synth is None:
                    chis[i] = np.nan
                    continue

                #For effective temperature, include photometric constraints in the chi-squared
                #determination

                #if not dlamfix:
                if pvar == 'teff' or pvar == 'dlam':

                    teffinit = min([max([phot_dict['teffphot'], par_limits['teff'][0]]),
                    par_limits['teff'][1]])
                    dlaminit = min([max([dlam_dict['dlam'], par_limits['dlam'][0]]),
                    par_limits['dlam'][1]])

                    fluxp = np.insert(flux_obs[mask], 0, teffinit)
                    fluxp = np.insert(fluxp, 1, dlaminit)

                    ivarp = np.insert(ivar[mask], 0, float(len(ivar[mask]))/phot_dict['flex_factor']*\
                    phot_dict['teffphoterr']**(-2.))
                    ivarp = np.insert(ivarp, 1, dlam_dict['dlamerr']**(-2.))

                    synthp = np.insert(synth[mask], 0, vals_dict['teff'][i])
                    synthp = np.insert(synthp, 1, vals_dict['dlam'][i])

                    chi = compare_spectra(fluxp, synthp, stat_name='chisqr',
                    weights=ivarp)
                    chis[i] = chi

                else:

                    chi = compare_spectra(flux_obs[mask], synth[mask], stat_name='chisqr',
                    weights=ivar[mask])
                    chis[i] = chi

        #Otherwise, if the delta lambda parameter is fixed
        else:

            #If the x-values for the chi-squared contours are the same, then use the previously
            #determined value using the same x-value to save computation time
            if (vals_dict['teff'][i] == vals_dict['teff'][i-1]) and (vals_dict['feh'][i] ==\
            vals_dict['feh'][i-1]) and (vals_dict['alphafe'][i] == vals_dict['alphafe'][i-1])\
            and (i > 0):
                chis[i] = chis[i-1]

            #Otherwise, load in a new synthetic spectrum, and calculate the appropriate
            #chi-squared value
            else:
                synth = get_synth(vals_dict['teff'][i], vals_dict['logg'][i], vals_dict['feh'][i],
                vals_dict['alphafe'][i], best_params['dlam'])

                if synth is None:
                    chis[i] = np.nan
                    continue

                if pvar == 'teff':

                    teffinit = min([max([phot_dict['teffphot'], par_limits['teff'][0]]),
                    par_limits['teff'][1]])

                    fluxp = np.insert(flux_obs[mask], 0, teffinit)
                    ivarp = np.insert(ivar[mask], 0, float(len(ivar[mask]))/phot_dict['flex_factor']*\
                    phot_dict['teffphoterr']**(-2.))
                    synthp = np.insert(synth[mask], 0, vals_dict['teff'][i])

                    chi = compare_spectra(fluxp, synthp, stat_name='chisqr',
                    weights=ivarp)
                    chis[i] = chi

                else:

                    chi = compare_spectra(flux_obs[mask], synth[mask], stat_name='chisqr',
                    weights=ivar[mask])
                    chis[i] = chi

    return xvals, chis

def smooth_gauss_wrapper(lambda1, spec1, lambda2, dlam_in):
    """
    A wrapper around the Fortran routine smooth_gauss.f, which
    interpolates the synthetic spectrum onto the wavelength array of the
    observed spectrum, while smoothing it to the specified resolution of the
    observed spectrum.
    Adapted into Python from IDL (E. Kirby)

    Parameters
    ----------
    lambda1: array-like: synthetic spectrum wavelength array
    spec1: array-like: synthetic spectrum normalized flux values
    lambda2: array-like: observed wavelength array
    dlam_in: float, or array-like: full-width half max resolution in Angstroms
             to smooth the synthetic spectrum to, or the FWHM as a function of wavelength

    Returns
    -------
    spec2: array-like: smoothed and interpolated synthetic spectrum, matching observations
    """

    if not isinstance(lambda1, np.ndarray): lambda1 = np.array(lambda1)
    if not isinstance(lambda2, np.ndarray): lambda2 = np.array(lambda2)
    if not isinstance(spec1, np.ndarray): spec1 = np.array(spec1)

    #Make sure the synthetic spectrum is within the range specified by the
    #observed wavelength array
    n2 = lambda2.size; n1 = lambda1.size

    def findex(u, v):
        """
        Return the index, for each point in the synthetic wavelength array, that corresponds
        to the bin it belongs to in the observed spectrum
        e.g., lambda1[i-1] <= lambda2 < lambda1[i] if lambda1 is monotonically increasing
        The minus one changes it such that lambda[i] <= lambda2 < lambda[i+1] for i = 0,n2-2
        in accordance with IDL
        """
        result = np.digitize(u, v)-1
        w = [int((v[i] - u[result[i]])/(u[result[i]+1] - u[result[i]]) + result[i]) for i in range(n2)]
        return np.array(w)

    f = findex(lambda1, lambda2)

    #Make it such that smooth_gauss.f takes an array corresponding to the resolution
    #each point of the synthetic spectrum will be smoothed to
    if isinstance(dlam_in, list) or isinstance(dlam_in, np.ndarray): dlam = dlam_in
    else: dlam = np.full(n2, dlam_in)
    dlam = np.array(dlam)

    dlambda1 = np.diff(lambda1)
    dlambda1 = dlambda1[dlambda1 > 0.]
    halfwindow = int(np.ceil(1.1*5.*dlam.max()/dlambda1.min()))

    #pure-Python implementation of smooth_gauss.f
    """
    temp = np.zeros(500); gauss = np.zeros(500)
    spec2 = np.zeros(n2)

    for i in range(n2):

        low = f[i] - halfwindow
        if low < 0: low = 0
        high = f[i] + halfwindow
        if (high < 0): high = 0
        if (high > n1 - 1): high = n1 - 1

        if (low < n1) and (low < high):

            temp2 = 0.
            temp3 = 0.
            temp4 = 0.

            for j in range(low,high+1):

                temp5 = lambda1[j] - lambda2[i]

                if (np.abs(temp5) < dlam[i]*40.):
                    gauss = np.exp(-1.*temp5**2./(2.*dlam[i]**2.))
                    temp2 += gauss
                    temp3 += spec1[j]*gauss
                    temp4 += gauss**2.

            if temp2 > 0.:
                spec2[i] = temp3 / temp2
    """

    #Python wrapped fortran implementation of smooth gauss
    spec2 = smooth_gauss(lambda1, spec1, lambda2, dlam, f, halfwindow)

    return spec2

def get_photometry(data, index, phot_path='', phot_type=''):

    """
    Calculate the photometric effective temperature and surface gravity to use
    in the determination of the best fit synthetic spectrum, for a given observed
    spectrum.

    The photometric effective temperature is calculated from an average of effective
    temperatures estimated from the Victoria-Regina ('van'), Padova, and Yonsei-Yale ('yy')
    theoretical isochrones, assuming an age of 14 Gyr and [alpha/Fe] = +0.3 for
    globular clusters. If color temperature information is available, using
    calibration methods of Ramirez & Melendez 2005, it is also used in the determination
    of the effective temperature. Total photometric error is also calcualted, including
    systematic and random components. For a detailed description, see Kirby et al. 2009
    Sec 4.5.

    The photometric surface gravity is calculated by a similar method, except that
    no color-logg relation exists. No error is calculated since the errors are very small
    in general (< 0.06 dex), and MRS spectra cannot constrain logg owing to lack of visible
    ionized lines.

    Parameters
    ----------
    data: dictionary: contains effective temperature information from photometry for
                      a given slitmask
    index: integer: the index corresponding to the object under consideration
    phot_path: string, optional: path to directory containing the relevant data on isochrones
    phot_type: string, optional: if specified as 'sdss', then use the photometry based on
                the SDSS keys in the moogify files

    Returns
    -------
    teffphot: float: estimated photometric effective temperature
    teffphoterr: float: estimated error on teffphot
    loggphot: float: estimated photometric surface gravity
    fehphot: float: estimated photometric metallicity
    """

    if phot_path[-1] != '/': phot_path += '/'

    filename = phot_path+'ages.fits'
    hdulist = fits.open(filename)
    cols = hdulist[1].columns.names
    phot = hdulist[1].data

    if phot_type in ['sdss', '']:

        w14G = []
        for col in cols:
            iso_ages = phot[col][0]
            diff = np.abs(iso_ages - 14.)
            w = np.where(diff == diff.min())[0][0]
            w14G.append(w)

    else:

        w9G = np.where( phot['parsec'][0] == 9. )[0][0]

    if phot_type == 'sdss':

        logg = data['logg_padova_sdss'][index][w14G[1]]
        loggerr = max([data['loggerr_padova_sdss'][index][w14G[1]], 0.1])
        teff = data['teff_padova_sdss'][index][w14G[1]]
        tefferr = max([data['tefferr_padova_sdss'][index][w14G[1]], 100.])


        return teff, tefferr, logg, loggerr

    elif phot_type == 'cfht':

        #Note that right now this code is written specifically
        #for field D (6 Gyr)

        logg = data['logg_cfht'][index]
        loggerr = data['loggerr_cfht'][index]
        if np.isnan(loggerr) or loggerr == 0.: loggerr = 0.1

        teff = data['teff_cfht'][index]
        tefferr = data['tefferr_cfht'][index]
        if np.isnan(tefferr) or tefferr == 0.: tefferr = 100.

        return teff, tefferr, logg, loggerr

    elif phot_type == 'pandas':

        logg = data['logg_pandas'][index][w9G]
        teff = data['teff_pandas'][index][w9G]

        if np.isnan(teff):
            teff = data['teff_cfht'][index]
            tefferr = data['tefferr_cfht'][index]
        else:
            tefferr = data['tefferr_pandas'][index][w9G]

        if np.isnan(logg):
            logg = data['logg_cfht'][index]
            loggerr = data['loggerr_cfht'][index]
        else:
            loggerr = data['teff_pandas'][index][w9G]

        if np.isnan(loggerr) or loggerr == 0.: loggerr = 0.1
        if np.isnan(tefferr) or tefferr == 0.: tefferr = 100.

        return teff, tefferr, logg, loggerr

    elif phot_type == '2mass' or phot_type == 'ferre':

        logg = data['logg_'+phot_type][index]
        loggerr = data['loggerr_'+phot_type][index]
        if np.isnan(loggerr) or loggerr == 0.: loggerr = 0.1

        teff = data['teff_'+phot_type][index]
        tefferr = data['tefferr_'+phot_type][index]
        if np.isnan(tefferr) or tefferr == 0.: tefferr = 100.
        tefferr = max([tefferr, 100.])

        return teff, tefferr, logg, loggerr

    elif phot_type == 'hrs':

        logg = data['loggphot'][index]
        teff = data['teffphot'][index]

        tefferr = data['teffphoterr'][index]
        if np.isnan(tefferr) or tefferr == 0.: tefferr = 100.

        loggerr = data['loggphoterr'][index]
        if np.isnan(loggerr) or loggerr == 0.: loggerr = 0.1

        return teff, tefferr, logg, loggerr

    else: #this is the technique used by moogify

        iso_name = ['van', 'padova', 'yy', 'ram']

        fehs = np.array([data['feh_'+iso_name[0]][index][w14G[0]],
        data['feh_'+iso_name[1]][index][w14G[1]], data['feh_'+iso_name[2]][index][w14G[2]]])

        feherrs = np.array([data['feherr_'+iso_name[0]][index][w14G[0]],
        data['feherr_'+iso_name[1]][index][w14G[1]], data['feherr_'+iso_name[2]][index][w14G[2]]])

        loggs = np.array([data['logg_'+iso_name[0]][index][w14G[0]],
        data['logg_'+iso_name[1]][index][w14G[1]], data['logg_'+iso_name[2]][index][w14G[2]]])

        loggerrs = np.array([data['loggerr_'+iso_name[0]][index][w14G[0]],
        data['loggerr_'+iso_name[1]][index][w14G[1]], data['loggerr_'+iso_name[2]][index][w14G[2]]])

        #Perform a check to ensure that the loggerr values are valid
        if all(g == 0 for g in loggerrs): loggerrs = np.full(len(loggerrs), 0.1)

        loggphot = np.average(loggs, weights=np.reciprocal(loggerrs**2.))

        loggerr_rand = np.average(loggerrs, weights=np.reciprocal(loggerrs**2.))
        loggerr_sys = np.sqrt(np.average((loggs - loggphot)**2., weights=np.reciprocal(loggerrs**2.)))
        loggphoterr = np.sqrt(loggerr_rand**2. + loggerr_sys**2.)

        teffs = [data['teff_'+iso_name[0]][index][w14G[0]],
        data['teff_'+iso_name[1]][index][w14G[1]], data['teff_'+iso_name[2]][index][w14G[2]]]

        tefferrs = [data['tefferr_'+iso_name[0]][index][w14G[0]],
        data['tefferr_'+iso_name[1]][index][w14G[1]], data['tefferr_'+iso_name[2]][index][w14G[2]]]

        #Perform a check to ensure that the tefferr values are valid
        if all(t == 0 for t in tefferrs): tefferrs = np.full(len(tefferrs), 100.).tolist()

        if data['teff_'+iso_name[3]][index] >= 0.:
            teffs.append(data['teff_'+iso_name[3]][index])
            tefferrs.append(data['tefferr_'+iso_name[3]][index])

        teffs = np.array(teffs); tefferrs = np.array(tefferrs)

        teffphot = np.average(teffs, weights=np.reciprocal(tefferrs**2.))

        tefferr_rand = np.average(tefferrs, weights=np.reciprocal(tefferrs**2.))
        tefferr_sys = np.sqrt(np.average((teffs - teffphot)**2., weights=np.reciprocal(tefferrs**2.)))
        teffphoterr = np.sqrt(tefferr_rand**2. + tefferr_sys**2.)

        fehphot = np.average(fehs, weights=np.reciprocal(feherrs**2.))

        feherr_sys = np.sqrt(np.average((fehs - fehphot)**2., weights=np.reciprocal(feherrs**2.)))
        feherr_rand = np.average(feherrs, weights=np.reciprocal(feherrs**2.))
        fehphoterr = np.sqrt(feherr_rand**2. + feherr_sys**2.)

        return teffphot, teffphoterr, loggphot, loggphoterr

def get_slits(slitmask_name, slitmask_path=''):

    filenames = glob.glob(slitmask_path+slitmask_name+'/spec1d.*.fits.gz')
    slit_strs = [re.search('\.\d\d\d\.', file).group()[1:-1] for file in filenames]
    return slit_strs

def calc_breakpoints(arr, npix):
    """
    Helper function for CONT_NORM. Calculate the breakpoints for a given array,
    for use with a B-spline, given a number of pixels to include in each interval.
    """

    #Based on the IDL code FINDBKPT

    nbkpts = int(float(len(arr))/npix)
    xspot = np.array(list(range(nbkpts)))*len(arr)/(nbkpts - 1)
    xspot = np.round(xspot).astype(int)
    if len(xspot) > 2: bkpt = arr[xspot[1:-1]]
    else: bkpt = None

    return bkpt


def slatec_splinefit(x, y, innvar, sigma_l, sigma_u, npix, silent=True, maxiter=5,
fit_type='slatec', pixmask=None):

    if pixmask is None:
        mask = np.full(len(x), True)
    else:
        mask = np.full(len(x), False)
        mask[pixmask] = True
    bad = np.where(innvar <= 0.)[0]
    if len(bad) > 0: mask[bad] = False

    k = 0
    size0 = x[mask].size
    while k < maxiter:

        oldmask = mask

        #Now recalculate the fit after sigma clipping
        if fit_type == 'slatec':
            break_points = calc_breakpoints(x[mask], npix)
            tck = splrep(x[mask], y[mask], w=innvar[mask], t=break_points)
            cont = splev(x, tck)

        if fit_type == 'apogee':

            chpoly = Chebyshev.fit(x[mask], y[mask], 3, w=innvar[mask])
            cont = chpoly(x)

        #SLATEC
        diff = (y - cont)*np.sqrt(innvar)
        wbad = np.where( (diff < -sigma_l)| (diff > sigma_u) | (np.sqrt(innvar) <= 0.) )[0]

        #'STD'
        #diff = y - cont
        #sigma = np.std(diff)
        #wbad = np.where( (diff < -sigma_l*sigma) | (diff > sigma_u*sigma) )[0]

        if len(wbad) == 0:
            if not silent: print('Continuum fit converged, size = '+str(x[mask].size))
            break
        else:
            mask = np.full(len(x), True)
            mask[wbad] = False
            if np.sum(np.abs(mask.astype(int) - oldmask.astype(int))) == 0:
                if not silent: print('Continuum fit converged, size = '+str(x[mask].size))
                break
            else: k += 1

    mask = oldmask
    if k == maxiter:
        if not silent: print('Continuum fit converged, size = '+str(x[mask].size))

    if fit_type == 'slatec':
        return break_points, tck
    if fit_type == 'apogee':
        return cont

def cont_norm(wave, flux, ivar, wvl_range=[4100.,9100.], sigma_l_blue=0.1, sigma_u=5.,
cont_regions=None, sigma_l_red=0.1, cont_mask=None, grating='600ZD', zrest=0., npix=None,
maxiter=None, fit_type='apogee'):
#sigma_l_red = 5. formerly, cont_regions were used for 1200G

    """
    Find the B-spline represenation of a 1D curve.
    Default assumption k = 3, cubic splines recommended. Even values of k should be
    avoided, especially with small s values. 1 <= k <= 5.

    Contiuum regions defined by KGS08 not present in the blue due to the large number
    of absorption lines.
    """

    #Note that typically use npix = 200 and maxiter = 2 for 600ZD and employ the
    #shortened tell cont mask
    if npix is None:
        if grating == '1200G':
            #npix = 100
            #maxiter = 5
            npix = 200; maxiter = 2
        if grating == '600ZD':
            npix = 200; maxiter = 2

    #Mask the telluric regions in the continuum normalization
    #tell_cont_mask = np.array([[6560.,6568.],[6864.,7020.], [7162.,7350.], [7591.,7703.],
    #[8128.,8352.], [8938.,9100.]])/(1. + zrest)

    tell_cont_mask = np.array([[6864., 6935.], [7591., 7694.], [8938.,9100.]])/(1. + zrest)
    #tell_cont_mask = []

    #cat_mask = np.array([[8488., 8508.], [8525., 8561.], [8645., 8679.]])

    #Do separately for the blue and red ends of the CCD
    norm = []; contmask = []
    side = ['blue', 'red']
    continuum_flag = np.ones(2)

    for i in range(len(wave)):

        wavei = wave[i]; fluxi = flux[i]; ivari = ivar[i]

        m = len(wavei); k = 3
        if m <= k:
            sys.stderr.write('Fewer data points (m = {}) than degree of spline (k = {})\
            in {} side of CCD\n'.format(m,k,side[i]))
            norm += [np.full(len(wavei), np.nan)]

        else:

            mask = np.full(len(wavei), False, dtype=bool)

            #Consider only the continuum regions in the red
            if grating == '1200G':

                #for region in cont_regions:
                #    mask[np.where((wavei >= region[0])&(wavei <= region[1]))] = True
                #mask[np.where((fluxi < 0.)|(ivari <= 0.)|(~np.isfinite(ivari))|np.isnan(fluxi))] = False

                mask[np.where((fluxi >= 0.) & (ivari > 0.) & np.isfinite(ivari) & (~np.isnan(fluxi)))] = True

            if grating == '600ZD':

                mask[np.where((fluxi >= 0.) & (ivari > 0.) & np.isfinite(ivari) & (~np.isnan(fluxi)))] = True

            #Mask the telluric regions
            for tellmask in tell_cont_mask:
                mask[np.where((wavei >= tellmask[0])&(wavei <= tellmask[1]))] = False

            #Mask the calcium triplet
            #for camask in cat_mask:
            #    mask[np.where((wavei >= camask[0])&(wavei <= camask[1]))] = False

            #Mask regions problematic to continuum normalization
            if cont_mask is not None:
                for region in cont_mask:
                    mask[np.where((wavei >= region[0])&(wavei <= region[1]))] = False

            #Mask the first 4 pixels at the beginning and end of the spectrum
            mask[0:4] = False; mask[-5:-1] = False

            #Make sure that there are enough points from which to calculate the continuum
            print('npix '+side[i]+' =',len(wavei[mask]))
            if wavei[mask].size < 300:
                sys.stderr.write('Insufficient number of pixels to determine the continuum\n')
                continuum_flag[i] = 0

            #Determine the initial B-spline fit, prior to sigma clipping
            #The initial B-spline fit approximates a running mean of the continuum
            #the function considers only interior knots, end knots added automatically
            #returns tuple of knots, B-spline coefficients, and the degree of the spline

            if grating == '1200G': sigma_l = sigma_l_red
            if grating == '600ZD': sigma_l = sigma_l_blue

            if fit_type == 'slatec':
                bkpt, tck = slatec_splinefit(wavei[mask], fluxi[mask], ivari[mask],
                sigma_l, sigma_u, npix, silent=False, maxiter=maxiter, fit_type=fit_type)
                conti = splev(wavei, tck)
            if fit_type == 'apogee':
                conti = slatec_splinefit(wavei, fluxi, ivari,
                sigma_l, sigma_u, npix, silent=False, maxiter=maxiter, fit_type=fit_type,
                pixmask=mask)

            #Check for any negative values of the continuum determination
            #wzero = conti < 0.
            #if len(conti[wzero]) > 0: conti[wzero] = np.nan

            norm += [conti]
            contmask += [mask]

    flux_norm = []; ivar_norm = []
    for i in range(len(wave)):
        flux_norm.append(flux[i]/norm[i])
        ivar_norm.append(ivar[i]*norm[i]**2.)

    return flux_norm, ivar_norm, norm, contmask, continuum_flag

def cont_refine(waver, flux_norm, ivar_norm, wave_flat, flux_synth, wvl_range=[4100.,9100.],
grating='', sigma_l=3., sigma_u=3., cont_mask=None, norm=None, npix=None, maxiter=None,
zrest=0., fit_type='apogee'):

    """
    Refine the continuum normalization by fitting a B-spline to the quotient of the
    continuum divided, observed spectrum and the best-fit synthethic spectrum. This
    quotient is equivalent to a flat noise spectrum, which should correspond to the
    higher order terms in the continuum. Then divide the CONTINUUM DIVIDED observed spectrum
    by the flat noise spectrum to refine the continuum.
    """

    if npix is None:
        if grating == '1200G':
            npix = 150; maxiter = 5
        if grating == '600ZD':
            npix = 100; maxiter = 2

    cont = []
    #Then determine the flat noise spectrum and the associated fit
    for i in range(len(waver)):

        wavei = waver[i]; fluxi = flux_norm[i]; ivari = ivar_norm[i]

        m = len(wavei); k = 3
        if m <= k:
            side = ['blue', 'red']
            sys.stderr.write('Fewer data points (m = {}) than degree of spline (k = {}) in {} side of CCD\n'.format(m,k,side[i]))
            cont += [np.full(len(wavei), np.nan)]

        else:

            #Find out the values of the flat synthetic spectrum that are in range
            wrange = np.where((wave_flat >= wavei[0])&(wave_flat <= wavei[-1]))[0]
            fsynthi = flux_synth[wrange]

            #Mask nonsensical values from the continuum determination
            mask = np.full(len(wavei), False, dtype=bool)
            mask[np.where((fluxi > 0.) & (ivari > 0.) & np.isfinite(ivari) &\
            np.invert(np.isnan(fluxi)) & (fsynthi > 0.) & np.invert(np.isnan(fsynthi)))] = True
            mask[0:4] = False; mask[-5:-1] = False

            if norm is not None:
                pixmask = np.ones(len(wavei))
                pixmask[np.where((fluxi - norm[i])*np.sqrt(ivari) > 3.)] = 0
                mask[np.where(pixmask == 0)] = False

            #Exclude regions that are problematic to continuum normalization
            if cont_mask is not None:
                for region in cont_mask:
                    mask[np.where((wavei >= region[0])&(wavei <= region[1]))] = False

            #Make sure that the standard masked regions are excluded
            standard_mask = np.array([[4856., 4866.],[5885.,5905.],[6270.,6300.],[6341., 6346.],[6356.,6365.],
            [6559.797, 6565.797], [7662., 7668.],[8113.,8123.], [8317.,8330.],[8488.023, 8508.023],
            [8525.091, 8561.091],[8645.141, 8679.141], [8804.756, 8809.756]])
            #tell_mask = np.array([[6864.,7020.], [7162.,7320.],[7591.,7703.],[8128.,8352.],
            #[8938.,9900.]])/(1. + zrest)
            #standard_mask = np.append(standard_mask, tell_mask, axis=0)

            for ranges in standard_mask:
                mask[np.where(((wavei >= ranges[0]) & (wavei <= ranges[1])))] = False

            ## NOTE, ENK USES JUST INVERSE VARIANCE INSTEAD OF INVERSE VARIANCE
            ## MULTIPLIED BY FSYNTHI

            if fit_type == 'slatec':
                bkpt, tck = slatec_splinefit(wavei[mask], fluxi[mask]/fsynthi[mask],
                ivari[mask]*fsynthi[mask]**2., sigma_l, sigma_u, npix, maxiter=maxiter,
                fit_type=fit_type)
                conti = splev(wavei, tck)
            if fit_type == 'apogee':
                conti = slatec_splinefit(wavei, fluxi/fsynthi,
                ivari*fsynthi**2., sigma_l, sigma_u, npix, maxiter=maxiter,
                fit_type=fit_type, pixmask=mask)

            #plt.figure()
            #plt.plot(wavei[mask], fluxi[mask]/fsynthi[mask])
            #plt.plot(wavei, conti)
            #plt.show()

            cont += [conti]

    #Divide the CONTINUUM DIVIDED observed spectrum by the flat noise B-spline
    flux_new = []; ivar_new = []
    for i in range(len(waver)):
            flux_new.append(flux_norm[i]/cont[i])
            ivar_new.append(ivar_norm[i]*cont[i]**2.)

    #Plotting the refined spectrum as a check
    #plt.figure()
    #plt.plot(waver[0], flux_new[0])
    #plt.plot(waver[1], flux_new[1])
    #plt.ylim(0.7, 1.1)
    #plt.show()
    #plt.close()

    flux_refine = np.array(flux_new[0].tolist() + flux_new[1].tolist())
    ivar_refine = np.array(ivar_new[0].tolist() + ivar_new[1].tolist())
    cont_flat = np.array(cont[0].tolist() + cont[1].tolist())

    return flux_refine, ivar_refine, cont_flat

def telluric_correct(wvlarr, spec, ivarspec, xspec, halfwindow=100., npix=50, slitmask_path='',
grating='', telluric_path='', m31disk=False):

    """
    Correct for telluric absorption in a given observed spectrum using a spectrum
    of a hot star observed with the same science configuration. For the 600ZD grating,
    the hot star is HD066665, observed on April 23 2012 , with an airmass of 1.081. The
    object was observed for 30 s using the LVMslit with the configuration 600ZD/GG455, with
    a Gaussian sky line width of 1.397.

    For the 1200G grating, the spectrophotometric standard is BD +28 4211, observed on
    November 14, 2007, with an airmass of 1.018. The object was observed using the LVMslit
    with the configuration 1200G/OG550.
    """

    #If the telluric standard information file does not exist, then generate it
    if grating == '600ZD':
        if not os.path.exists(telluric_path+'telluric/telluric_standard_norm.pkl'):

            #Load in the file
            telluric_file = glob.glob(telluric_path+'telluric/spec1d.*.fits')[0]
            tell = fits.open(telluric_file)

            xtell = tell[1].header['airmass']

            #Now load in the spectral data for the telluric standard
            blue = tell[1].data
            red = tell[2].data

            wave = np.array([blue['lambda'][0], red['lambda'][0]])
            flux = np.array([blue['spec'][0], red['spec'][0]])
            ivar = np.array([blue['ivar'][0], red['ivar'][0]])

            #Shift the telluric standard into the rest frame
            #vrtell = 50.3; c = 3.*10**5.; zrest = vrtell/c
            #wave = wave/(1 + zrest)

            #Regions most susceptible to telluric absorption, from KGS08
            #[6560.,6568.]
            tellregions = np.array([[6864.,6950.], [7162.,7350.], [7591.,7703.],
                                    [8128.,8352.], [8938.,9100.]])

            tellflux = []; tellivar = []
            for i in range(len(wave)):

                wavei = wave[i]; fluxi = flux[i]; ivari = ivar[i]

                mask = np.full(len(wavei), True, dtype=bool)
                mask[0:4] = False; mask[-5:-1] = False

                #Define the 100 Angstrom bands on the either side of each telluric absorption region
                #in order to normalize the telluric standard. If at the edge of the chip gap,
                #use only a single region on the same chip

                tellregionsi = []
                for j in range(len(tellregions)):
                    if (tellregions[j][0] < wavei[mask].max()) and (tellregions[j][1] > wavei[mask].min()):

                        tellstart = max([min([tellregions[j][0], wavei[mask].max()]), wavei[mask].min()])
                        tellend = min([tellregions[j][1], wavei[mask].max()])
                        tellregionsi.append([tellstart, tellend])
                tellregionsi = np.array(tellregionsi)

                wavestart = tellregionsi[:,0] - halfwindow
                wavestart = np.append(wavestart, tellregionsi[:,1])

                wstart = np.where(wavestart < wavei[mask].min())[0]
                if len(wstart) > 0: wavestart[wstart] = wavei[mask].min()

                waveend = tellregionsi[:,0]
                waveend = np.append(waveend, tellregionsi[:,1] + halfwindow)

                wend = np.where(waveend > wavei[mask].max())[0]
                if len(wend) > 0: waveend[wend] = wavei[mask].max()

                conts = []
                for k in range(len(tellregionsi)):

                    if (wavestart[k] != waveend[k]) and (wavestart[k+len(tellregionsi)] !=\
                    waveend[k+len(tellregionsi)]):
                        wfit = np.where(((wavei > wavestart[k])&(wavei < waveend[k]))|\
                        ((wavei > wavestart[k+len(tellregionsi)])&(wavei < waveend[k+len(tellregionsi)])))[0]

                    elif (wavestart[k] != waveend[k]) and (wavestart[k+len(tellregionsi)] ==\
                    waveend[k+len(tellregionsi)]):
                        wfit = np.where((wavei > wavestart[k])&(wavei < waveend[k]))[0]

                    else:
                        wfit = np.where((wavei > wavestart[k+len(tellregionsi)])&\
                        (wavei < waveend[k+len(tellregionsi)]))[0]

                    wtell = np.where((wavei >= tellregionsi[k][0])&(wavei <= tellregionsi[k][1]))[0]

                    #Fit a straight line to the 100 A windows on either side of the continuum
                    #regions

                    #def f(x, m, b): return m*x + b
                    #m, b = curve_fit(f, wavei[wfit], fluxi[wfit], sigma=ivari[wfit]**(-0.5))[0]
                    #cont = m*wavei[wtell] + b
                    #fit = m*wavei[wfit] + b

                    bkpt0 = calc_breakpoints(wavei[wfit], 75)
                    tck0 = splrep(wavei[wfit], fluxi[wfit], w=ivari[wfit], t=bkpt0)
                    cont0 = splev(wavei[wtell], tck0)

                    fluxi[wtell] /= cont0
                    ivari[wtell] *= cont0**2.

                #Then define a mask, masking out pixels that fall within the telluric regions
                tellmask = np.full(len(wavei), True)
                for region in tellregionsi:
                    tellmask[np.where((wavei >= region[0])&(wavei <= region[1]))] = False

                #Set all remaining pixels to unity
                fluxi[tellmask] = 1.
                #ivarflat[mask] *= fluxflat[mask]**2.
                ivari[tellmask] = 1.e10

                tellflux += [fluxi]
                tellivar += [ivari]

            waveflat = np.array(wave[0].tolist() + wave[1].tolist())
            fluxflat = np.array(tellflux[0].tolist() + tellflux[1].tolist())
            ivarflat = np.array(tellivar[0].tolist() + tellivar[1].tolist())

            telluric = {'wave': waveflat, 'spec': fluxflat, 'ivar': ivarflat, 'xtell': xtell}

            with open(telluric_path+'/telluric/telluric_standard_norm.pkl', 'wb') as f:
                pickle.dump(telluric, f)
                f.close()

        #If the telluric standard information file exists, then load in the data
        else:
            with open(telluric_path+'/telluric/telluric_standard_norm.pkl', 'rb') as f:
                telluric = pickle.load(f, encoding='latin1')

    if grating == '1200G':

        tell = fits.open(telluric_path+'telluric/deimos_telluric_1.0.fits')
        tell = tell[1].data[8] #select only the data for BD +28 4211

        #w = tell['spec'] > 0.
        #telluric = {'wave': tell['lambda'][w], 'spec': tell['spec'][w], 'ivar': tell['ivar'][w],
        #'xtell': tell['airmass']}

        telluric = {'wave': tell['lambda'], 'spec': tell['spec'], 'ivar': tell['ivar'],
        'xtell': tell['airmass']} #more similar to what Evan has

    #Then, divide the observed spectrum by the template adjusted by the ratio of the airmasses,
    #following the Beer-Lambert law

    aratio = xspec/telluric['xtell']
    tellspec = telluric['spec']**aratio
    tellivar = telluric['ivar']*(telluric['spec']/(aratio*tellspec))**2.

    ivarmissing = 10.e10
    w = np.where(tellivar >= 1.e8)[0]
    if len(w) > 0: tellivar[w] = ivarmissing

    #Flatten the given spectrum wavelength array before proceeding, because
    #the chip gaps are not located at the same place

    wvlarr_flat = np.array(wvlarr[0].tolist() + wvlarr[1].tolist())
    spec_flat = np.array(spec[0].tolist() + spec[1].tolist())
    ivar_flat = np.array(ivarspec[0].tolist() + ivarspec[1].tolist())

    #f = np.where((np.isfinite(tellspec))&(tellspec > 0.)&(np.isfinite(tellivar))&\
    #(tellivar > 0.)&(tellivar < 1.e8)&(telluric['wave'] >= wvlarr_flat[0])&\
    #(telluric['wave'] <= wvlarr_flat[-1]))[0]

    f = np.where((np.isfinite(tellspec))&(tellspec > 0.)&(np.isfinite(tellivar))&\
    (tellivar > 0.)&(tellivar < 1.e8))[0] #more similar to what Evan has

    """
    #Set up for the telluric cross correlation
    wtell = np.zeros(len(wvlarr_flat))
    for i in range(len(tellstart)):
        w = np.where( (wvlarr_flat >= tellstart[i]) & (wvlarr_flat < tellend[i]) )[0]
        if len(w) > 0: wtell[w] = 1
    """

    if len(f) == 0:
        sys.stderr.write('No valid telluric regions in target wavelength array\n')
        return

    #Now interpolate the telluric spectrum onto the wavelength array of the given spectrum

    f_flux = interp1d(telluric['wave'][f], tellspec[f], fill_value=1., bounds_error=False)
    f_ivar = interp1d(telluric['wave'][f], tellivar[f], fill_value=ivarmissing, bounds_error=False)
    #f_ivar = interp1d(telluric['wave'], tellivar, fill_value=ivarmissing, bounds_error=False)

    #note that in Evan's code, he uses the full inverse variance array here instead of
    #just the tell regions

    t = f_flux(wvlarr_flat)
    tivar = f_ivar(wvlarr_flat)

    d = spec_flat/t
    divar = (ivar_flat**(-1.) + d**2.*tivar**(-1.))**(-1.) * t**2.

    if not m31disk:

        #Separate the results back into original array form, corresponding to each side
        #of the chip gap
        wblue = np.where(wvlarr_flat <= wvlarr[0][-1])[0]
        wred = np.where(wvlarr_flat >= wvlarr[1][0])[0]

        dt = np.array([d[wblue], d[wred]])
        dtivar = np.array([divar[wblue], divar[wred]])
        return dt, dtivar

    else:
        return d, divar

def calc_sn(wvl, flux, grating='600ZD', cont_regions=None, continuum=None,
zrest=0., contmask=None):

    """
    Calculate the signal to noise ratio for a given grating. Note that the signal
    to noise is estimated differently for the 600ZD vs. 1200G grating. Most significantly,
    the 1200G S/N is determined using only "continuum regions" and after the initial
    attempt at continuum normalization (Sec 6.4 of KGS08). Owing to the fact that
    continuum normalization is more difficult for the 600ZD grating, we estimate the S/N
    after the determination of the best-fit synthetic spectrum and the iterative
    continuum refinement process
    """

    tell_cont_mask = np.array([[6864., 6935.], [7591., 7694.], [8938.,9100.]])/(1. + zrest)

    if grating == '1200G':

        pixel_scale = 0.33 #in Angstroms

        if contmask is None:

            contmask = np.full(len(wvl), False, dtype=bool)
            for region in cont_regions:
                contmask[np.where((wvl >= region[0]) & (wvl <= region[1]))] = True

            for mask in tell_cont_mask:
                contmask[np.where((wvl >= mask[0]) & (wvl <= mask[1]))] = False

            nhalf = int(round(float(len(wvl))/2.)); n = len(wvl)
            contmask[0:2] = False
            contmask[nhalf-3:nhalf+2] = False
            contmask[n-3:n-1] = False

        else:
            if contmask.dtype != 'bool':
                contmask = np.array([bool(item) for item in contmask])

    if grating == '600ZD':

        pixel_scale = 0.64 #in Angstroms

        if contmask is None:

            contmask = np.full(len(wvl), True, dtype=bool)
            for mask in tell_cont_mask:
                contmask[np.where((wvl >= mask[0]) & (wvl <= mask[1]))] = False

            nhalf = int(round(float(len(wvl))/2.)); n = len(wvl)
            contmask[0:2] = False
            contmask[nhalf-3:nhalf+2] = False
            contmask[n-3:n-1] = False

        else:
            if contmask.dtype != 'bool':
                contmask = np.array([bool(item) for item in contmask])

    wcont = np.where(contmask[3:-4])[0]+3
    wcont = wcont[np.where( np.isfinite(flux[wcont]) & np.isfinite(continuum[wcont]) &\
            (continuum[wcont] != 0.) & ~np.isnan(continuum[wcont]) & ~np.isnan(flux[wcont]) )]

    dev = np.abs( (flux[wcont] - continuum[wcont])/ continuum[wcont] )
    #avgdev = weighted_avg_std(abs_dev, ivar[sn_mask])[0]
    #avgdev = np.mean(dev)
    meddev = np.median(dev)

    #w = np.where(dev < 3.*avgdev)[0]
    w = np.where(dev < 3.*meddev)[0]
    if len(w) > 0:
        #sn_pix = 1./np.mean(dev[w])
        sn_pix = 1./np.median(dev[w])
        sn_ang = sn_pix/np.sqrt(pixel_scale)
    else: sn_ang = np.nan

    return sn_ang

def find_chip_gap_spec1d(wave, flux, ivar):
    """
    In the event of a flattened array containing regions of zero flux, find the chip
    gap and separate the array into the blue and red sides of the CCD, while excluding 5
    pixels at the beginning and end of the spectrum around the zero regions, and 5 pixels
    around the chip gap.

    Instead of using an algorithm to infer the location of the chip gap, this verison
    uses the existing spec1d files to find the precise location of the chip gap
    """

    """
    obs_file = glob.glob(slitmask_path+slitmask_name+'/spec1d.*'+slit_str+'*.fits.gz')[0]
    obs_spec = fits.open(obs_file) #open files

    #Get the observed spectral data
    blue = obs_spec[1].data['lambda'][0]
    red = obs_spec[2].data['lambda'][0]
    obs_spec.close()

    #ccd_mask = np.array([blue[6], blue[-5], red[6], red[-5]])
    ccd_mask = np.array([blue[0], blue[-1], red[0], red[-1]])

    wchip1 = np.where((wave >= ccd_mask[0])&(wave <= ccd_mask[1]))[0]
    wchip2 = np.where((wave >= ccd_mask[2])&(wave <= ccd_mask[3]))[0]
    """

    nhalf = int(round(float(len(wave))/2.))
    wchip1 = np.array(list(range(nhalf)))
    wchip2 = wchip1 + len(wchip1)

    wvarr = np.array([wave[wchip1], wave[wchip2]])
    fluxarr = np.array([flux[wchip1], flux[wchip2]])
    ivarr = np.array([ivar[wchip1], ivar[wchip2]])

    return wvarr, fluxarr, ivarr

def helio_correct(moog_data, w, mjd):

    """
    Apply a heliocentric correction to the 600ZD data, to account for differences in the
    date of observation and therefore heliocentric radial velocity of stars for the zrest
    as measured in the 1200G moogify files vs. the 600ZD data

    Note: Precision using radial_velocity_correction module from astropy is 3 m/s. For
    more precise corrections, use a barycentric correction.
    """

    #Dates of observation for the 600ZD masks, on which we are relying on 1200G data
    #for the radial velocity
    #obs_dates = {'n2419c': '2015-10-9', 'n6656b': '2015-5-18', 'n6864a': '2015-5-19',
    #'1904l2': '2015-10-8', 'f130_2': Time(moog_data['jdobs'][w], format='jd').iso}

    jdobs = moog_data['jdobs'][w]
    zrest0 = moog_data['zrest'][w]
    ra = moog_data['ra'][w]
    dec = moog_data['dec'][w]

    time0 = Time(jdobs, format='jd', scale='utc')
    time0.format = 'iso'

    #if mjd is None: time = Time(obs_dates[slitmask_name], format='iso', scale='utc')
    time = Time(mjd, format='mjd', scale='utc')
    time.format = 'iso'

    keck = EarthLocation.from_geodetic(lat=19.8283*units.deg, lon=-155.4783*units.deg,
    height=4160*units.m)
    sc = SkyCoord(ra=ra*units.deg, dec=dec*units.deg)

    #Calculate the velocity correction in m/s
    #Note that the astropy function returns an additive (not subtractive) quantity
    heliocorr0 = sc.radial_velocity_correction('heliocentric', obstime=time0, location=keck).to(units.m/units.s)
    heliocorr = sc.radial_velocity_correction('heliocentric', obstime=time, location=keck).to(units.m/units.s)

    zrest = zrest0 + (heliocorr0 - heliocorr)/const.c

    return zrest.value
