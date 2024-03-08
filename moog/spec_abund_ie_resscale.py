"""
@author: Ivanna Escala 2018-2022
"""

from scipy.optimize import curve_fit
from astropy.io import fits
import numpy as np
import read_synth
import data_redux
import pickle
import scipy
import glob
import sys
import os

def spec_abund(slitmask_name, slit_str=None, slit_num=None, feh_def = -2., alphafe_def = 0.,
resscale_def=0.85, resscale_bounds=[0.1, 1.0], synth_path_blue='', synth_path_red='',
slitmask_path='', moogify_path='', mask_ranges=None, flex_factor=400., mask_path='',
save_path='', replace=False, grating='600ZD', cont_mask=None, spec1dfile=True,
telluric_path='', wgood=False, object_id=None,
resscaleerr = 0.1):

    """
    Measure Teff, [Fe/H], [alpha/Fe], and resolution (Dlam) from spectral synthesis of
    medium-resolution spectra

    Parameters
    ----------
    slitmask_name: str: name of the slitmask
    slit_num: int: slit number on slitmask, if given
    slit_str: str: string corresponding to slit number of slitmask. NOTE: either the
              slit string or slit number must be given
    feh_def: float, optional: initial metallicity to assume in the Levenberg-Marquardt
             minimization
    alphafe_def: float, optional: initial alpha abundance to assume in the Levenberg-Marquardt
                 minimization
    resscale_def: float, optional: initial slit filling factor (or resolution scale parameter) t
              to assume when smoothing the synthethic spectrum. This accounts for the fact
              that the resolution of the star is likely higher than determined by the initial
              spectral resolution determination.
    resscale_bounds: array-like, optional: bounds to limit the values of resscale in the fitting.
    mask_ranges: array-like, optional: 2D array containing a list of ranges in Angstroms
                 to mask in the minimization
    flex_factor: float, optional: an arbitrary weighting factor to use when deciding the
                        relative influence of the photometric temperature in the fit, e.g.,
                        if flex_factor = 1, the "effective temperature pixel", which ensures
                        that the spectroscopic temeprature remains within the constraints
                        of the photometric temperature, has equal weight to all pixels
                        in the observed spectrum
    synth_path: string, optional: full path to parent directory containing the synthetic
                spectra -- options for both blue (4100 - 6300) and red (6300 - 9100) grids
    phot_path: string: full path to the directory containing the photometric data (ages.fits)
    telluric_path: string: full path to the directory containing the folder 'telluric'
    slitmask_path: string, optional: fulll path to directory containing data for a given
                   slitmask
    moogify_path: string, optional: full path to directory containing gzipped MOOGIFY files,
                  which are outputs from Evan Kirby's abundance analysis routine, containing
                  information such as photometry, wavelength shift, etc.
    mask_path: string, optional: full path to directory containing wavelength masks, e.g.,
               custom pixel masks for a given spectrum, wavelength region masks to measure
               the abundance of a given element, etc.
    save_path: string, optional: full path to directory where output data is saved
    replace: boolean, optional: if True, replace pre-existing output files
    grating: string, optional: grating used to collect slitmask data, relevant for
             specifying which wavelength masks to use
    cont_mask: 2D array-like, optional: an additional mask to use in the initial continuum
                normalization, if neccessary
    spec1dfile: boolean, optional: if True, then the wavlength, flux, and inverse variance
                arrays are read directly from the spec1d files. If False, then the arrays
                are read from ENK's moogify files. In the latter case, the arrays have
                been flattened and tweaks to the wavelength solution have been applied.
    zrv: float, optional: apply a wavelength shift (in terms of redshift) to the DEIMOS
            spectrum from the spec1d file, if given.
    phot_type: string, optional: if specified as 'sdss', then use the photometry based on
                the SDSS keys in the moogify files
    object_id: string, optional: specify the object ID that describes the relevant filename,
               if using the longslit

    Returns
    -------
    specabund: dict: output result of minimization routine containing best-fit parameters
               for teff, logg, feh, alphafe, and dlam, the associated errors based on the
               square root of the diagnols of the covariance matrices from the fits for
               each parameters, the associated chi-squared contours for each parameter fit,
               the final best fit synthetic spectrum, the continuum-refined observed
               spectrum, etc.
    """

    #Convert the slit number to a string, or the slit string to a slit number
    if slit_num is not None:
        slit_str = str(slit_num)
        if len(slit_str) == 1: slit_str = '00'+slit_str
        elif len(slit_str) == 2: slit_str = '0'+slit_str
        else: pass

    if slit_str is not None:
        slit_num = int(slit_str)

    if not replace:
        if slit_str is not None:
            filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+slit_str+'_resscale.pkl'
        else:
            filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+object_id+'_resscale.pkl'
        if os.path.exists(filename): return
        else: pass

    if spec1dfile:

        #Find filenames (for a typical slitmask)
        if slit_str is not None:
            obs_file = glob.glob(slitmask_path+slitmask_name+'/spec1d.'+slitmask_name+'.'+slit_str+'.*.fits.gz')[0]
        #Find filenames (for single longslit observations)
        else:
            obs_file = glob.glob(slitmask_path+slitmask_name+'/spec1d.*'+object_id+'.fits.gz')[0]

        obs_spec = fits.open(obs_file) #open files

        #Get the observed spectral data
        blue = obs_spec[1].data
        red = obs_spec[2].data

        #Define the airmass
        xspec = obs_spec[1].header['airmass']
        mjd = float(obs_spec[1].header['mjd-obs'])
        obs_spec.close()

        #if both chips have data
        if blue is not None and red is not None:
            wave = np.array([blue['lambda'][0], red['lambda'][0]])
            flux = np.array([blue['spec'][0], red['spec'][0]])
            ivar = np.array([blue['ivar'][0], red['ivar'][0]])

        if red is None:
            nan_arr = np.full(4096, np.nan)
            wave = np.array([blue['lambda'][0], nan_arr])
            flux = np.array([blue['spec'][0], nan_arr])
            ivar = np.array([blue['ivar'][0], nan_arr])

        ### Ensure that the wavelength is monotonically increasing ###

        def monotonic(x):
            xf = np.concatenate( (x[0], x[1]), axis=None )
            dx = np.diff(xf)
            return np.all(dx <= 0) or np.all(dx >= 0)

        if not monotonic(wave):
            sys.stderr.write('ERROR with wavelength solution: not monotonic. Skipping '+\
                slitmask_name+' slit number '+slit_str+'\n')
            return

    if slit_str is not None:
        moog_file = glob.glob(moogify_path+slitmask_name+'/moogify.fits.gz')[0]
    else:
        moog_file = glob.glob(moogify_path+slitmask_name+'/'+slitmask_name+'.fits.gz')[0]

    moog = fits.open(moog_file)

    #Determine which parameters each file contains
    moog_data = moog[1].data
    moog.close()

    if slit_num is not None:

        slit = moog_data['slit']
        try:
            w = np.where(slit == slit_num)[0][0]
            if wgood:
                if moog_data['good'][w] == 0:
                    sys.stderr.write('Not a good spectrum for '+slitmask_name+' slit number '+\
                    slit_str+'\n')
                    return
        except:
            return

    else:
        w = np.where(moog_data['objname'] == object_id)[0][1]

    if not spec1dfile:

        xspec = moog_data['airmass'][w]
        wavew = moog_data['lambda'][w]
        fluxw= moog_data['spec'][w]
        ivarw = moog_data['ivar'][w]

        wave, flux, ivar = data_redux.find_chip_gap_spec1d(wavew, fluxw, ivarw, slit_str,
        slitmask_path=slitmask_path, slitmask_name=slitmask_name)

    #Perform the telluric absorption correction
    tell_result = data_redux.telluric_correct(wave, flux, ivar, xspec,
    slitmask_path=slitmask_path, grating=grating, telluric_path=telluric_path)

    if tell_result is not None: flux_tell, ivar_tell = tell_result
    else: return

    #Identify the redshift of the object and shift the observed spectrum tos the rest frame
    zrest = moog_data['zrest'][w]
    wave_rest = wave/(1. + zrest)

    #Load in the photometric data of the object for the chi-squared fitting

    #deprecated
    #teffphot, teffphoterr, loggphot, loggphoterr = data_redux.get_photometry(moog_data, w,
    #phot_path=phot_path, phot_type=phot_type)

    teffphot, teffphoterr = moog_data['teffphot'][w], moog_data['teffphoterr'][w]
    loggphot, loggphoterr = moog_data['loggphot'][w], moog_data['loggphoterr'][w]

    if (teffphot < 0) or (loggphot < 0) or np.isnan(teffphot) or np.isnan(loggphot):
        if slit_str is not None:
            sys.stderr.write('Nonsense photometry: skipping '+slitmask_name+' slit number'+slit_str+'\n')
            return
        else:
            sys.stderr.write('Nonsense photometry: skipping '+slitmask_name+' object ID '+object_id+'\n')
        return

    #Construct the mask for the continuum normalization
    cont_regions = data_redux.check_pkl_file_exists('mask', file_path=mask_path+'contregion/'+grating+'/')

    #Perform the initial continuum normalization
    norm_result = data_redux.cont_norm(wave_rest, flux_tell, ivar_tell, cont_regions=cont_regions,
    cont_mask=cont_mask, grating=grating, zrest=zrest)

    if norm_result == None: return
    flux_norm, ivar_norm, norm, contmask, cont_flag = norm_result

    #Flatten the arrays
    def flatten(arr):
        return np.array(arr[0].tolist() + arr[1].tolist())

    wave_flat = flatten(wave_rest)
    flux_flat = flatten(flux_norm)
    ivar_flat = flatten(ivar_norm)

    ## Grab the spectral resolution as a function of wavelength from the moogify file
    dlam = moog_data['dlam'][w]

    #Calculate an approximation of the signal-to-noise for the 1200G grating
    if grating == '1200G':
        sn = data_redux.calc_sn(wave_flat, flatten(flux_tell), grating=grating,
        cont_regions=cont_regions, continuum=flatten(norm), zrest=zrest)

    ## Construct the wavelength masks ##

    #Now construct a mask based on regions to fit for [Fe/H]
    if grating == '600ZD':

        feh_mask_blue = data_redux.check_pkl_file_exists('mask_fe_'+grating+'_final',
        file_path=mask_path+'specregion/blue/')

        feh_mask_red = data_redux.check_pkl_file_exists('mask_fe_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        feh_mask = np.array(feh_mask_blue.tolist() + feh_mask_red.tolist())

        #Now construct a mask based on regions to fit for [alpha/Fe]
        alphafe_mask_blue = data_redux.check_pkl_file_exists('mask_alphafe_'+grating+'_final',
        file_path=mask_path+'specregion/blue/')

        alphafe_mask_red = data_redux.check_pkl_file_exists('mask_alphafe_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        alphafe_mask = np.array(alphafe_mask_blue.tolist() + alphafe_mask_red.tolist())

        #Construct additional masks for trying to measure just [Mg/Fe] and [Ca/Fe]
        cafe_mask_blue = data_redux.check_pkl_file_exists('mask_ca_'+grating+'_final',
        file_path=mask_path+'specregion/blue/')

        cafe_mask_red = data_redux.check_pkl_file_exists('mask_ca_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        cafe_mask = np.array(cafe_mask_blue.tolist() + cafe_mask_red.tolist())

        mgfe_mask_blue = data_redux.check_pkl_file_exists('mask_mg_'+grating+'_final',
        file_path=mask_path+'specregion/blue/')

        mgfe_mask_red = data_redux.check_pkl_file_exists('mask_mg_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        mgfe_mask = np.array(mgfe_mask_blue.tolist() + mgfe_mask_red.tolist())

    if grating == '1200G':

        feh_mask = data_redux.check_pkl_file_exists('mask_fe_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        alphafe_mask = data_redux.check_pkl_file_exists('mask_alphafe_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        cafe_mask = data_redux.check_pkl_file_exists('mask_ca_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        mgfe_mask = data_redux.check_pkl_file_exists('mask_mg_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

    feh_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, fit_ranges=feh_mask, zrest=zrest)

    alphafe_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, fit_ranges=alphafe_mask, zrest=zrest)

    cafe_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, fit_ranges=cafe_mask, zrest=zrest)

    mgfe_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, fit_ranges=mgfe_mask, zrest=zrest)

    #In the event that you are measuring Mg using only MgT, which is not included in
    #the spectrum wavelength range
    if len(wave_flat[mgfe_fit_mask]) == 0: return

    #Construct a general mask for the spectrum
    spec_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, zrest=zrest)

    #Identify whether there is wavelength information in the blue and/or red
    #synthetic spectra ranges
    wb = np.where((wave_flat[spec_mask] >= 4100.)&(wave_flat[spec_mask] < 6300.))[0]
    wr = np.where((wave_flat[spec_mask] >= 6300.)&(wave_flat[spec_mask] < 9100.))[0]

    #Initialize a hash table to use to store the synthetic spectral data in memory
    hash_blue = {}; hash_red = {}

    def construct_synth(wvl, teff_in, feh_in, alphafe_in, hash_red=None, hash_blue=None):
        """
        Helper function for the get_synth() functions
        """

        if len(wb) > 0:
            wvl_synth_blue, relflux_synth_blue = read_synth.read_interp_synth(teff=teff_in,
            logg=loggphot, feh=feh_in, alphafe=alphafe_in, data_path=synth_path_blue,
            hash=hash_blue)

            #Make sure that the synthetic spectrum is within the data range for observations,
            #given the use of different gratings and central wavelengths
            wsynthb = np.where((wvl_synth_blue > wvl[0])&(wvl_synth_blue < wvl[-1]))[0]

        if len(wr) > 0:
            wvl_synth_red, relflux_synth_red = read_synth.read_interp_synth(teff=teff_in,
            logg=loggphot, feh=feh_in, alphafe=alphafe_in, data_path=synth_path_red,
            hash=hash_red, start=6300., sstop=9100.)

            wsynthr = np.where((wvl_synth_red > wvl[0])&(wvl_synth_red < wvl[-1]))[0]

        if len(wb) > 0 and len(wr) == 0:
            wvl_synth = wvl_synth_blue[wsynthb]; relflux_synth = relflux_synth_blue[wsynthb]
        elif len(wb) == 0 and len(wr) > 0:
            wvl_synth = wvl_synth_red[wsynthr]; relflux_synth = relflux_synth_red[wsynthr]
        else:
            wvl_synth = np.append(wvl_synth_blue[wsynthb], wvl_synth_red[wsynthr])
            relflux_synth = np.append(relflux_synth_blue[wsynthb], relflux_synth_red[wsynthr])

        return wvl_synth, relflux_synth

    #When an overall metallicity change occurs, repeat the refinement until the value
    #does not change significantly -- define significantly as within 0.1 dex
    #Based on Shetrone et al. 2009

    #Set the initial parameters
    phot_dict = {'teffphot': teffphot, 'loggphot': loggphot, 'teffphoterr': teffphoterr,
    'loggphoterr': loggphoterr, 'flex_factor': flex_factor}

    if teffphot > 8000.: teffphot = 8000.
    if teffphot < 3500.: teffphot = 3500.

    if teffphot < 4100.: fehmin = -4.5
    else: fehmin = -5.

    maxiter = 50
    feh_thresh = 0.001; alphafe_thresh = 0.001; resscale_thresh = 0.001; teff_thresh = 1.

    feh0 = feh_def; alphafe0 = alphafe_def; resscale0 = resscale_def; teff0 = teffphot

    i = 0; converge_flag = 0
    while i < maxiter:

        def get_synth_step1(wvl, teff_fit, feh_fit, resscale_fit):

            """
            Define the function to be used in the curve fitting, such that
            ydata = f(xdata, *params) + eps and r = ydata - f(xdata, *params),
            where f is the model function

            In this iteration, the effective temperature is determined within the constraints
            of photometry, the metallicity is varied, and the resolution. [alpha/Fe] is
            fixed constant at solar

            Parameters
            ----------
            wvl: array-like: the flattened wavelength array of the observed spectrum
            teff_fit: float: spectroscopic effective temperature, to be fitted
            feh_fit: float: spectroscopic metallicity, to be fitted
            dlam_fit: float: FWHM spectral resolution, to be fitted

            Returns
            -------
            relflux_synth_interp: array-like: the synthethic spectrum with the given
                                  spectroscopic effective temperature, metallicity,
                                  default alpha abundance, and photometric surface gravity,
                                  interpolated onto the same wavelengtha array as the observed
                                  spectrum and smoothed to the given FWHM spectral resolution
            """
            #Read in the synthetic spectrum with both blue and red components
            wvl_synth, relflux_synth = construct_synth(wvl, teff_fit, feh_fit, alphafe0,
            hash_red=hash_red, hash_blue=hash_blue)

            #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array

            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl[2:], resscale_fit*dlam[feh_fit_mask])

            #Insert the effective temperature pixel to the beginning of the synthetic spectrum
            flux_synth_teff = np.insert(relflux_interp_smooth, 0, teff_fit)
            flux_synth_teff = np.insert(flux_synth_teff, 1, resscale_fit)

            return flux_synth_teff

        #Perform the fit for [Fe/H], Dlam, Teff

        #Insert the effective temperature pixel at the beginning of the observed spectrum
        #and the inverse variance array

        params0 = [teff0, feh0, resscale0]

        flux_obs_teff = np.insert(flux_flat[feh_fit_mask], 0, teffphot)
        flux_obs_teff = np.insert(flux_obs_teff, 1, resscale_def)

        wave_teff = np.insert(wave_flat[feh_fit_mask], 0, wave_flat[feh_fit_mask][0])
        wave_teff = np.insert(wave_teff, 1, wave_flat[feh_fit_mask][0])

        sigma_teff0 = ivar_flat[feh_fit_mask]**(-0.5)
        npix_fit = float(len(sigma_teff0))

        sigma_teff = np.insert(sigma_teff0, 0, teffphoterr * np.sqrt(flex_factor/npix_fit))
        sigma_teff = np.insert(sigma_teff, 1, resscaleerr)

        try: best_params0, covar0 = curve_fit(get_synth_step1, wave_teff, flux_obs_teff, p0=params0,
        sigma=sigma_teff, bounds=([3500., fehmin, resscale_bounds[0]],[8000., 0., resscale_bounds[1]]),
        absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)
        except RuntimeError: return

        teff_best0, feh_best0, resscale_best0 = best_params0

        def get_synth_step2(wvl, alphafe_fit):
            """
            In this iteration, the effective temperature, metallicity, and smoothing parameter
            are held constant at the values from step1, and the alpha abundance is varied
            along with the FWHM spectral resolution.
            """

            #Read in the synthetic spectrum
            wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best0, alphafe_fit,
            hash_red=hash_red, hash_blue=hash_blue)

            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, resscale_best0*dlam[alphafe_fit_mask])

            return relflux_interp_smooth

        #Perform the fit [alpha/Fe]
        params1 = [alphafe0]
        sigma_flat = ivar_flat[alphafe_fit_mask]**(-0.5)

        best_params1, covar1 = curve_fit(get_synth_step2, wave_flat[alphafe_fit_mask],
        flux_flat[alphafe_fit_mask], p0=params1, sigma=sigma_flat, bounds=([-0.8], [1.2]),
        absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

        def get_best_synth(wvl, alphafe_best):
            """
            In this iteration, the effective temperature, metallicity, and smoothing parameter
            are held constant at the values from step1, and the alpha abundance is varied
            along with the FWHM spectral resolution.
            """

            #Read in the synthetic spectrum
            wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best0, alphafe_best,
            hash_red=hash_red, hash_blue=hash_blue)

            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, resscale_best0*dlam)

            return relflux_interp_smooth

        alphafe_best1 = best_params1[0]
        best_synth = get_best_synth(wave_flat, alphafe_best1)

        #Refine the continuum normalization
        flux_refine, ivar_refine, refinedcont = data_redux.cont_refine(wave_rest, flux_norm, ivar_norm,
        wave_flat, best_synth, cont_mask=cont_mask, norm=None, grating=grating)

        if (np.abs(feh_best0 - feh0) < feh_thresh) and (np.abs(alphafe_best1 - alphafe0) <\
        alphafe_thresh) and (np.abs(resscale_best0 - resscale0) < resscale_thresh) and\
        (np.abs(teff_best0 - teff0) < teff_thresh):
            print('Continuum iteration converged')
            converge_flag = 1
            break
        else:
            print(teff_best0, feh_best0, alphafe_best1, resscale_best0)
            feh0 = feh_best0; alphafe0 = alphafe_best1; resscale0 = resscale_best0; teff0 = teff_best0
            flux_flat = flux_refine; ivar_flat = ivar_refine
            i += 1

    #Calculate the chi-squared value at termination of the continuum iteration
    chi0 = data_redux.compare_spectra(flux_obs_teff, get_synth_step1(wave_teff,
    teff_best0, feh_best0, resscale_best0), stat_name='chisqr',  weights=sigma_teff**(-0.5),
    nvarys=len(params0))

    #Calculate the chi-squared statistic
    chi1 = data_redux.compare_spectra(flux_refine[alphafe_fit_mask],
    get_synth_step2(wave_flat[alphafe_fit_mask], alphafe_best1), stat_name='chisqr',
    weights=ivar_refine[alphafe_fit_mask], nvarys=len(params1))

    #Final best fit parameters
    best_params = [teff_best0, feh_best0, alphafe_best1, resscale_best0]
    print(best_params)

    #Unreduced parameter errors based on the covariance matrix returned by the fit
    #(Statistical uncertainty)

    params_err = np.sqrt([np.diag(covar0)[0], np.diag(covar0)[0], np.diag(covar1)[0],
                          np.diag(covar0)[2]])
    print(params_err)

    #Now perform a check of the uncertainties returned by the fit by analyzing the full
    #chi-squared distribution about the minimum in UNREDUCED chi-squared space

    params_dict = {'teff': best_params[0], 'logg': loggphot, 'feh': best_params[1],
    'alphafe': best_params[2], 'resscale': best_params[3], 'dlam': dlam}

    hash_blue = {}; hash_red = {}

    #Now scale the statistical uncertainties by the reduced chi-squared value to construct
    #the fit uncertainty

    chiteff = chi0/float(len(flux_refine[feh_fit_mask])-1)
    chifeh = chi0/float(len(flux_refine[feh_fit_mask]))
    chialpha = chi1/float(len(flux_refine[alphafe_fit_mask]))
    chiresscale = chiteff

    params_err *= np.sqrt( np.array( [chiteff, chifeh, chialpha, chiresscale] ) )

    """
    #Calculate the signal-to-noise ratio for the 600ZD grating
    if grating == '600ZD':
        sn = data_redux.calc_sn(wave_flat, flux_refine, grating=grating, continuum=best_synth,
        zrest=zrest)
    """

    #Save the data
    specabund = params_dict.copy()
    specabund.update({'tefferr': params_err[0], 'feherr': params_err[1],
    'alphafeerr': params_err[2], 'resscaleerr': params_err[3], 'chifeh': chifeh,
    'teffphot': teffphot, 'teffphoterr': teffphoterr, 'converge_flag': converge_flag,
    'fehnpix': len(flux_refine[feh_fit_mask]), 'alphafenpix': len(flux_refine[alphafe_fit_mask]),
    'npix': len(flux_refine[spec_mask]), 'chialpha': chialpha,
    'wvl': wave_flat, 'flux': flux_refine, 'ivar': ivar_refine, 'synth': best_synth,
    'chiteff': chiteff, 'teffnpix': len(flux_refine[feh_fit_mask])-1,
    'resscalenpix': len(flux_refine[feh_fit_mask])-1,
    'chiresscale': chiresscale, 'objname': moog_data['objname'][w], 'slit': int(slit_str)})

    if not os.path.exists(save_path+slitmask_name): os.makedirs(save_path+slitmask_name)

    if slit_str is not None:
        save_filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+slit_str+'_resscale.pkl'
    else:
        save_filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+object_id+'_resscale.pkl'

    with open(save_filename, 'wb') as f:
       pickle.dump(specabund, f)

    return specabund
