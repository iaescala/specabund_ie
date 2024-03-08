"""
@author: Ivanna Escala 2018-2022
"""

from scipy.optimize import curve_fit
from astropy.io import fits
import numpy as np
import read_synth_ts_cub as read_synth
import data_redux_ts_cub as data_redux
import pickle
import scipy
import glob
import sys
import os

def spec_abund(slitmask_name=None, slit_str=None, slit_num=None, feh_def = -0.75, alphafe_def = 0.,
dlam=1.2, synth_path='', slitmask_path='', moogify_path='',
mask_ranges=None, flex_factor=400., mask_path='', save_path='', replace=False,
grating='', cont_mask=None, spec1dfile=True, save_filename='',
telluric_path='', object_id=None, wgood=False, m31disk=None):

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
    dlam: float, optional: Gaussian sigma to assume in smoothing of synthetic
          spectrum to DEIMOS resolution, where the Gaussian sigma is equal to the
          FWHM in Angstroms divided by approximately 2.35. For 1200G and 0.7"" slit,
          assume dlam_def = 0.45.
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
    save_filename: string, optional: the filename to save the results under. If not specified,
                    the filename will be constructed based on the slitmask name and slit number.
    replace: boolean, optional: if True, replace pre-existing output files
    grating: string, optional: grating used to collect slitmask data, relevant for
             specifying which wavelength masks to use
    cont_mask: 2D array-like, optional: an additional mask to use in the initial continuum
                normalization, if neccessary
    spec1dfile: boolean, optional: if True, then the wavelength, flux, and inverse variance
                arrays are read directly from the spec1d files. If False, then the arrays
                are read from the input dictionary. In the latter case, the arrays have
                been flattened and tweaks to the wavelength solution have been applied.
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
    if slitmask_name is not None:

        if slit_num is not None:
            slit_str = str(slit_num)
            if len(slit_str) == 1: slit_str = '00'+slit_str
            elif len(slit_str) == 2: slit_str = '0'+slit_str
            else: pass

        if slit_str is not None:
            slit_num = int(slit_str)

        if not replace:
            if slit_str is not None:
                filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+slit_str+'_dlamfix.pkl'
            else:
                 filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+object_id+'_dlamfix.pkl'
            if os.path.exists(filename): return
            else: pass

        if spec1dfile:

            #Find filenames (for a typical slitmask)
            if slit_str is not None:
                obs_file = glob.glob(slitmask_path+slitmask_name+'/spec1d.'+slitmask_name+'*.'+slit_str+'.*.fits.gz')[0]
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
            w = np.where(moog_data['objname'] == object_id)[0][0]

        if not spec1dfile:

            xspec = moog_data['airmass'][w]
            wavew = moog_data['lambda'][w]
            fluxw= moog_data['spec'][w]
            ivarw = moog_data['ivar'][w]

            wave, flux, ivar = data_redux.find_chip_gap_spec1d(wavew, fluxw, ivarw)

        def separate(arr):
            nhalf = int(round(float(len(arr))/2.))
            wchip1 = np.array(list(range(nhalf)))
            wchip2 = wchip1 + len(wchip1)
            arr2d = np.array([arr[wchip1], arr[wchip2]])
            return arr2d

        #Perform the telluric absorption correction
        flux_tell, ivar_tell = data_redux.telluric_correct(wave, flux, ivar, xspec,
        slitmask_path=slitmask_path, grating=grating, telluric_path=telluric_path)

        #Identify the redshift of the object and shift the observed spectrum to the rest frame
        zrest = moog_data['zrest'][w]
        wave_rest = wave/(1. + zrest)

        #Load in the photometric data of the object for the chi-squared fitting

        #deprecated
        #teffphot, teffphoterr, loggphot, loggphoterr = data_redux.get_photometry(moog_data, w,
        #phot_path=phot_path, phot_type=phot_type)

        teffphot, teffphoterr = moog_data['teffphot'][w], moog_data['teffphoterr'][w]
        loggphot, loggphoterr = moog_data['loggphot'][w], moog_data['loggphoterr'][w]

        print(teffphot, loggphot)

        if (teffphot < 0) or (loggphot < 0) or np.isnan(teffphot) or np.isnan(loggphot):
            if slit_str is not None:
                sys.stderr.write('Nonsense photometry: skipping '+slitmask_name+' slit number'+slit_str+'\n')
                return
            else:
                sys.stderr.write('Nonsense photometry: skipping '+slitmask_name+' object ID '+object_id+'\n')
            return

    if m31disk is not None:

        if not replace:
            filename = save_path+'m31disk/specabund_m31disk_'+m31disk['slitmask'][slit_num]+'_'+\
                        m31disk['slit'][slit_num]+'.pkl'
            if os.path.exists(filename): return

        spec0 = m31disk['spec'][slit_num]
        ivar0 = m31disk['ivar'][slit_num]
        wvl0 = m31disk['lambda'][slit_num]
        wchip = m31disk['wchip'][slit_num]

        if ~np.isnan(wchip).any():

            wave = np.array([wvl0[int(wchip[0]):int(wchip[1])], wvl0[int(wchip[2]):int(wchip[3])]],dtype='object')
            flux = np.array([spec0[int(wchip[0]):int(wchip[1])], spec0[int(wchip[2]):int(wchip[3])]],dtype='object')
            ivar = np.array([ivar0[int(wchip[0]):int(wchip[1])], ivar0[int(wchip[2]):int(wchip[3])]],dtype='object')

        else:

            wb = wchip[:2]
            wr = wchip[2:]

            if ~np.isnan(wb).all():

                wave = np.array([wvl0[int(wchip[0]):int(wchip[1])]],dtype='object')
                flux = np.array([spec0[int(wchip[0]):int(wchip[1])]],dtype='object')
                ivar = np.array([ivar0[int(wchip[0]):int(wchip[1])]],dtype='object')

            if ~np.isnan(wr).all():

                wave = np.array([wvl0[int(wchip[2]):int(wchip[3])]],dtype='object')
                flux = np.array([spec0[int(wchip[2]):int(wchip[3])]],dtype='object')
                ivar = np.array([ivar0[int(wchip[2]):int(wchip[3])]],dtype='object')

            if np.isnan(wchip).all():
                return

        xspec = m31disk['airmass'][slit_num]
        zrest = 0.

        grating = m31disk['grating'][slit_num]
        if grating == 600: grating = '600ZD'
        if grating == 1200: grating = '1200G'

        #Set the resolution according to the grating
        if grating == '600ZD': dlam = 1.2
        if grating == '1200': dlam = 0.45

        flux_tell_flat, ivar_tell_flat = data_redux.telluric_correct(wave, flux, ivar, xspec,
            grating=grating, telluric_path=telluric_path, m31disk=True)

        flux_tell = np.array( [ flux_tell_flat[:len(flux[0])], flux_tell_flat[len(flux[0]):] ],dtype='object' )
        ivar_tell = np.array( [ ivar_tell_flat[:len(ivar[0])], ivar_tell_flat[len(ivar[0]):] ],dtype='object' )

        wave_rest = wave

        #Assuming 4 Gyr isochrones for the disk
        teffphot = m31disk['teffphot'][slit_num]; loggphot = m31disk['loggphot'][slit_num]
        teffphoterr = m31disk['teffphoterr'][slit_num]; loggphoterr = m31disk['loggphoterr'][slit_num]

        if np.isnan(teffphoterr): teffphoterr = 100.
        if np.isnan(loggphoterr): loggphoterr = 0.1
        print(teffphot, teffphoterr, loggphot, loggphoterr)

    #Construct the mask for the continuum normalization
    cont_regions = data_redux.check_pkl_file_exists('mask', file_path=mask_path+'contregion/'+grating+'/')

    #Perform the initial continuum normalization
    norm_result = data_redux.cont_norm(wave_rest, flux_tell, ivar_tell, zrest=zrest,
    cont_regions=cont_regions, grating=grating, cont_mask=cont_mask)

    if norm_result == None: return
    flux_norm, ivar_norm, norm, contmask, cont_flag = norm_result

    #Flatten the arrays
    def flatten(arr):
        return np.array(arr[0].tolist() + arr[1].tolist())

    wave_flat = flatten(wave_rest)
    flux_flat = flatten(flux_norm)
    ivar_flat = flatten(ivar_norm)

    #Calculate an approximation of the signal-to-noise for the 1200G grating
    if grating == '1200G':
        sn = data_redux.calc_sn(wave_flat, flatten(flux_tell), grating=grating,
        cont_regions=cont_regions, continuum=flatten(norm), zrest=zrest)

    ## Construct the wavelength masks ##

    #Now construct a mask based on regions to fit for [Fe/H]
    """
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

    if grating == '1200G':

        feh_mask = data_redux.check_pkl_file_exists('mask_fe_'+grating+'_final',
        file_path=mask_path+'specregion/red/')

        alphafe_mask = data_redux.check_pkl_file_exists('mask_alphafe_'+grating+'_final',
        file_path=mask_path+'specregion/red/')
    """

    #feh_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    #mask_ranges=mask_ranges, fit_ranges=feh_mask, zrest=zrest)

    #alphafe_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    #mask_ranges=mask_ranges, fit_ranges=alphafe_mask, zrest=zrest)

    #Construct a general mask for the spectrum
    spec_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, zrest=zrest)

    alphafe_fit_mask = spec_mask
    feh_fit_mask = spec_mask

    #Identify whether there is wavelength information in the blue and/or red
    #synthetic spectra ranges
    #wb = np.where((wave_flat[spec_mask] >= 4100.)&(wave_flat[spec_mask] < 6300.))[0]
    #wr = np.where((wave_flat[spec_mask] >= 6300.)&(wave_flat[spec_mask] < 9100.))[0]

    #Initialize a hash table to use to store the synthetic spectral data in memory
    hash = {}

    def construct_synth(wvl, teff_in, feh_in, alphafe_in, hash=hash):
        """
        Helper function for the get_synth() functions
        """

        wvl_synth, relflux_synth = read_synth.read_interp_synth(teff=teff_in, logg=loggphot,
        feh=feh_in, alphafe=alphafe_in, data_path=synth_path, hash=hash)

        #Make sure that the synthetic spectrum is within the data range for observations,
        #given the use of different gratings and central wavelengths
        wsynth = np.where((wvl_synth > wvl[0])&(wvl_synth < wvl[-1]))[0]

        return wvl_synth[wsynth], relflux_synth[wsynth]

    #When an overall metallicity change occurs, repeat the refinement until the value
    #does not change significantly -- define significantly as within 0.1 dex
    #Based on Shetrone et al. 2009

    #Set the initial parameters
    phot_dict = {'teffphot': teffphot, 'loggphot': loggphot, 'teffphoterr': teffphoterr,
    'loggphoterr': loggphoterr, 'flex_factor': flex_factor}

    if teffphot > 6500.: teffphot = 6500.
    if teffphot < 3000: teffphot = 3000.

    #if teffphot < 4100.: fehmin = -4.5
    #else: fehmin = -5.

    maxiter = 50
    feh_thresh = 0.001; alphafe_thresh = 0.001; teff_thresh = 1.
    feh0 = feh_def; alphafe0 = alphafe_def; teff0 = teffphot

    i = 0; converge_flag = 0
    while i < maxiter:

        def get_synth_step1(wvl, teff_fit, feh_fit):

            """
            Define the function to be used in the curve fitting, such that
            ydata = f(xdata, *params) + eps and r = ydata - f(xdata, *params),
            where f is the model function

            In this iteration, the effective temperature is determined within the constraints
            of photometry and the metallicity is varied. [alpha/Fe] is fixed constant at solar

            Parameters
            ----------
            wvl: array-like: the flattened wavelength array of the observed spectrum
            teff_fit: float: spectroscopic effective temperature, to be fitted
            feh_fit: float: spectroscopic metallicity, to be fitted

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
            hash=hash)

            #Make sure that the synthetic spectrum is within the data range for observations,
            #given the use of different gratings and central wavelengths
            wsynth = np.where((wvl_synth > wvl[0])&(wvl_synth < wvl[-1]))[0]

            wvl_synth = wvl_synth[wsynth]
            relflux_synth = relflux_synth[wsynth]

            #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array

            if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
                relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
                wvl[1:], dlam[feh_fit_mask])
            else:
                relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
                wvl[1:], dlam)

            #Insert the effective temperature pixel to the beginning of the synthetic spectrum
            flux_synth_teff = np.insert(relflux_interp_smooth, 0, teff_fit)

            return flux_synth_teff

        #Perform the fit for [Fe/H], Dlam, Teff

        #Insert the effective temperature pixel at the beginning of the observed spectrum
        #and the inverse variance array
        params0 = [teff0, feh0]

        flux_obs_teff = np.insert(flux_flat[feh_fit_mask], 0, teffphot)
        wave_teff = np.insert(wave_flat[feh_fit_mask], 0, wave_flat[feh_fit_mask][0])

        sigma_teff0 = ivar_flat[feh_fit_mask]**(-0.5)
        npix_fit = float(len(sigma_teff0))

        sigma_teff = np.insert(sigma_teff0, 0, teffphoterr * np.sqrt(flex_factor/npix_fit))
        #sigma_teff = np.insert(sigma_teff0, 0, teffphoterr)

        best_params0, covar0 = curve_fit(get_synth_step1, wave_teff, flux_obs_teff, p0=params0,
        sigma=sigma_teff, bounds=([3000., -2.5],[6500., 1.]), absolute_sigma=True,
        ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

        #Load in the best fit synthetic spectrum
        teff_best0, feh_best0 = best_params0

        def get_synth_step2(wvl, alphafe_fit):
            """
            In this iteration, the effective temperature, metallicity, and smoothing parameter
            are held constant at the values from step1, and the alpha abundance is varied.
            """

            #Read in the synthetic spectrum
            wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best0, alphafe_fit,
            hash=hash)

            #Smooth the synthetic spectrum according to the spectral resolution
            if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
                relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
                wvl, dlam[alphafe_fit_mask])
            else:
                relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
                wvl, dlam)

            return relflux_interp_smooth

        #Perform the fit [alpha/Fe]
        params1 = [alphafe0]
        sigma_flat = ivar_flat[alphafe_fit_mask]**(-0.5)

        best_params1, covar1 = curve_fit(get_synth_step2, wave_flat[alphafe_fit_mask],
        flux_flat[alphafe_fit_mask], p0=params1, sigma=sigma_flat, bounds=([-1.], [1.]),
        absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

        def get_best_synth(wvl, feh_best, alphafe_best, dlam_mask):

            #Read in the synthetic spectrum
            wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best, alphafe_best,
            hash=hash)

            #Smooth the synthetic spectrum according to the spectral resolution
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam_mask)

            return relflux_interp_smooth

        alphafe_best1 = best_params1[0]
        best_synth1 = get_best_synth(wave_flat, feh_best0, alphafe_best1, dlam)

        #Refine the continuum normalization
        flux_refine, ivar_refine, refinedcont = data_redux.cont_refine(wave_rest,
        flux_norm, ivar_norm, wave_flat, best_synth1, cont_mask=cont_mask, norm=None,
        grating=grating)

        ##Dump some of the memory, retain info from last ~2 calls to cub
        if hash is not None:
            keys = np.asarray(list(hash.keys()))
            keylim = (4**4)*2 #base**ndim x 2
            if len(keys) > keylim:
                for key in keys[:len(keys)-keylim]: del hash[key]

        if (np.abs(feh_best0 - feh0) < feh_thresh) and (np.abs(alphafe_best1 - alphafe0) <\
        alphafe_thresh) and (np.abs(teff_best0 - teff0) < teff_thresh):
            print('Continuum iteration converged')
            converge_flag = 1
            break
        else:
            print(teff_best0, feh_best0, alphafe_best1)
            feh0 = feh_best0; alphafe0 = alphafe_best1; teff0 = teff_best0
            flux_flat = flux_refine; ivar_flat = ivar_refine
            i += 1

    #### Perform the fit again with the revised spectrum ####
    def get_synth_step3(wvl, feh_fit):
        """
        In this iteration, the effective temperature is held constant at the values from step1,
        the alpha abundance is assumed to be solar, and the metallicity is re-determined
        based on the revised observed spectrum
        """

        #Read in the synthetic spectrum
        wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_fit, alphafe_best1,
        hash=hash)

        #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
        if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam[feh_fit_mask])
        else:
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

        return relflux_interp_smooth

    #Calculate the chi-squared value
    chi0 = data_redux.compare_spectra(flux_obs_teff, get_synth_step1(wave_teff,
    teff_best0, feh_best0), stat_name='chisqr', weights=sigma_teff**(-0.5),
    nvarys=len(params0))

    params2 = [feh0]

    best_params2, covar2 = curve_fit(get_synth_step3, wave_flat[feh_fit_mask],
    flux_refine[feh_fit_mask], p0=params2, sigma=ivar_refine[feh_fit_mask]**(-0.5),
    bounds=([-2.5],[1.]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

    feh_best2 = best_params2[0]

    #### Now re-determine the total alpha abundance of the atmosphere ####
    def get_synth_step4(wvl, alphafe_fit):
        """
        In this iteration, the effective temperature is held constant at the value from step1,
        the metallicity is held constant at the previously determined value, and the [alpha/Fe]
        is re-determined based on the revised observed spectrum
        """
        #Read in the synthetic spectrum
        wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best2, alphafe_fit,
        hash=hash)

        #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
        if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam[alphafe_fit_mask])
        else:
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

        return relflux_interp_smooth

    params3 = [alphafe0]
    best_params3, covar3 = curve_fit(get_synth_step4, wave_flat[alphafe_fit_mask],
    flux_refine[alphafe_fit_mask], p0=params3, sigma=ivar_refine[alphafe_fit_mask]**(-0.5),
    bounds=([-1.],[1.]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

    alphafe_best3 = best_params3[0]
    alphafe_best_final = alphafe_best3

    #Calculate the chi-squared statistic
    chi3 = data_redux.compare_spectra(flux_refine[alphafe_fit_mask],
    get_synth_step4(wave_flat[alphafe_fit_mask], alphafe_best3), stat_name='chisqr',
    weights=ivar_refine[alphafe_fit_mask], nvarys=len(params3))

    #### Recalculate the metallicity a final time ####
    def get_synth_step5(wvl, feh_fit):

        #Read in the synthetic spectrum
        wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_fit, alphafe_best3,
        hash=hash)

        #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
        if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam[feh_fit_mask])
        else:
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

        return relflux_interp_smooth

    params4 = [feh0]

    best_params4, covar4 = curve_fit(get_synth_step5, wave_flat[feh_fit_mask],
    flux_refine[feh_fit_mask], p0=params4, sigma=ivar_refine[feh_fit_mask]**(-0.5),
    bounds=([-2.5],[1.]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

    feh_best4 = best_params4[0]
    feh_best_final = feh_best4

    #Calculate the chi-squared value
    chi4 = data_redux.compare_spectra(flux_refine[feh_fit_mask],
    get_synth_step5(wave_flat[feh_fit_mask], feh_best4), stat_name='chisqr',
    weights=ivar_refine[feh_fit_mask], nvarys=len(params4))

    best_synth = get_best_synth(wave_flat, feh_best_final, alphafe_best_final, dlam)

    #### Additional steps to measure [Mg/Fe] and [Ca/Fe] ####
    #### Performed with the final values of [Fe/H] and Dlam ####
    """
    def get_synth_mg(wvl, alphafe_fit):

        #Read in the synthetic spectrum
        wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best4, alphafe_fit,
        hash_red=hash_red, hash_blue=hash_blue)

        #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
        if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam[mgfe_fit_mask])
        else:
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

        return relflux_interp_smooth

    params_mg = [alphafe_def]
    best_params_mg, covar_mg = curve_fit(get_synth_mg, wave_flat[mgfe_fit_mask],
    flux_refine[mgfe_fit_mask], p0=params_mg, sigma=ivar_refine[mgfe_fit_mask]**(-0.5),
    bounds=([-0.8], [1.2]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)
    mgfe_best = best_params_mg[0]

    #Calculate the chi-squared for fitting Mg
    chi_mg = data_redux.compare_spectra(flux_refine[mgfe_fit_mask],
    get_synth_mg(wave_flat[mgfe_fit_mask], mgfe_best), stat_name='chisqr',
    weights=ivar_refine[mgfe_fit_mask], nvarys=len(params_mg))

    def get_synth_ca(wvl, alphafe_fit):

        #Read in the synthetic spectrum
        wvl_synth, relflux_synth = construct_synth(wvl, teff_best0, feh_best4, alphafe_fit,
        hash_red=hash_red, hash_blue=hash_blue)

        #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
        if isinstance(dlam, list) or isinstance(dlam, np.ndarray):
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam[cafe_fit_mask])
        else:
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

        return relflux_interp_smooth

    params_ca = [alphafe_def]
    best_params_ca, covar_ca = curve_fit(get_synth_ca, wave_flat[cafe_fit_mask],
    flux_refine[cafe_fit_mask], p0=params_ca, sigma=ivar_refine[cafe_fit_mask]**(-0.5),
    bounds=([-0.8], [1.2]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)
    cafe_best = best_params_ca[0]

    #Calculate the chi-squared for fitting Ca
    chi_ca = data_redux.compare_spectra(flux_refine[cafe_fit_mask],
    get_synth_ca(wave_flat[cafe_fit_mask], cafe_best), stat_name='chisqr',
    weights=ivar_refine[cafe_fit_mask], nvarys=len(params_ca))
    """

    #Final best fit parameters
    best_params = [teff_best0, feh_best_final, alphafe_best_final]
    print(best_params)

    #Unreduced parameter errors based on the covariance matrix returned by the fit
    #(Statistical uncertainty)

    params_err = np.sqrt([np.diag(covar0)[0], np.diag(covar4)[0], np.diag(covar3)[0]])
                              #np.diag(covar_mg)[0], np.diag(covar_ca)[0]])

    #Now perform a check of the uncertainties returned by the fit by analyzing the full
    #chi-squared distribution about the minimum
    params_dict = {'teff': best_params[0], 'logg': loggphot, 'feh': best_params[1],
    'alphafe': best_params[2], 'dlam': dlam}

    perr_teff = {'teff': params_err[0], 'logg': 0., 'feh': 0., 'alphafe': 0., 'dlam':0.}
    perr_feh = {'teff': 0., 'logg': 0., 'feh': params_err[1], 'alphafe': 0., 'dlam': 0.}
    perr_alphafe = {'teff': 0., 'logg': 0., 'feh': 0., 'alphafe': params_err[2], 'dlam': 0.}

    xteff, teffcontour = data_redux.check_fit(params_dict, perr_teff, wave_flat, flux_refine,
        ivar_refine, mask=feh_fit_mask, hash=hash, phot_dict=phot_dict, synth_path=synth_path,
        dlamfix=True)

    xfeh, fehcontour = data_redux.check_fit(params_dict, perr_feh, wave_flat, flux_refine,
        ivar_refine, mask=feh_fit_mask, hash=hash, phot_dict=phot_dict, synth_path=synth_path,
        dlamfix=True)

    xalphafe, alphafecontour = data_redux.check_fit(params_dict, perr_alphafe, wave_flat,
        flux_refine, ivar_refine, mask=alphafe_fit_mask, hash=hash, phot_dict=phot_dict,
        synth_path=synth_path, dlamfix=True)

    #hash = {}
    del hash

    #Now scale the statistical uncertainties by the reduced chi-squared value to construct
    #the fit uncertainty

    nvarys = 3
    chiteff = chi0/float(len(flux_refine[feh_fit_mask])-1-nvarys)
    #chimg = chi_mg/float(len(flux_refine[mgfe_fit_mask]))
    #chica = chi_ca/float(len(flux_refine[cafe_fit_mask]))

    chifeh = chi4/float(len(flux_refine[feh_fit_mask])-nvarys)
    chialpha = chi3/float(len(flux_refine[alphafe_fit_mask])-nvarys)

    params_err *= np.sqrt( np.array( [chiteff, chifeh, chialpha]) )
    #chimg, chica]

    #Calculate the signal-to-noise ratio for the 600ZD grating
    if grating == '600ZD':
        sn = data_redux.calc_sn(wave_flat, flux_refine, grating=grating,
        continuum=best_synth, zrest=zrest, contmask=flatten(contmask))

    #Save the data
    specabund = params_dict.copy()
    specabund.update({'tefferr': params_err[0], 'feherr': params_err[1],
    'alphafeerr': params_err[2], 'chifeh': chifeh, 'teffphot': teffphot,
    'teffphoterr': teffphoterr, 'fehnpix': len(flux_refine[feh_fit_mask]),
    'alphafenpix': len(flux_refine[alphafe_fit_mask]), 'npix': len(flux_refine[spec_mask]),
    'chialpha': chialpha, 'teffcontour': teffcontour, 'fehcontour': fehcontour,
    'alphafecontour': alphafecontour, 'wvl': wave_flat, 'flux': flux_refine,
    'ivar': ivar_refine, 'synth': best_synth, 'zrest': zrest,
    'teffcontourx': xteff, 'fehcontourx': xfeh, 'alphafecontourx': xalphafe,
    'chiteff': chiteff, 'teffcontourx': xteff,
    'teffnpix': len(flux_refine[feh_fit_mask])-1, 'converge_flag': converge_flag,
    'contspec': flux_flat, 'contivar': ivar_flat, 'continuum': flatten(norm),
    'contmask': flatten(contmask), 'cont_flag': cont_flag, 'refinedcont': refinedcont})

    if slitmask_name is not None:
        specabund.update({'spec': flatten(flux), 'ivarspec': flatten(ivar),
        'tellspec': flatten(flux_tell), 'tellivar': flatten(ivar_tell), 'sn': sn,
        'objname': moog_data['objname'][w], 'spec1dfile': obs_file})

    if m31disk is not None:
        specabund.update({'spec': m31disk['spec'][slit_num],
                          'ivarspec': m31disk['ivar'][slit_num],
                          'tellspec': flatten(flux_tell),
                          'tellivar': flatten(ivar_tell), 'sn': sn,
                          'objname': m31disk['objname'][slit_num],
                          'slitmask': m31disk['slitmask'][slit_num],
                          'slit': m31disk['slit'][slit_num]})

    if slitmask_name is not None:
        if not os.path.exists(save_path+slitmask_name): os.makedirs(save_path+slitmask_name)
        if len(save_filename) == 0:
            if slit_str is not None:
                save_filename = 'specabund_'+slitmask_name+'.'+slit_str+'_dlamfix'
            else:
                save_filename = 'specabund_'+slitmask_name+'.'+object_id+'_dlamfix'
        with open(save_path+slitmask_name+'/'+save_filename+'.pkl', 'wb') as f:
            pickle.dump(specabund, f)

    if m31disk is not None:
        if not os.path.exists(save_path+'m31disk'): os.makedirs(save_path+'m31disk')
        save_filename = 'specabund_m31disk_'+m31disk['slitmask'][slit_num]+'_'+m31disk['slit'][slit_num]
        with open(save_path+'m31disk/'+save_filename+'.pkl', 'wb') as f:
            pickle.dump(specabund, f)

    return specabund
