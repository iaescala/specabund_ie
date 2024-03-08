"""
@author: Ivanna Escala 2018-2022
"""

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.io import fits
import numpy as np
import read_synth_ts as read_synth
import data_redux_ts_coadd as data_redux
import matplotlib.pyplot as plt
import pickle
import scipy
import glob
import sys
import os

def spec_abund_coadd(slitmask_name=None, slit_list_str=None, feh_def = -0.75, alphafe_def = 0.,
dlam=1.2, synth_path='', slitmask_path='', moogify_path='',
mask_ranges=None, mask_path='', save_path='', replace=False,
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
          assume dlam = 0.45.
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

        #construct filename for saving
        fname_slit_str = ''
        for item in slit_list_str:
            fname_slit_str += item; fname_slit_str += '.'

        #check if file already exists
        if not replace:
            filename = save_path+slitmask_name+'/specabund_'+slitmask_name+'.'+fname_slit_str+'pkl'
            if os.path.exists(filename): return
            else: pass

        wave_split_list = []; flux_split_list = []; ivar_split_list = []; norm_split_list = []
        teffphot_list = []; teffphoterr_list = []; loggphot_list = []; loggphoterr_list = []
        wave_list = []; flux_list = []; ivar_list = []; norm_list = []

        #get and process every spectrum in list of slits
        for slit_str in slit_list_str:
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

            if (teffphot < 0) or (loggphot < 0) or np.isnan(teffphot) or np.isnan(loggphot):
                if slit_str is not None:
                    sys.stderr.write('Nonsense photometry: skipping '+slitmask_name+' slit number'+slit_str+'\n')
                    return
                else:
                    sys.stderr.write('Nonsense photometry: skipping '+slitmask_name+' object ID '+object_id+'\n')
                return

            teffphot_list.append(teffphot); teffphoterr_list.append(teffphoterr)
            loggphot_list.append(loggphot);loggphoterr_list.append(loggphoterr)

            #Construct the mask for the continuum normalization
            cont_regions = data_redux.check_pkl_file_exists('mask', file_path=mask_path+'contregion/'+grating+'/')

            #Perform the initial continuum normalization
            norm_result = data_redux.cont_norm(wave_rest, flux_tell, ivar_tell, zrest=zrest,
            cont_regions=cont_regions, grating=grating, cont_mask=cont_mask)

            if norm_result == None: return
            flux_norm, ivar_norm, norm, contmask, cont_flag = norm_result

            #norm = separate(moog_data['continuum'][w])
            #ivar_norm = separate(moog_data['contdivivar'][w])
            #flux_norm = separate(moog_data['contdiv'][w])
            #cont_flag = [1, 1]
            #contmask = np.ones(np.shape(wave_rest))

            #Flatten the arrays
            def flatten(arr):
                return np.array(arr[0].tolist() + arr[1].tolist())

            wave_flat = flatten(wave_rest)
            flux_flat = flatten(flux_norm)
            ivar_flat = flatten(ivar_norm)
            norm_flat = flatten(norm)

             #Calculate an approximation of the signal-to-noise for the 1200G grating
            # if grating == '1200':
            #     sn = data_redux.calc_sn(wave_flat, flatten(flux_tell), grating=grating,
            #     cont_regions=cont_regions, continuum=flatten(norm), zrest=zrest)


            #modifications for the coadds
            if grating == '1200G':
                wavelengths = np.arange(6300.0,9000.0,0.32)
                wavelengths_split = [wavelengths[:4219],wavelengths[4219:]]

            if grating == '600ZD':
                wavelengths = np.arange(4100.0,9100.0,0.64)
                wavelengths_split = [wavelengths[:3906],wavelengths[3906:]]

            flux_split, ivar_split, norm_split = [],[],[]
            f = interp1d(wave_flat,flux_flat,fill_value=0.0,bounds_error=False)
            flux_split.append(f(wavelengths_split[0]))
            flux_split.append(f(wavelengths_split[1]))

            f = interp1d(wave_flat,ivar_flat,fill_value=0.0,bounds_error=False)
            ivar_split.append(f(wavelengths_split[0]))
            ivar_split.append(f(wavelengths_split[1]))

            f = interp1d(wave_flat,norm_flat,fill_value=0.0,bounds_error=False)
            norm_split.append(f(wavelengths_split[0]))
            norm_split.append(f(wavelengths_split[1]))

            #save a version of these split arrays for continuum refinement?
            wave_split_list.append(wavelengths_split);flux_split_list.append(flux_split);ivar_split_list.append(ivar_split)
            norm_split_list.append(norm_split)

    if m31disk is not None:

        #construct filename for saving
        fname_slit_str = ''
        for item in slit_list_str:
            fname_slit_str += item; fname_slit_str += '.'

        #check if file already exists
        if not replace:
            filename = save_path+'m31disk/coadds/specabund_m31disk.'+fname_slit_str+'pkl'
            if os.path.exists(filename): return
            else: pass

        wave_split_list = []; flux_split_list = []; ivar_split_list = []; norm_split_list = []
        teffphot_list = []; teffphoterr_list = []; loggphot_list = []; loggphoterr_list = []
        wave_list = []; flux_list = []; ivar_list = []; norm_list = []

        for slit_str in slit_list_str:

            wobj = np.where(m31disk['objname'] == slit_str)[0][0]

            flux0 = m31disk['spec'][wobj]
            ivar0 = m31disk['ivar'][wobj]
            wave0 = m31disk['lambda'][wobj]
            wchip = m31disk['wchip'][wobj]

            if ~np.isnan(wchip).any():

                wave = np.array([wave0[int(wchip[0]):int(wchip[1])], wave0[int(wchip[2]):int(wchip[3])]])
                flux = np.array([flux0[int(wchip[0]):int(wchip[1])], flux0[int(wchip[2]):int(wchip[3])]])
                ivar = np.array([ivar0[int(wchip[0]):int(wchip[1])], ivar0[int(wchip[2]):int(wchip[3])]])

            else:

                wb = wchip[:2]
                wr = wchip[2:]

                if ~np.isnan(wb).all():

                    wave = np.array([wave0[int(wchip[0]):int(wchip[1])]])
                    flux = np.array([flux0[int(wchip[0]):int(wchip[1])]])
                    ivar = np.array([ivar0[int(wchip[0]):int(wchip[1])]])

                if ~np.isnan(wr).all():

                    wave = np.array([wave0[int(wchip[2]):int(wchip[3])]])
                    flux = np.array([flux0[int(wchip[2]):int(wchip[3])]])
                    ivar = np.array([ivar0[int(wchip[2]):int(wchip[3])]])

                if np.isnan(wchip).all():
                    return

            xspec = m31disk['airmass'][wobj]
            zrest = 0.

            grating = m31disk['grating'][wobj]
            if grating == 600: grating = '600ZD'
            if grating == 1200: grating = '1200G'

            #Set the resolution according to the grating
            if grating == '600ZD': dlam = 1.2
            if grating == '1200': dlam = 0.45

            flux_tell_flat, ivar_tell_flat = data_redux.telluric_correct(wave, flux, ivar, xspec,
                grating=grating, telluric_path=telluric_path, m31disk=True)

            flux_tell = np.array( [ flux_tell_flat[:len(flux[0])], flux_tell_flat[len(flux[0]):] ] )
            ivar_tell = np.array( [ ivar_tell_flat[:len(ivar[0])], ivar_tell_flat[len(ivar[0]):] ] )

            wave_rest = wave

            #Assuming 4 Gyr isochrones for the disk
            teffphot = m31disk['teffphot'][wobj]; loggphot = m31disk['loggphot'][wobj]
            teffphoterr = m31disk['teffphoterr'][wobj]; loggphoterr = m31disk['loggphoterr'][wobj]

            teffphot_list.append(teffphot); loggphot_list.append(loggphot)
            teffphoterr_list.append(teffphoterr); loggphoterr_list.append(loggphoterr)

            #Construct the mask for the continuum normalization
            cont_regions = data_redux.check_pkl_file_exists('mask', file_path=mask_path+'contregion/'+grating+'/')

            #Perform the initial continuum normalization
            norm_result = data_redux.cont_norm(wave_rest, flux_tell, ivar_tell, zrest=zrest,
            cont_regions=cont_regions, grating=grating, cont_mask=cont_mask)

            if norm_result == None: return
            flux_norm, ivar_norm, norm, contmask, cont_flag = norm_result

            #plt.figure()
            #plt.plot(wave[0], flux[0])
            #plt.plot(wave[1], flux[1])
            #plt.plot(wave[0], norm[0])
            #plt.plot(wave[1], norm[1])
            #plt.ylim(np.nanmedian(flux[1])-200, np.nanmedian(flux[1])+200)
            #plt.show()

            #Flatten the arrays
            def flatten(arr):
                return np.array(arr[0].tolist() + arr[1].tolist())

            wave_flat = flatten(wave_rest)
            flux_flat = flatten(flux_norm)
            ivar_flat = flatten(ivar_norm)
            norm_flat = flatten(norm)

            #modifications for the coadds
            if grating == '1200G':
                wavelengths = np.arange(6300.0,9000.0,0.32)
                wavelengths_split = [wavelengths[:4219],wavelengths[4219:]]

            if grating == '600ZD':
                wavelengths = np.arange(4100.0,9100.0,0.64)
                wavelengths_split = [wavelengths[:3906],wavelengths[3906:]]

            flux_split, ivar_split, norm_split = [],[],[]
            f = interp1d(wave_flat,flux_flat,fill_value=0.0,bounds_error=False)
            flux_split.append(f(wavelengths_split[0]))
            flux_split.append(f(wavelengths_split[1]))

            f = interp1d(wave_flat,ivar_flat,fill_value=0.0,bounds_error=False)
            ivar_split.append(f(wavelengths_split[0]))
            ivar_split.append(f(wavelengths_split[1]))

            f = interp1d(wave_flat,norm_flat,fill_value=0.0,bounds_error=False)
            norm_split.append(f(wavelengths_split[0]))
            norm_split.append(f(wavelengths_split[1]))

            #save a version of these split arrays for continuum refinement?
            wave_split_list.append(wavelengths_split);flux_split_list.append(flux_split);ivar_split_list.append(ivar_split)
            norm_split_list.append(norm_split)

    def sig_clip(ivar_list_input,sigma_pass1=10.,sigma_pass2=3.5):

        sum_masked_1 = 0;sum_masked_2 = 0
        #Do separately for the blue and red ends of the CCD
        for i in range(2):

            ivar_list_i = np.array([_ivar[i] for _ivar in ivar_list_input])


            for j in range(len(ivar_list_i[i])):

                #first sigma clip
                sigma = sigma_pass1
                pixel_values = np.array([item[j] for item in ivar_list_i])

                nan_idx = np.isnan(pixel_values)
                pixel_values[nan_idx] = 0.0

                if len(np.where(nan_idx == False)[0]) != 1:
                    deviation = pixel_values - np.median(pixel_values)

                    mask_ids = np.where((deviation < (-sigma*np.std(deviation))) |
                                     (deviation > (sigma*np.std(deviation))))[0]

                else:
                    mask_ids = np.arange(len(pixel_values))

                sum_masked_1 += len(mask_ids)

                for idx in mask_ids:
                    ivar_list_i[idx][j] = 0.0

                #second clip
                sigma = sigma_pass2
                pixel_values = np.array([item[j] for item in ivar_list_i])

                nan_idx = np.isnan(pixel_values)
                pixel_values[nan_idx] = 0.0

                if len(np.where(nan_idx == False)[0]) != 1:
                    deviation = pixel_values - np.median(pixel_values)

                    mask_ids = np.where((deviation < (-sigma*np.std(deviation))) |
                                     (deviation > (sigma*np.std(deviation))))[0]

                else:
                    mask_ids = np.arange(len(pixel_values))

                sum_masked_2 += len(mask_ids)

                for idx in mask_ids:
                    ivar_list_i[idx][j] = 0.0

            if i == 0:
                ivar_list_b = ivar_list_i
            else:
                ivar_list_r = ivar_list_i

        print('# masked pixels (first pass): '+str(sum_masked_1))
        print('# masked pixels (second pass): '+str(sum_masked_2))
        ivar_list_clipped = [[ivar_list_b[i],ivar_list_r[i]] for i in range(len(ivar_list_input))]

        return ivar_list_clipped

    ivar_split_list_clipped = sig_clip(ivar_split_list,sigma_pass1=10.0,sigma_pass2=3.5)

    wave_list = []; flux_list = []; ivar_list = []; norm_list = []

    for i in range(len(slit_list_str)):

        flux_flat = flatten(flux_split_list[i])
        ivar_flat = flatten(ivar_split_list_clipped[i])
        #ivar_flat = flatten(ivar_split_list[i])
        norm_flat = flatten(norm_split_list[i])

        wave_list.append(wavelengths);flux_list.append(flux_flat);ivar_list.append(ivar_flat)

        norm_list.append(norm_flat)

    ## Coadd all spectra in the list ##
    flux_stack_flat = np.nansum(np.array(flux_list)*np.array(ivar_list),axis=0)/np.nansum(np.array(ivar_list),axis=0)
    ivar_stack_flat = np.nansum(np.array(ivar_list),axis=0)
    norm_stack_flat = np.nansum(np.array(norm_list)*np.array(ivar_list),axis=0)/np.nansum(np.array(ivar_list),axis=0)

    wave = wavelengths_split

    if grating == '600ZD':
        flux = [flux_stack_flat[:3906],flux_stack_flat[3906:]]
        ivar = [ivar_stack_flat[:3906],ivar_stack_flat[3906:]]

    if grating == '1200G':
        flux = [flux_stack_flat[:4219],flux_stack_flat[4219:]]
        ivar = [ivar_stack_flat[:4219],ivar_stack_flat[4219:]]

    wave_flat = wavelengths
    flux_flat = flux_stack_flat
    ivar_flat = ivar_stack_flat

    #Calculate an approximation of the signal-to-noise for the 1200G grating
    #if grating == '1200G':
    #    sn = data_redux.calc_sn(wave_flat, flux_stack_flat, grating=grating,
    #    cont_regions=cont_regions, continuum=norm_stack_flat, zrest=zrest)

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

    feh_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, fit_ranges=feh_mask, zrest=zrest)

    alphafe_fit_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, fit_ranges=alphafe_mask, zrest=zrest)
    """

    #Construct a general mask for the spectrum
    spec_mask = data_redux.construct_mask(wave_flat, flux_flat, ivar_flat,
    mask_ranges=mask_ranges, zrest=zrest)

    feh_fit_mask = spec_mask
    alphafe_fit_mask = spec_mask

    #Identify whether there is wavelength information in the blue and/or red
    #synthetic spectra ranges
    #wb = np.where((wave_flat[spec_mask] >= 4100.)&(wave_flat[spec_mask] < 6300.))[0]
    #wr = np.where((wave_flat[spec_mask] >= 6300.)&(wave_flat[spec_mask] < 9100.))[0]

    #Initialize a hash table to use to store the synthetic spectral data in memory
    hash = {}

    def construct_synth(wvl, teff_in, logg_in, feh_in, alphafe_in, hash=None):
        """
        Helper function for the get_synth() functions
        """

        wvl_synth, relflux_synth = read_synth.read_interp_synth(teff=teff_in, logg=logg_in,
        feh=feh_in, alphafe=alphafe_in, data_path=synth_path, hash=hash)

        #Make sure that the synthetic spectrum is within the data range for observations,
        #given the use of different gratings and central wavelengths
        wsynth = np.where((wvl_synth > wvl[0])&(wvl_synth < wvl[-1]))[0]

        return wvl_synth[wsynth], relflux_synth[wsynth]

    #When an overall metallicity change occurs, repeat the refinement until the value
    #does not change significantly -- define significantly as within 0.1 dex
    #Based on Shetrone et al. 2009

    #Set the initial parameters
    #phot_dict = {'teffphot': teffphot, 'loggphot': loggphot, 'teffphoterr': teffphoterr,
    #'loggphoterr': loggphoterr, 'flex_factor': flex_factor}

    #if teffphot > 6000.: teffphot = 6000.
    #if teffphot < 3000: teffphot = 3000.

    maxiter = 50
    feh_thresh = 0.001; alphafe_thresh = 0.001
    feh0 = feh_def; alphafe0 = alphafe_def

    i = 0; converge_flag = 0
    while i < maxiter:

        def get_synth_step1(wvl, feh_fit, fit=True):

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
            relflux_synth_list = []
            for i in range(len(slit_list_str)):
                wvl_synth, relflux_synth = construct_synth(wvl, teffphot_list[i], loggphot_list[i], feh_fit, alphafe0,
                hash=hash)

                #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array

                relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
                wvl, dlam)

                relflux_synth_list.append(relflux_interp_smooth)

            if fit == True:
                ivar_masked = [item[feh_fit_mask] for item in ivar_list]
            else:
                ivar_masked = ivar_list

            relflux_interp_smooth_stack = np.ma.average(np.array(relflux_synth_list), axis=0, weights = np.array(ivar_masked)).filled(0.)

            return relflux_interp_smooth_stack

        #Perform the fit for [Fe/H], Dlam, Teff

        params0 = feh0
        sigma_flat = ivar_flat[feh_fit_mask]**(-0.5)
        npix_fit = float(len(sigma_flat))

        best_params0, covar0 = curve_fit(get_synth_step1, wave_flat[feh_fit_mask], flux_flat[feh_fit_mask], p0=params0,
        sigma=sigma_flat, bounds=([-2.5],[1.]), absolute_sigma=True,
        ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

        #Load in the best fit synthetic spectrum
        feh_best0 = best_params0[0]

        def get_synth_step2(wvl, alphafe_fit, fit=True):
            """
            In this iteration, the effective temperature, metallicity, and smoothing parameter
            are held constant at the values from step1, and the alpha abundance is varied.
            """

            relflux_synth_list = []

            for i in range(len(slit_list_str)):
                #Read in the synthetic spectrum
                wvl_synth, relflux_synth = construct_synth(wvl, teffphot_list[i], loggphot_list[i], feh_best0, alphafe_fit,
                hash=hash)

                relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
                wvl, dlam)

                relflux_synth_list.append(relflux_interp_smooth)

            if fit == True:
                ivar_masked = [item[alphafe_fit_mask] for item in ivar_list]
            else:
                ivar_masked = ivar_list

            relflux_interp_smooth_stack = np.ma.average(np.array(relflux_synth_list), axis=0, weights = np.array(ivar_masked)).filled(0.)

            return relflux_interp_smooth_stack

        #Perform the fit [alpha/Fe]
        params1 = alphafe0
        sigma_flat = ivar_flat[alphafe_fit_mask]**(-0.5)

        best_params1, covar1 = curve_fit(get_synth_step2, wave_flat[alphafe_fit_mask],
        flux_flat[alphafe_fit_mask], p0=params1, sigma=sigma_flat, bounds=([-1.], [1.]),
        absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

        alphafe_best1 = best_params1[0]
        best_synth1 = get_synth_step1(wave_flat,alphafe_best1, fit=False)

        #plt.figure()
        #plt.plot(wave_flat, best_synth1, c='k')
        #plt.plot(wave_flat, flux_flat, c='silver', lw=0.5)
        #plt.show()

        #Refine the continuum normalization
        flux_refine, ivar_refine, refinedcont = data_redux.cont_refine(wavelengths_split,
        flux, ivar, wave_flat, best_synth1, cont_mask=cont_mask, grating=grating, zrest=zrest)

        if (np.abs(feh_best0 - feh0) < feh_thresh) and (np.abs(alphafe_best1 - alphafe0) <\
        alphafe_thresh):
            print('Continuum iteration converged')
            converge_flag = 1
            break
        else:
            print(np.mean(teffphot_list), feh_best0, alphafe_best1, flush=True)
            feh0 = feh_best0; alphafe0 = alphafe_best1
            flux_flat = flux_refine; ivar_flat = ivar_refine
            #feh_fit_mask = data_redux.construct_mask(wave_flat, flux_refine, ivar_refine)
            #alphafe_fit_mask = data_redux.construct_mask(wave_flat, flux_refine, ivar_refine)
            i += 1

    #### Perform the fit again with the revised spectrum ####
    def get_synth_step3(wvl, feh_fit, fit=True):
        """
        In this iteration, the effective temperature is held constant at the values from step1,
        the alpha abundance is assumed to be solar, and the metallicity is re-determined
        based on the revised observed spectrum
        """

        relflux_synth_list = []

        for i in range(len(slit_list_str)):

            #Read in the synthetic spectrum
            wvl_synth, relflux_synth = construct_synth(wvl, teffphot_list[i], loggphot_list[i], feh_fit, alphafe_best1,
            hash=hash)

            #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

            relflux_synth_list.append(relflux_interp_smooth)

            if fit==True:
                ivar_masked = [item[feh_fit_mask] for item in ivar_list]
            else:
                ivar_masked = ivar_list

        relflux_interp_smooth_stack = np.ma.average(np.array(relflux_synth_list), axis=0, weights = np.array(ivar_masked)).filled(0.)

        return relflux_interp_smooth_stack

    #Calculate the chi-squared value at termination of the continuum iteration
    sigma_flat = ivar_flat[feh_fit_mask]**(-0.5)

    chi0 = data_redux.compare_spectra(flux_flat[feh_fit_mask], get_synth_step1(wave_flat[feh_fit_mask],
    feh_best0, fit=True), stat_name='chisqr', weights=sigma_flat**(-0.5),
    nvarys=1)

    params2 = feh_best0

    best_params2, covar2 = curve_fit(get_synth_step3, wave_flat[feh_fit_mask],
    flux_refine[feh_fit_mask], p0=params2, sigma=ivar_refine[feh_fit_mask]**(-0.5),
    bounds=([-2.5],[1.]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

    feh_best2 = best_params2[0]
    print('feh step 2: ',feh_best2)

    #### Now re-determine the total alpha abundance of the atmosphere ####
    def get_synth_step4(wvl, alphafe_fit, fit=True):
        """
        In this iteration, the effective temperature is held constant at the value from step1,
        the metallicity is held constant at the previously determined value, and the [alpha/Fe]
        is re-determined based on the revised observed spectrum
        """

        relflux_synth_list = []

        for i in range(len(slit_list_str)):

            #Read in the synthetic spectrum
            wvl_synth, relflux_synth = construct_synth(wvl, teffphot_list[i], loggphot_list[i], feh_best2, alphafe_fit,
            hash=hash)

            #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

            relflux_synth_list.append(relflux_interp_smooth)

        if fit==True:
            ivar_masked = [item[alphafe_fit_mask] for item in ivar_list]
        else:
            ivar_masked = ivar_list

        relflux_interp_smooth_stack = np.ma.average(np.array(relflux_synth_list), axis=0, weights = np.array(ivar_masked)).filled(0.)

        return relflux_interp_smooth_stack

    params3 = alphafe_best1
    best_params3, covar3 = curve_fit(get_synth_step4, wave_flat[alphafe_fit_mask],
    flux_refine[alphafe_fit_mask], p0=params3, sigma=ivar_refine[alphafe_fit_mask]**(-0.5),
    bounds=([-1.],[1.]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

    alphafe_best3 = best_params3[0]
    print('alphafe step 3: ',alphafe_best3)

    #Calculate the chi-squared statistic
    chi3 = data_redux.compare_spectra(flux_refine[alphafe_fit_mask],
    get_synth_step4(wave_flat[alphafe_fit_mask], alphafe_best3, fit=True), stat_name='chisqr',
    weights=ivar_refine[alphafe_fit_mask], nvarys=1)

    #### Recalculate the metallicity a final time ####
    def get_synth_step5(wvl, feh_fit, fit=True):

        #Read in the synthetic spectrum
        relflux_synth_list = []

        for i in range(len(slit_list_str)):

            #Read in the synthetic spectrum with both blue and red components
            wvl_synth, relflux_synth = construct_synth(wvl, teffphot_list[i], loggphot_list[i], feh_fit, alphafe_best3,
            hash=hash)

            #Smooth the synthetic spectrum and interpolate it onto the observed wavelength array
            relflux_interp_smooth = data_redux.smooth_gauss_wrapper(wvl_synth, relflux_synth,
            wvl, dlam)

            relflux_synth_list.append(relflux_interp_smooth)

        if fit==True:
            ivar_masked = [item[feh_fit_mask] for item in ivar_list]
        else:
            ivar_masked = ivar_list

        relflux_interp_smooth_stack = np.ma.average(np.array(relflux_synth_list), axis=0, weights = np.array(ivar_masked)).filled(0.)

        return relflux_interp_smooth_stack

    params4 = feh_best2

    best_params4, covar4 = curve_fit(get_synth_step5, wave_flat[feh_fit_mask],
    flux_refine[feh_fit_mask], p0=params4, sigma=ivar_refine[feh_fit_mask]**(-0.5),
    bounds=([-2.5],[1.]), absolute_sigma=True, ftol=1.e-10, gtol=1.e-10, xtol=1.e-10)

    feh_best4 = best_params4[0]
    print('feh step 4:', feh_best4)
    best_synth = get_synth_step5(wave_flat, feh_best4, fit=False)

    #Calculate the chi-squared value
    chi4 = data_redux.compare_spectra(flux_refine[feh_fit_mask],
    get_synth_step5(wave_flat[feh_fit_mask], feh_best4, fit=True), stat_name='chisqr',
    weights=ivar_refine[feh_fit_mask], nvarys=1)

    #Final best fit parameters
    best_params = [np.mean(teffphot_list), feh_best4, alphafe_best3]
    print(best_params)

    #Unreduced parameter errors based on the covariance matrix returned by the fit
    #(Statistical uncertainty)

    params_err = np.sqrt([np.diag(covar4)[0],
                          np.diag(covar3)[0]])
    print(params_err)

    #Now perform a check of the uncertainties returned by the fit by analyzing the full
    #chi-squared distribution about the minimum
    params_dict = {'teff': best_params[0], 'logg': loggphot_list, 'feh': best_params[1],
    'alphafe': best_params[2], 'dlam': dlam}

    #perr_teff = {'teff': params_err[0], 'logg': 0., 'feh': 0., 'alphafe': 0., 'dlam':0.}
    perr_feh = {'teff': 0., 'logg': 0., 'feh': params_err[0], 'alphafe': 0., 'dlam': 0.}
    perr_alphafe = {'teff': 0., 'logg': 0., 'feh': 0., 'alphafe': params_err[1], 'dlam': 0.}

    #xteff, teffcontour = data_redux.check_fit(params_dict, perr_teff, wave_flat, flux_refine,
    #    ivar_refine, mask=feh_fit_mask, hash=hash, phot_dict=phot_dict, synth_path=synth_path,
    #    dlamfix=True)

    xfeh, fehcontour = data_redux.check_fit_coadd(params_dict, perr_feh, wave_flat, flux_refine,
    ivar_refine, mask=feh_fit_mask, hash=hash, synth_path=synth_path,
    slit_list=slit_list_str, teffphot_list=teffphot_list, loggphot_list=loggphot_list, ivar_list=ivar_list,
    grating=grating)

    xalphafe, alphafecontour = data_redux.check_fit_coadd(params_dict, perr_alphafe, wave_flat,
    flux_refine, ivar_refine, mask=alphafe_fit_mask, hash=hash, synth_path=synth_path,
    slit_list=slit_list_str,teffphot_list=teffphot_list,loggphot_list=loggphot_list,ivar_list=ivar_list,
    grating=grating)

    hash = {}

    #Now scale the statistical uncertainties by the reduced chi-squared value to construct
    #the fit uncertainty

    nvarys = 2
    chifeh = chi4/float(len(flux_refine[feh_fit_mask])-nvarys)
    chialpha = chi3/float(len(flux_refine[alphafe_fit_mask])-nvarys)

    params_err *= np.sqrt( np.array( [chifeh, chialpha] ) )

    #Calculate the signal-to-noise ratio for the 600ZD grating
    if grating == '600ZD' or grating == '1200G':
        sn = data_redux.calc_sn(wave_flat, flux_refine, grating=grating, continuum=best_synth,
        zrest=zrest)

    #Save the data
    specabund = params_dict.copy()
    specabund.update({'tefferr': 0., 'feherr': params_err[0],
    'alphafeerr': params_err[1], 'chifeh': chifeh, 'teffphot': np.mean(teffphot_list),
    'teffphoterr': np.sqrt(np.sum(np.array(teffphoterr_list)**2.)), 'fehnpix': len(flux_refine[feh_fit_mask]),
    'alphafenpix': len(flux_refine[alphafe_fit_mask]), 'npix': len(flux_refine[spec_mask]),
    'chialpha': chialpha, 'fehcontour': fehcontour,
    'alphafecontour': alphafecontour, 'wvl': wave_flat, 'flux': flux_refine,
    'ivar': ivar_refine, 'synth': best_synth, 'zrest': zrest,
    'fehcontourx': xfeh, 'alphafecontourx': xalphafe,
    'converge_flag': converge_flag,
    'contspec': flux_flat, 'contivar': ivar_flat, 'continuum': flatten(norm),
    'contmask': flatten(contmask), 'cont_flag': cont_flag, 'refinedcont': refinedcont,
    'norm_list': norm_list, 'ivar_list': ivar_list, 'flux_list': flux_list,
    'spec': flatten(flux), 'ivarspec': flatten(ivar),
    'tellspec': flatten(flux_tell), 'tellivar': flatten(ivar_tell), 'sn': sn})

    if m31disk is not None:
        save_path += 'm31disk/coadds'
    else:
        save_path += slitmask_name

    if not os.path.exists(save_path): os.makedirs(save_path)

    if m31disk is not None:
        with open(save_path+'/specabund_m31disk.'+fname_slit_str+'pkl', 'wb') as f:
            pickle.dump(specabund, f)
            f.close()
    else:
        with open(save_path+'/specabund_'+slitmask_name+'.'+fname_slit_str+'pkl', 'wb') as f:
            pickle.dump(specabund, f)
            f.close()

    return specabund
