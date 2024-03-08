"""
@author: I.Escala (iescala@carnegiescience.edu)

A Python implentation/modification of feh.pro (E.N. Kirby)
built for compatability with PARSEC isochrones (CMD v3.3 and up, see below)
and various photomtetric filter sets

Includes an additional option to use AGB isochrones (with fixed stellar age)
for stars above the TRGB

WARNING: Johnson-Cousins and HST/ACS WFC isochrones from more recent versions
of CMD input form than other sets -- probably recommended that you
check that you are using the most recent versions of the PARSEC isochrones
before usage
"""

from scipy.interpolate import griddata, SmoothBivariateSpline, interp1d, interp2d
import pickle
import numpy as np
import glob
import sys
import os

def from_phot(mag_in, color_in, err_mag_in=None, err_color_in=None, kind='parsec', filter='VI',
             dm=None, ddm=None, extrap=True, enforce_bounds=False, age=None, check_agb=False,
             root=os.getcwd(), k=2):

    """
    PURPOSE: Calculate photometric metallicity with PARSEC isochrones.

    CALLING SEQUENCE:
    -----------------
    phot = from_phot(mag_in, clr_in, err_mag_in=err_mag_in, err_color_in=err_color_in,
            filter='cfht', dm=dm, ddm=ddm)

    INPUTS:
    -------
    mag_in: An array of extinction-corrected magnitudes.
    clr_in: An array of dereddened colors.  mag_in is always the
       redder filter. For example, if your color is V-I, then mag_in
       must be I.  clr_in must have the same number of elements as
       mag_in.
    dm: The absolute distance modulus (m-M)_0.  It should have
        one element for all the input stars, or it should have as many
        elements as the input stars.  In other words, each star can
        have its own distance modulus.
    ddm: The error on the distance modulus.

    err_mag_in: An array of photometric errors on the magnitudes.  It
        must have the same number of elements as mag_in.
    err_clr_in: An array of photometric errors on the colors.  It
        must have the same number of elements as mag_in.
    kind: string: The isochrone set to use. Currently only compatible with PARSEC.
    filter: string: The photometric system for mag_in and clr_in. Currently only compatible
            with 2MASS, CFHT, and Johnson-Cousins photometric systems.
    extrap: boolean: Whether to extrapolate beyond the most metal-poor isochrone and beyond
            the tip of the red giant branch. Default is true.
    enforce_bounds: boolean: If a value of [Fe/H], Teff, or logg that is interpolated
                    based on the isochrones is outside of the range of the isochrones,
                    set the values to NaN. Default is False.
    age: int, the age of the RGB isochrones to use in Gyr. If None,
         return an array with quantities computed for various stellar ages
    check_agb: boolean, whether to check for AGB stars in the data based on
               magnitude above the TRGB, and if
               they are present, use 0.5 Gyr AGB isochrones (instead of RGB)
               for this subset. Default False.
    root: string, root directory that hosts isochrone data folder
    k: int: degree of spline for extrapolation beyond bounds of isochrone grid

    OUTPUTS:
    --------
    A structure with as many elements as mag_in.  The structure
    contains the following tags:
     AGES:    An array of stellar ages in Gyr.  The number of ages
        depends on the choice of isochrone.
     FEH:     An array of photometric metallicities corresponding to
        each age.
     TEFF:    An array of effective temperatures (K) corresponding to
        each age.
     LOGG:    An array of surface gravities (log cm/s^2) corresponding
        to each age.
     ERR_FEH: An array of errors on the photometric metallicity
        corresponding to each element in FEH.  This tag exists only if
        the inputs include photometric errors.
     ERR_TEFF: An array of errors on the effective temperatures
        corresponding to each element in TEFF.  This tag exists only
        if the inputs include photometric errors.
     ERR_TEFF_ARR: A 2D array containing the full MCMC distribution for the photometric
        errors, with dimensions equal to NAGES (number of stellar ages for a given isochrone)
        and NMC (the number of iterations in the MCMC that determines the photometric errors).
     ERR_LOGG: An array of errors on the surface gravities
        corresponding to each element in LOGG.  This tag exists only
        if the inputs include photometric errors.
    """

    isochrones = ['parsec']
    filters = ['cfht', '2mass', 'vi', 'hstacswfc']

    if kind.lower() not in isochrones:
        sys.stderr.write('Please specificy a valid isochrone set')
        return
    if filter.lower() not in filters:
        sys.stderr.write('Please specify a valid filter')
        return

    if check_agb: print('CHECK_AGB == TRUE. Using BOTH RGB and AGB isochrones as appropriate')

    #if kind == 'parsec': Zsun = 0.0152

    if isinstance(mag_in, list) or isinstance(mag_in, np.ndarray):
        pass
    else:
        mag_in = [mag_in]; color_in = [color_in]
        if err_mag_in is not None and err_color_in is not None:
            err_mag_in = [err_mag_in]; err_color_in = [err_color_in]
            err_mag_in = np.array(err_mag_in); err_color_in = np.array(err_color_in)

    mag_in = np.array(mag_in); color_in = np.array(color_in)

    n = len(mag_in)
    dm_uniq = np.unique(dm)
    ndm = len(dm_uniq)
    if (ndm == 1) and (n > 1): dm = np.full(n, dm)

    if ddm is not None:

        if err_mag_in is not None:
            err_mag_in = np.sqrt(err_mag_in**2. + ddm**2.)
        else:
            if (ndm == 1) and (n > 1):
                err_mag_in = np.full(n, ddm)
            else:
                err_mag_in = ddm

        if err_color_in is None:
            err_color_in = np.zeros(n)

    #If still None
    if err_color_in is None and err_mag_in is None:
        err_color_in = np.full(n, np.nan)
        err_mag_in = np.full(n, np.nan)

    if age is None:
        iso_files = glob.glob(os.path.join(root, 'isochrones/rgb/feh_'+filter+'_*_'+kind+'.dat'))
    else:
        iso_files = [os.path.join(root,'isochrones/rgb/feh_'+filter+'_'+str(age)+'_'+kind+'.dat')]

    nages = len(iso_files)
    nmc = 1000

    phots = {}
    for i in range(n):
        phot = {}
        phot['mag'] = mag_in[i]
        phot['color'] = color_in[i]
        phot['err_mag'] = err_mag_in[i]
        phot['err_color'] = err_color_in[i]
        phot['ages'] = np.full(nages, np.nan)
        phot['feh'] = np.full(nages, np.nan)
        phot['teff'] = np.full(nages, np.nan)
        phot['L'] = np.full(nages, np.nan)
        phot['logg'] = np.full(nages, np.nan)
        phot['err_feh'] = np.full(nages, np.nan)
        phot['err_teff'] = np.full(nages, np.nan)
        phot['err_logg'] = np.full(nages, np.nan)
        phot['err_teff_arr'] = np.full((nages, nmc), np.nan)
        if check_agb: phot['wagb'] = np.full(nages, False)
        phots[i] = phot

    if kind == 'parsec':

        if filter.lower() == 'cfht':
            usecols = (1,2,4,5,6,7,24,26) #needs to be updated
            skiprows = 0
        if filter.lower() == '2mass':
            usecols = (1,2,5,6,7,23,25) #needs to be updated
            skiprows = 0
        if filter.lower() == 'vi':
            usecols = (1,2,6,7,8,9,30,32) #CMD v3.6
            skiprows = 0
        if filter.lower() == 'hstacswfc':
            #usecols = (0,2,6,7,8,9,26,35) CMD v3.6
            usecols = (1,2,6,7,8,9,29,34)
            skiprows = 0

    if check_agb: #If this flag is set to true, check where given stars are AGB stars assuming 12 Gyr RGB isocrhones

        wagb_in = np.full((nages, n), False)
        for kk,rgb_file in enumerate(iso_files):

            mh,_,_,_,_,label, mag_blue, mag_red = np.loadtxt(rgb_file,
                                                skiprows=skiprows, unpack=True,usecols=usecols)

            mh_uniq = np.unique(mh)
            mag_trgb = []
            color_trgb = []

            for ii in range(len(mh_uniq)):

                wrgb = (label == 3) & (mh == mh_uniq[ii])
                mag_trgb.append( mag_red[wrgb][-1] ) # IN ABSOLUTE MAGS
                color_trgb.append( mag_blue[wrgb][-1] - mag_red[wrgb][-1] )

            if err_mag_in is not None:
                agb_thresh = np.array([np.nanmax([err_mag_in[i], 0.1]) for i in range(mag_in.size)])
            else:
                agb_thresh = np.full(mag_in.size, 0.1)

            #Need to calculate it according to every different distance modulus
            wagb_kk = np.full(len(mag_in), False)
            for jj in range(ndm):

                mag_trgb += dm_uniq[jj]

                wdm = np.where( dm == dm_uniq[jj] )
                if len(wdm) == 1: wdm = wdm[0]

                ftrgb = interp1d(color_trgb, mag_trgb, fill_value="extrapolate")
                trgb = ftrgb(color_in[wdm])

                wagb_kk[wdm] = (mag_in[wdm] - trgb) < -agb_thresh[wdm]

                mag_trgb -= dm_uniq[jj]

            for i in range(len(mag_in)):
                phots[i]['wagb'][kk] = wagb_kk[i]

            wagb_in[kk] = wagb_kk

        if extrap: agb_save_file = 'isochrones/agb/feh_'+filter+'_1_'+kind+'_agb_grid_extrap.pkl'
        else: agb_save_file = 'isochrones/agb/feh_'+filter+'_1_'+kind+'_agb_grid.pkl'

        if not os.path.exists(os.path.join(root, agb_save_file)):

            #Now build up the AGB grid independently of the RGB grid because need to assume 0.5 Gyr age
            agb_file = 'isochrones/agb/feh_'+filter+'_1_'+kind+'.dat'
            mh, age, logL, Te, logg, label, mag_blue, mag_red = np.loadtxt(os.path.join(root,agb_file),
                                                                skiprows=skiprows, unpack=True,
                                                                usecols=usecols)

            mh_uniq = np.unique(mh)
            mag_grid_agb = []; clr_grid_agb = []; feh_grid_agb = []
            logt_grid_agb = []; logg_grid_agb = []; logL_grid_agb = []

            for ii in range(len(mh_uniq)):

                wagb = (label == 7) & (mh == mh_uniq[ii])

                mag = mag_red[wagb]
                color = mag_blue[wagb]-mag_red[wagb]
                #feh = np.log10(Zi[wagb]/Zsun)
                feh = mh[wagb]
                grav = logg[wagb]
                teff = Te[wagb]
                lum = logL[wagb]

                if extrap:

                    #Extrapolate beyond the tip of the red giant branch
                    m, b = np.polyfit(color[-2:], mag[-2:], 1)

                    #Extend the colors redward by 1.2 mags
                    xx = np.arange(color[-1], color[-1]+1.2, 0.1)
                    yy = xx*m + b

                    #Fill in this extension with the values for Teff, Logg,
                    #and [Fe/H] from the TRGB
                    zz = np.full(len(xx), feh[-1])
                    gg = np.full(len(xx), grav[-1])
                    tt = np.full(len(xx), teff[-1])
                    lll = np.full(len(xx), lum[-1])

                    #Extend the grid with these values
                    color = np.append(color, xx)
                    mag = np.append(mag, yy)
                    feh = np.append(feh, zz)
                    grav = np.append(grav, gg)
                    teff = np.append(teff, tt)
                    lum = np.append(lum, lll)

                mag_grid_agb.extend(mag)
                clr_grid_agb.extend(color)
                feh_grid_agb.extend(feh)
                logg_grid_agb.extend(grav)
                logt_grid_agb.extend(teff)
                logL_grid_agb.extend(lum)

            mag_grid_agb = np.array(mag_grid_agb)
            clr_grid_agb = np.array(clr_grid_agb)
            feh_grid_agb = np.array(feh_grid_agb)
            logt_grid_agb = np.array(logt_grid_agb)
            logg_grid_agb = np.array(logg_grid_agb)
            logL_grid_agb = np.array(logL_grid_agb)

            agb_dict = {'mag': mag_grid_agb, 'clr': clr_grid_agb, 'feh': feh_grid_agb, 'logt': logt_grid_agb,
                        'logg': logg_grid_agb, 'logL': logL_grid_agb}

            with open(os.path.join(root,agb_save_file), 'wb') as f:
                pickle.dump(agb_dict, f)

        else:

            with open(os.path.join(root,agb_save_file), 'rb') as f:
                agb_dict = pickle.load(f, encoding='latin1')

            mag_grid_agb = agb_dict['mag']; clr_grid_agb = agb_dict['clr']
            feh_grid_agb = agb_dict['feh']; logt_grid_agb = agb_dict['logt']
            logg_grid_agb = agb_dict['logg']; logL_grid_agb = agb_dict['logL']

    ### Now back to the RGB stars

    for kk,rgb_file in enumerate(iso_files):

        mh, age, logL, Te, logg, label, mag_blue, mag_red = np.loadtxt(rgb_file,
                                                                         skiprows=skiprows, unpack=True,
                                                                         usecols=usecols)
        if filter == 'hstacswfc' or filter.lower() == 'vi':
            age = 10**age

        age_str = str(np.round(np.unique(age)[0]/1.e9, decimals=0))[:-2]

        if not extrap: rgb_save_file = 'isochrones/rgb/feh_'+filter.lower()+'_'+age_str+'_'+kind+'_rgb_grid.pkl'
        else: rgb_save_file = 'isochrones/rgb/feh_'+filter.lower()+'_'+age_str+'_'+kind+'_rgb_grid_extrap.pkl'

        if not os.path.exists(os.path.join(root,rgb_save_file)):

            feh_val = np.unique(mh)

            print(kind.upper()+' isochrone set using '+filter.upper()+' filter, assuming age = '+\
            str( np.round( np.unique(age)[0]/1.e9, decimals=0) )+' Gyr')

            mag_grid = []; clr_grid = []; feh_grid = []
            logt_grid = []; logg_grid = []; logL_grid = []

            for ii in range(len(feh_val)):

                wrgb = (label == 3) & (mh == feh_val[ii])

                mag = mag_red[wrgb]
                color = mag_blue[wrgb]-mag_red[wrgb]
                #feh = np.log10(Zi[wrgb]/Zsun)
                feh = mh[wrgb]
                grav = logg[wrgb]
                teff = Te[wrgb]
                lum = logL[wrgb]

                if extrap:

                    #Extrapolate beyond the tip of the red giant branch
                    m, b = np.polyfit(color[-2:], mag[-2:], 1)

                    #Extend the colors redward by 1.2 mags
                    xx = np.arange(color[-1], color[-1]+1.2, 0.1)
                    yy = xx*m + b

                    #Fill in this extension with the values for Teff, Logg,
                    #and [Fe/H] from the TRGB
                    zz = np.full(len(xx), feh[-1])
                    gg = np.full(len(xx), grav[-1])
                    tt = np.full(len(xx), teff[-1])
                    lll = np.full(len(xx), lum[-1])

                    #Extend the grid with these values
                    color = np.append(color, xx)
                    mag = np.append(mag, yy)
                    feh = np.append(feh, zz)
                    grav = np.append(grav, gg)
                    teff = np.append(teff, tt)
                    lum = np.append(lum, lll)

                mag_grid.extend(mag)
                clr_grid.extend(color)
                feh_grid.extend(feh)
                logg_grid.extend(grav)
                logt_grid.extend(teff)
                logL_grid.extend(lum)

            mag_grid = np.array(mag_grid); clr_grid = np.array(clr_grid)
            logg_grid = np.array(logg_grid); logt_grid = np.array(logt_grid)
            feh_grid = np.array(feh_grid); logL_grid = np.array(logL_grid)

            rgb_dict = {'mag': mag_grid, 'clr': clr_grid, 'logg': logg_grid, 'logt': logt_grid,
                   'feh': feh_grid, 'logL': logL_grid}

            with open(os.path.join(root,rgb_save_file), 'wb') as f:
                pickle.dump(rgb_dict, f)

        else:

            with open(os.path.join(root,rgb_save_file), 'rb') as f:
                rgb_dict = pickle.load(f, encoding='latin1')

            mag_grid = rgb_dict['mag']; clr_grid = rgb_dict['clr']
            logg_grid = rgb_dict['logg']; logt_grid = rgb_dict['logt']
            feh_grid = rgb_dict['feh']; logL_grid = rgb_dict['logL']

        print(kind.upper()+' isochrone set using '+filter.upper()+' filter, assuming age = '+\
            str( np.round( np.unique(age)[0]/1.e9, decimals=0) )+' Gyr')

        for jj in range(ndm):

            mag_grid += dm_uniq[jj]
            if check_agb: mag_grid_agb += dm_uniq[jj]

            wdm = np.where( dm == dm_uniq[jj] )
            if len(wdm) == 1: wdm = wdm[0]

            #If check_agb is True, use the RGB or AGB grids as appropriate
            if check_agb:

                wagb_dm = wagb_in[kk][wdm]
                color_n = color_in[wdm]
                mag_n = mag_in[wdm]

                feh_i = np.zeros(len(color_n))
                logt_i = np.zeros(len(color_n))
                logg_i = np.zeros(len(color_n))
                logL_i = np.zeros(len(color_n))

                feh_i[wagb_dm] = griddata((clr_grid_agb, mag_grid_agb), feh_grid_agb,
                                          (color_n[wagb_dm], mag_n[wagb_dm]))
                logt_i[wagb_dm] = griddata((clr_grid_agb, mag_grid_agb), logt_grid_agb,
                                           (color_n[wagb_dm], mag_n[wagb_dm]))
                logg_i[wagb_dm] = griddata((clr_grid_agb, mag_grid_agb), logg_grid_agb,
                                           (color_n[wagb_dm], mag_n[wagb_dm]))
                logL_i[wagb_dm] = griddata((clr_grid_agb, mag_grid_agb), logL_grid_agb,
                                           (color_n[wagb_dm], mag_n[wagb_dm]))

                feh_i[~wagb_dm] = griddata((clr_grid, mag_grid), feh_grid,
                                           (color_n[~wagb_dm], mag_n[~wagb_dm]))
                logt_i[~wagb_dm] = griddata((clr_grid, mag_grid), logt_grid,
                                            (color_n[~wagb_dm], mag_n[~wagb_dm]))
                logg_i[~wagb_dm] = griddata((clr_grid, mag_grid), logg_grid,
                                            (color_n[~wagb_dm], mag_n[~wagb_dm]))
                logL_i[~wagb_dm] = griddata((clr_grid, mag_grid), logL_grid,
                                            (color_n[~wagb_dm], mag_n[~wagb_dm]))

            #Use a uniform RGB grid for all stars if check_agb is False
            else:

                feh_i = griddata((clr_grid, mag_grid), feh_grid, (color_in[wdm], mag_in[wdm]))
                logt_i = griddata((clr_grid, mag_grid), logt_grid, (color_in[wdm], mag_in[wdm]))
                logg_i = griddata((clr_grid, mag_grid), logg_grid, (color_in[wdm], mag_in[wdm]))
                logL_i = griddata((clr_grid, mag_grid), logL_grid, (color_in[wdm], mag_in[wdm]))

            t_i = 10**logt_i
            L_i = 10**logL_i

            out_of_bounds = np.isnan(feh_i)

            #Extrapolate blueward of the bluest isochrone ONLY for Teff and Logg
            #Use interp2d for these points ONLY since it is less robust than griddata
            if (len(feh_i[out_of_bounds]) > 0) and (not enforce_bounds):

                #Identify indices for points that are out of bounds
                arg = np.arange(0, len(feh_i))[out_of_bounds]

                #Construct an interpolator that can extrapolate
                #Note that griddata cannot extrapolate, but it is much more robust than *Spline functions
                #*Spline functions very dependent on smoothing parameter s and can generate wild swings
                f_t = SmoothBivariateSpline(clr_grid, mag_grid, logt_grid, kx=k, ky=k)
                f_g = SmoothBivariateSpline(clr_grid, mag_grid, logg_grid, kx=k, ky=k)
                f_f = SmoothBivariateSpline(clr_grid, mag_grid, feh_grid, kx=k, ky=k)
                f_l = SmoothBivariateSpline(clr_grid, mag_grid, logL_grid, kx=k, ky=k)

                #f_t = interp2d(clr_grid, mag_grid, logt_grid, fill_value='extrapolate')
                #f_g = interp2d(clr_grid, mag_grid, logg_grid, fill_value='extrapolate')
                #f_f = interp2d(clr_grid, mag_grid, feh_grid, fill_value='extrapolate')
                #f_l = interp2d(clr_grid, mag_grid, logL_grid, fill_value='extrapolate')


                #Now construct an interpolator for the AGB stars, if check_agb is True
                if check_agb:
                    f_t_agb = SmoothBivariateSpline(clr_grid_agb, mag_grid_agb, logt_grid_agb, kx=k, ky=k)
                    f_g_agb = SmoothBivariateSpline(clr_grid_agb, mag_grid_agb, logg_grid_agb, kx=k, ky=k)
                    f_f_agb = SmoothBivariateSpline(clr_grid_agb, mag_grid_agb, feh_grid_agb, kx=k, ky=k)
                    f_l_agb = SmoothBivariateSpline(clr_grid_agb, mag_grid_agb, logL_grid_agb, kx=k, ky=k)

                    #f_t_agb = interp2d(clr_grid_agb, mag_grid_agb, logt_grid_agb, fill_value='extrapolate')
                    #f_g_agb = interp2d(clr_grid_agb, mag_grid_agb, logg_grid_agb, fill_value='extrapolate')
                    #f_f_agb = interp2d(clr_grid_agb, mag_grid_agb, feh_grid_agb, fill_value='extrapolate')
                    #f_l_agb = interp2d(clr_grid_agb, mag_grid_agb, logL_grid_agb, fill_value='extrapolate')

                for ll in arg:

                    #If using a mix of RGB and AGB grids
                    if check_agb:

                        if wagb_dm[ll]:

                            logt_l = f_t_agb(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                            logg_l = f_g_agb(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                            feh_l = f_f_agb(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                            logL_l = f_l_agb(color_in[wdm][ll], mag_in[wdm][ll])[0][0]

                        else:

                            logt_l = f_t(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                            logg_l = f_g(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                            feh_l = f_f(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                            logL_l = f_l(color_in[wdm][ll], mag_in[wdm][ll])[0][0]

                    #If a uniform RGB grid is being used for all stars
                    else:

                        logt_l = f_t(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                        logg_l = f_g(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                        feh_l = f_f(color_in[wdm][ll], mag_in[wdm][ll])[0][0]
                        logL_l = f_l(color_in[wdm][ll], mag_in[wdm][ll])[0][0]

                    logt_i[ll] = logt_l
                    logg_i[ll] = logg_l
                    feh_i[ll] = feh_l
                    t_i[ll] = 10**logt_l
                    L_i[ll] = 10**logL_l

                w = np.where( (feh_i < feh_grid_mins) | (feh_i > feh_grid_maxs) )[0]
                if len(w) > 0:
                    feh_i[w] = np.nan

                w = np.where( (logt_i < logt_grid_mins) | (logt_i > logt_grid_maxs) )[0]
                if len(w) > 0:
                    t_i[w] = np.nan

                w = np.where( (logg_i < logg_grid_mins) | (logg_i > logg_grid_maxs) )[0]
                if len(w) > 0:
                    logg_i[w] = np.nan

                w = np.where( (logL_i < logL_grid_mins) | (logL_i > logL_grid_maxs) )[0]
                if len(w) > 0:
                    L_i[w] = np.nan

            for iw in range(len(wdm)):

                phots[wdm[iw]]['feh'][kk] = feh_i[iw]
                phots[wdm[iw]]['teff'][kk] = t_i[iw]
                phots[wdm[iw]]['ages'][kk] = np.unique(age)
                phots[wdm[iw]]['logg'][kk] = logg_i[iw]
                phots[wdm[iw]]['L'][kk] = L_i[iw]

            #if err_mag_in.all() != 0. and err_color_in.all() != 0.:
            if (err_mag_in is not None) and (err_color_in is not None):
                for ii in range(len(wdm)):

                    if np.isnan(err_color_in[wdm[ii]]) or np.isnan(err_mag_in[wdm[ii]]):
                        continue

                    if np.isnan(feh_i[ii]):
                        #phots[wdm[ii]]['err_feh'][kk] = np.nan
                        pass
                    else:
                        rand = np.random.normal(size=(nmc,2))
                        clr_mc = color_in[wdm[ii]] + rand[:,0]*err_color_in[wdm[ii]]
                        mag_mc = mag_in[wdm[ii]] + rand[:,1]*err_mag_in[wdm[ii]]

                        if check_agb:
                            if wagb_dm[ii]:
                                feh_mc = griddata((clr_grid_agb, mag_grid_agb), feh_grid_agb, (clr_mc, mag_mc), fill_value=np.nan)
                            else:
                                feh_mc = griddata((clr_grid, mag_grid), feh_grid, (clr_mc, mag_mc), fill_value=np.nan)

                        else:
                            feh_mc = griddata((clr_grid, mag_grid), feh_grid, (clr_mc, mag_mc), fill_value=np.nan)

                        wgood = np.where( (feh_mc >= -5.) & (feh_mc <= 2.) )[0]

                        if len(wgood) > 2:
                            phots[wdm[ii]]['err_feh'][kk] = np.std(feh_mc[wgood])
                        else:
                            phots[wdm[ii]]['err_feh'][kk] = np.nan

                    if np.isnan(t_i[ii]):
                        #phots[wdm[ii]]['err_teff'][kk] = np.nan
                        pass
                    else:
                        rand = np.random.normal(size=(nmc,2))
                        clr_mc = color_in[wdm[ii]] + rand[:,0]*err_color_in[wdm[ii]]
                        mag_mc = mag_in[wdm[ii]] + rand[:,1]*err_mag_in[wdm[ii]]

                        if check_agb:
                            if wagb_dm[ii]:
                                logt_mc = griddata((clr_grid_agb, mag_grid_agb), logt_grid_agb, (clr_mc, mag_mc), fill_value=np.nan)
                            else:
                                logt_mc = griddata((clr_grid, mag_grid), logt_grid, (clr_mc, mag_mc), fill_value=np.nan)

                        else:
                            logt_mc = griddata((clr_grid, mag_grid), logt_grid, (clr_mc, mag_mc), fill_value=np.nan)

                        wgood = np.where( (10**logt_mc >= 3000.) & (10**logt_mc <= 10000) )[0]

                        print(len(wgood))
                        if len(wgood) > 2:
                            phots[wdm[ii]]['err_teff'][kk] = np.std(10**logt_mc[wgood])
                        else:
                            phots[wdm[ii]]['err_teff'][kk] = np.nan

                        for mm in range(nmc):
                            phots[wdm[ii]]['err_teff_arr'][kk][mm] = 10**logt_mc[mm]

                    if np.isnan(logg_i[ii]):
                        #phots[wdm[ii]]['err_logg'][kk] = np.nan
                        pass
                    else:
                        rand = np.random.normal(size=(nmc,2))
                        clr_mc = color_in[wdm[ii]] + rand[:,0]*err_color_in[wdm[ii]]
                        mag_mc = mag_in[wdm[ii]] + rand[:,1]*err_mag_in[wdm[ii]]

                        if check_agb:
                            if wagb_dm[ii]:
                                logg_mc = griddata((clr_grid_agb, mag_grid_agb), logg_grid_agb, (clr_mc, mag_mc), fill_value=np.nan)
                            else:
                                logg_mc = griddata((clr_grid, mag_grid), logg_grid, (clr_mc, mag_mc), fill_value=np.nan)

                        else:
                            logg_mc = griddata((clr_grid, mag_grid), logg_grid, (clr_mc, mag_mc), fill_value=np.nan)

                        wgood = np.where( (logg_mc >= -2.) & (logg_mc <= 7.) )[0]

                        if len(wgood) > 2:
                            phots[wdm[ii]]['err_logg'][kk] = np.std(logg_mc[wgood])
                        else:
                            phots[wdm[ii]]['err_logg'][kk] = np.nan

            mag_grid -= dm_uniq[jj]
            if check_agb: mag_grid_agb -= dm_uniq[jj]

    #Sort the dictionary according to stellar age
    wsort = np.argsort(phots[0]['ages'])

    for i in range(len(phots)):

        phots[i]['ages'] = phots[i]['ages'][wsort]
        phots[i]['feh'] = phots[i]['feh'][wsort]
        phots[i]['teff'] = phots[i]['teff'][wsort]
        phots[i]['logg'] = phots[i]['logg'][wsort]
        phots[i]['L'] = phots[i]['L'][wsort]

        if err_mag_in is not None and err_color_in is not None:
            phots[i]['err_feh'] = phots[i]['err_feh'][wsort]
            phots[i]['err_teff'] = phots[i]['err_teff'][wsort]
            phots[i]['err_logg'] = phots[i]['err_logg'][wsort]
            phots[i]['err_teff_arr'] = phots[i]['err_teff_arr'][wsort]

    return phots
