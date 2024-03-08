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

def enforce_grid_check(teff, logg, feh, alphafe):

    """
    Enforce the grid constraints from Kirby et al. 2009 on a given combination
    of atmospheric model parameters.

    Parameters
    ----------
    teff: float: effective temperature
    logg: float: surface gravity
    feh: float: [Fe/H]
    alphafe: float: [alpha/Fe]

    Returns
    -------
    in_grid: boolean: if True, the specified parameters are within the K08 grid range
    """

    #Check that the effective temperature is within limits
    teff_lims = [3500., 8000.]
    if (teff < teff_lims[0]) or (teff > teff_lims[1]):
        in_grid = False; key = 0
        return in_grid, key

    logg_hi = 5.0
    if teff < 7000.: logg_lo = 0.0
    else: logg_lo = 0.5
    logg_lims = [logg_lo, logg_hi]

    #Check if the specified surface gravity is within limits
    if (logg < logg_lims[0]) or (logg > logg_lims[1]):
        in_grid = False; key = 1
        return in_grid, key

    #Check that the specified metallicity is within limits
    feh_lims = [-5., 0.]

    #Put checks in place based on the limits of the grid imposed by
    #difficulty in model atmosphere convergence
    teff_vals = np.array([3600., 3700., 3800., 3900., 4000., 4100.])
    feh_thresh = np.array([-4.9, -4.8, -4.8, -4.7, -4.4, -4.6])
    logg_thresh = np.array([[1.5], [2.5, 3.5, 4.0, 4.5], [4.0, 4.5],
                           [2.5, 3.0, 3.5, 4.0, 4.5, 5.0], [4.5, 5.0], [4.5, 5.0]])

    if teff in teff_vals:
        where_teff = np.where(teff_vals == teff)[0][0]
        if logg in logg_thresh[where_teff]:
            if (feh < feh_thresh[where_teff]) or (feh > feh_lims[1]):
                in_grid = False; key = 2
                return in_grid, key
    else:
        if (feh < feh_lims[0]) or (feh > feh_lims[1]):
            in_grid = False; key = 2
            return in_grid, key

    #Check that the alpha enhancement is within limits
    alpha_lims = [-0.8, 1.2]
    if (alphafe < alpha_lims[0]) or (alphafe > alpha_lims[1]):
        in_grid = False; key = 3
        return in_grid, key

    in_grid = True; key = 4
    return in_grid, key

def read_mult_synth(temp_index=0, logg_index=0, feh_index=0, alphafe_index=0,
data_path='', subset=True, kind='', xrange=[4100., 6300.], grid_check=True):

    teffarr = np.array([3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400,
                        4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400,
                        5500, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200,
                        7400, 7600, 7800, 8000])

    loggarr = np.around(np.arange(0.0,5.5,0.5),2)
    feharr = np.around(np.arange(-5.0,0.1,0.1),2)
    alphafearr = np.around(np.arange(-0.8,1.3,0.1),2)

    Teff = round(teffarr[temp_index]/100.)*100
    Logg = round(loggarr[logg_index]*10.)
    Feh = round(feharr[feh_index]*10.)
    Alphafe = round(alphafearr[alphafe_index]*10.)

    if Logg >= 0: gsign = '_'
    else: gsign = '-'

    if Feh >= 0: fsign = '_'
    else: fsign = '-'

    if Alphafe >= 0: asign = '_'
    else: asign = '-'

    if kind != '':

        if kind == 'feh':

            if subset: feharr = [0.0, -1.0, -2.0, -3.0, -4.0]
            title = "Varying [Fe/H] from %g to %g for "%(feharr[0],feharr[-1])
            suptitle = "T = %g, logg = %g, [$\\alpha$/Fe] = %g"%(teffarr[temp_index],loggarr[logg_index],alphafearr[alphafe_index])
            full_title = title + suptitle
            outfilename = ("t%2ig%s%2if%s%sa%s%2i"%(Teff,gsign,abs(Logg),'_','all',asign,abs(Alphafe))).replace(' ', '0')
            print(outfilename)

            wvls = []; relfluxes = []; labels=[]
            for feh in feharr:

                if grid_check:
                    #Make sure that the parameters are within the K10 grid range
                    in_grid = enforce_grid_check(teffarr[temp_index], loggarr[logg_index],
                    feh, alphafearr[alphafe_index])

                    if in_grid:

                        wvl, relflux,_,_ = read_synth(Teff=teffarr[temp_index], Logg=loggarr[logg_index],
                        Feh=feh, Alphafe=alphafearr[alphafe_index], gauss_sigma=0., data_path=data_path)

                        wvls += [wvl]; relfluxes += [relflux]

                        label = r'[Fe/H] = '+str(feh)
                        labels.append(label)

                else:

                    wvl, relflux,_,_ = read_synth(Teff=teffarr[temp_index], Logg=loggarr[logg_index],
                    Feh=feh, Alphafe=alphafearr[alphafe_index], gauss_sigma=0., data_path=data_path)

                    wvls += [wvl]; relfluxes += [relflux]

                    label = r'[Fe/H] = '+str(feh)
                    labels.append(label)

        elif kind == 'alpha':

            if subset: alphafearr=[-0.8, -0.4, 0.0, 0.4, 0.8]
            title = "Varying [$\\alpha$/Fe] from %g to %g for "%(alphafearr[0],alphafearr[-1])
            suptitle = "T = %g, logg = %g, [Fe/H] = %g"%(teffarr[temp_index],loggarr[logg_index],feharr[feh_index])
            full_title = title + suptitle
            outfilename = ("t%2ig%s%2if%s%2ia%s%s"%(Teff,gsign,abs(Logg),fsign,abs(Feh),'_','all')).replace(' ', '0')
            print(outfilename)

            wvls = []; relfluxes = []; labels=[]
            for alpha in alphafearr:

                if grid_check:

                    #Make sure that the parameters are within the K08 grid range
                    in_grid = enforce_grid_check(teffarr[temp_index], loggarr[logg_index],
                    feharr[feh_index], alpha)

                    if in_grid:

                        wvl, relflux,_,_ = read_synth(Teff=teffarr[temp_index], Logg=loggarr[logg_index],
                        Feh=feharr[feh_index], Alphafe=alpha, gauss_sigma=0., data_path=data_path)

                        wvls += [wvl]; relfluxes += [relflux]

                        label = r'[$\alpha$/Fe] = '+str(alpha)
                        labels.append(label)

                else:

                    wvl, relflux,_,_ = read_synth(Teff=teffarr[temp_index], Logg=loggarr[logg_index],
                    Feh=feharr[feh_index], Alphafe=alpha, gauss_sigma=0., data_path=data_path)

                    wvls += [wvl]; relfluxes += [relflux]

                    label = r'[$\alpha$/Fe] = '+str(alpha)
                    labels.append(label)

        elif kind == 'teff':

            if subset: teffarr=[3500, 4500, 5500, 6400, 7400]
            title = r"Varying T$_{eff}$ from %g to %g for "%(teffarr[0],teffarr[-1])
            suptitle = "logg = %g, [Fe/H] = %g, [$\\alpha$/Fe] = %g"%(loggarr[logg_index],feharr[feh_index], alphafearr[alphafe_index])
            full_title = title + suptitle
            outfilename = ("t%sg%s%2if%s%2ia%s%2i"%('all',gsign,abs(Logg),fsign,abs(Feh),asign,abs(Alphafe))).replace(' ', '0')
            print(outfilename)

            wvls = []; relfluxes = []; labels=[]
            for teff in teffarr:

                if grid_check:

                    #Make sure that the parameters are within the K08 grid range
                    in_grid = enforce_grid_check(teff, loggarr[logg_index],
                    feharr[feh_index], alphafearr[alphafe_index])

                    if in_grid:

                        wvl, relflux,_,_ = read_synth(Teff=teff, Logg=loggarr[logg_index],
                        Feh=feharr[feh_index], Alphafe=alphafearr[alphafe_index], gauss_sigma=0.,
                        data_path=data_path)

                        wvls += [wvl]; relfluxes += [relflux]

                        label = r'T$_{eff}$ = '+str(teff)+' K'
                        labels.append(label)

                else:

                    wvl, relflux,_,_ = read_synth(Teff=teff, Logg=loggarr[logg_index],
                    Feh=feharr[feh_index], Alphafe=alphafearr[alphafe_index], gauss_sigma=0.,
                    data_path=data_path)

                    wvls += [wvl]; relfluxes += [relflux]

                    label = r'T$_{eff}$ = '+str(teff)+' K'
                    labels.append(label)

        elif kind == 'logg':

            if subset: loggarr=[0.0, 1.0, 2.0, 3.0, 4.0]
            title = r"Varying log $g$ from %g to %g for "%(loggarr[0],loggarr[-1])
            suptitle = "T = %g, [Fe/H] = %g, [$\\alpha$/Fe] = %g"%(teffarr[temp_index],feharr[feh_index], alphafearr[alphafe_index])
            full_title = title + suptitle
            outfilename = ("t%2ig%s%sf%s%2ia%s%2i"%(Teff,'_','all',fsign,abs(Feh),asign,abs(Alphafe))).replace(' ', '0')
            print(outfilename)

            wvls = []; relfluxes = []; labels=[]
            for logg in loggarr:

                if grid_check:

                    #Make sure that the parameters are within the K08 grid range
                    in_grid = enforce_grid_check(teffarr[temp_index], logg,
                    feharr[feh_index], alphafearr[alphafe_index])

                    if in_grid:

                        wvl, relflux,_,_ = read_synth(Teff=teffarr[temp_index], Logg=logg,
                        Feh=feharr[feh_index], Alphafe=alphafearr[alphafe_index], gauss_sigma=0.,
                        data_path=data_path)

                        wvls += [wvl]; relfluxes += [relflux]

                        label = r'log $g$ = '+str(logg)
                        labels.append(label)

                else:

                    wvl, relflux,_,_ = read_synth(Teff=teffarr[temp_index], Logg=logg,
                    Feh=feharr[feh_index], Alphafe=alphafearr[alphafe_index], gauss_sigma=0.,
                    data_path=data_path)

                    wvls += [wvl]; relfluxes += [relflux]

                    label = r'log $g$ = '+str(logg)
                    labels.append(label)

        else:
            sys.stderr.write('Parameter '+kind+' not recognized')
            sys.exit()

        if not subset:
            outfilename += '_py'
            plot_synth.check_synth(wvls, relfluxes, title=full_title, plot_path='plots/',
            filename=outfilename, fmt='.png')
        else:
            plot_synth.mult_synth(wvls, relfluxes, title=full_title, plot_path='plots/',
            xrange=xrange, labels=labels, filename=outfilename)

    return

def find_incomplete_files(path='.', teff_dir='', logg_dir='', start=4100., sstop=6300.,
fullres=True):

    """
    For a specified directory, determine which files did not save properly,
    and have an inappropriate number of flux steps in the file, based on the
    user-specified wavelength range and step size for the synthesis

    Parameters
    ----------
    path: string: the path name to the directory which contains the relevant
                  subdirectories
    teff_dir: string: the effective temperature directory to look for files in
    logg_dir: string: the surface gravity directory to look for files in
    """

    #Formatting for path name
    if path[-1] != '/': path += '/'
    if teff_dir[-1] != '/': teff_dir += '/'
    if logg_dir[-1] != '/': logg_dir += '/'

    #Get all files in the directory
    full_path = path+teff_dir+logg_dir
    files = [f for f in os.listdir(full_path) if isfile(join(full_path, f))]

    #Make sure that we are only picking up gzipped files
    bin_gz_files = [f for f in files if f[-3:] == '.gz']
    complete = True

    #Now read in the files
    for file in bin_gz_files:
        wvl, relflux,_, _ = read_synth(filename=full_path+file, start=start, sstop=sstop,
        fullres=fullres)
        if len(relflux) != len(wvl):
            complete = False
            print(file+' length error len = '+str(len(relflux)))

    return complete

def read_files_in_dir(path='.', teff_dir='', logg_dir=''):

    """
    For a specificed directory, read in the files and determine the files that
    are missing and the associated parameters

    Parameters
    -----------
    path: string: the path name to the directory which contains the relevant
                  subdirectories
    teff_dir: string: the effective temperature directory to look for files in
    logg_dir: string: the surface gravity directory to look for files in

    Returns
    -------
    alpha_miss: array-like: values of [alpha/Fe] from atmosphere models for which
                the synthesis did not converge for a given teff_dir and logg_dir
    feh_miss: array-like: values of [Fe/H] from atmosphere models for which the
              synthesis did not converge for a given teff_dir and logg_dir
    """

    #Formatting for path name
    if path[-1] != '/': path += '/'
    if teff_dir[-1] != '/': teff_dir += '/'
    if logg_dir[-1] != '/': logg_dir += '/'

    #Get the actual files in the directory
    full_path = path+teff_dir+logg_dir
    files = [f for f in os.listdir(full_path) if isfile(join(full_path, f))]

    teff_dir = teff_dir[:-1]; logg_dir = logg_dir[:-1]

    feharr = np.around(np.arange(-5.0,0.1,0.1),2)
    alphafearr = np.around(np.arange(-0.8,1.3,0.1),2)

    #Construct the list of what files SHOULD be in the directory
    all_files = []; alpha_miss = []; feh_miss = []
    for feh in feharr:

        feh_str = np.round(feh*10.)
        if feh >= 0.: fsign = '_'
        else: fsign = '-'

        for alphafe in alphafearr:

            alphafe_str = np.round(alphafe*10.)
            if alphafe >= 0.: asign = '_'
            else: asign = '-'

            file_name = teff_dir+logg_dir+'f%s%2ia%s%2i.bin.gz'%(fsign,abs(feh_str),asign,abs(alphafe_str))
            file_name = file_name.replace(' ','0')

            if file_name not in files: alpha_miss.append(alphafe); feh_miss.append(feh)

            all_files.append(file_name)

    #missing_files = [f for f in all_files if (f not in files)]
    return feh_miss, alpha_miss

def check_if_converged(path='.', run_number=0, check_exist=False,
synth_dir='/panfs/ds08/hopkins/iescala/gridiepy/synths/'):

    """
    Read in the provided output file from a given job, and based on the printed
    statements regarding convergence (whether the maximum number of iterations when
    determining the source functions for the line and continuum flux was ever
    exceeded), determine which combinations of parameters are potentially
    problematic

    Parameters
    -----------
    path: string: the path name to the directory which contains the relevant
                  subdirectories
    run_number: integer: the number that points to the subdirectory containing
                         the run informatiion, e.g., run_0
    """

    #Formatting for path name
    if path[-1] != '/': path += '/'
    full_path = path+'run_'+str(run_number)+'/'
    os.chdir(full_path)

    niter_cont = []; niter_line = []; pfiles = []

    #Read the output files from the jobs and get the information on
    #which files exceeded the max number of iterations
    for file in glob.glob('*.pbs.o*'):

        with open(file, 'rb') as f:
            lines = f.readlines()
            f.close()

        for line in lines:
            if 'Convergence' in line:
                convg_line = line.strip()

                ncont = convg_line[22]; nline = convg_line[34]
                pfile = convg_line[36:53]

                if (int(ncont) > 0) or (int(nline) > 0):
                    niter_cont.append(ncont)
                    niter_line.append(nline)
                    pfiles.append(pfile)

    #Separate the filename into the respective parameters
    teffs = []; loggs = []; fehs = []; alphas = []
    for pfile in pfiles:

        teff = pfile[1:5]
        logg = pfile[7]+'.'+pfile[8]

        fehsign = pfile[10]
        feh = pfile[11]+'.'+pfile[12]
        if fehsign == '-': feh = fehsign+feh

        alphasign = pfile[14]
        alpha = pfile[15]+'.'+pfile[16]
        if alphasign == '-': alpha = alphasign+alpha

        teffs.append(teff)
        loggs.append(logg)
        fehs.append(feh)
        alphas.append(alpha)

    #Now, identify for each teff at which loggs there were convergence issues,
    #and for those loggs, determine which fehs were problematic, if any
    #issues where encountered in this run
    if len(pfiles) > 0:

        teff_where_convg = np.unique(teffs)
        loggs = np.array(loggs); fehs = np.array(fehs)
        #alphas = np.array(alphas)

        logg_where_convg = []; feh_where_convg = []
        #alpha_where_convg = []

        for teffi in teff_where_convg:

            where_teff = [i for i,teff in enumerate(teffs) if teff == teffi]
            where_teff = np.array(where_teff)
            logg_for_teff = np.unique(loggs[where_teff])

            feh_for_logg = []; alpha_for_feh = []
            for loggi in logg_for_teff:

                where_feh = [];
                for i in where_teff:
                    if loggs[i] == loggi: where_feh.append(i)
                feh_for_logg += [np.unique(fehs[where_feh]).tolist()]

                """
                for fehi in feh_for_logg:

                    where_alpha = []
                    for i in where_feh:
                        if fehs[i] == fehi: where_alpha.append(alphas[i])
                    alpha_for_feh += [np.unique(where_alpha).tolist()]
                """

            logg_where_convg += [logg_for_teff]
            feh_where_convg += [feh_for_logg]
            #alpha_where_convg += [alpha_for_feh]

        logg_where_convg = np.array(logg_where_convg).tolist()
        feh_where_convg = np.array(feh_where_convg).tolist()
        #alpha_where_convg = np.array(alpha_where_convg).tolist()

        #Lastly, check if all the files exist
        if check_exist:

            print('Checking if all flagged files exist...')
            for pfile in pfiles:
                teff_dir = pfile[:5]+'/'; logg_dir = pfile[5:9]+'/'
                filename = synth_dir+teff_dir+logg_dir+pfile+'.bin.gz'
                if not isfile(filename):
                    print(pfile+' does not exist')

        return teff_where_convg, logg_where_convg, feh_where_convg

    else: return [], [], []


def read_synth(Teff = np.nan, Logg = np.nan, Feh = np.nan, Alphafe = np.nan, fullres=True,\
    filename= '', gauss_sigma = 1, lambda_sigma = -1, data_path='', verbose=False,\
    start=4100., sstop=6300., file_ext = '.bin', title='', numiso=2):

    """
    Read the ".bin.gz" file containing the synthetic spectrum information
    that it output by running MOOGIE

    Parameters:
    ------------
    Teff: float: effective temperature (K) of synthetic spectrum
    Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
    Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
    Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
    fullres: boolean: if True, then use the unbinned, full-resolution version of the
                      synthetic spectrum
    filename: string: the filename of the desired synthetic spectrum. If the parameters
                       describing the synthetic spectrum are given, then a filename is
                       not necessary
    data_path: string: the path leading to the parent directory containing the synthetic
                       spectrum data
    verbose: boolean: if True, then print out statements during run time
    start: float: start wavelength of synthetic spectrum
    sstop: float: stop wavelength of synthetic spectrum
    file_ext: string: file extension of the filename to be read, default '.bin'

    Returns:
    ---------
    wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                whether it is full resolution or binned, and on stop and start wavelengths
    relflux: array: the normalized flux of the synthetic spectrum

    """

    if file_ext != '.bin': linestart = 3 + numiso
    else: linestart = 3

    #Determine which directory to point to (binned/full resolution)
    if file_ext == '.bin':
        if fullres == True:
            directory = 'synths/'
            step = 0.02
        else:
            directory = 'bin/'
            step = 0.14
    else:
        directory = ''
        step = 0.01

    #If given, determine the degree of gaussian smoothing of the spectrum
    if lambda_sigma > 0.:
        gauss_sigma = round(lambda_sigma/step)

    if filename == '':

        #Check if the parameters are specified, if the filename is not
        if np.all(np.isnan([Teff,Logg,Feh,Alphafe])) == True:
            print("Error: must define teff, logg, feh, and alphafe")
            return np.nan, np.nan, None, None

        path = data_path+directory #full path name

        #Redefine the parameters according to the file naming convention
        title, filename = construct_title_filename(Teff, Logg, Feh, Alphafe)

        if file_ext == '.bin': bin_gz_file = filename
        else: out_file = filename

        filename = path+filename

    #Otherwise, if a filename is specified
    else:
        if file_ext == '.bin': bin_gz_file = filename
        else: out_file = filename

    if verbose == True: print(filename)

    if file_ext == '.bin':

        #Open and read the contents of the gzipped binary file without directly
        #unzipping, for enhanced performance
        with gzip.open(filename, 'rb') as f:
            bstring = f.read()
            flux = np.fromstring(bstring, dtype=np.float32)

        #Alternative method of opening and reading in the file that requires
        #explicit unzipping and zipping
        """
        filename_decompress = filename[:-3] #get rid of the filename extension
        success = os.system('gunzip %s'%filename) #attempt to unzip the file

        if success != 0:
            if verbose==True:
                print "Error unzipping %s"%filename
            return np.nan, np.nan, None, None

        flux = np.fromfile(filename_decompress,'f4')
        success = os.system('gzip %s'%filename_decompress) #now zip the file back up

        if success != 0:
            if verbose==True:
                print "Error zipping %s"%filename
            return np.nan, np.nan
        """

    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
            f.close()
        flux = []
        for line in lines[linestart:]:
            i = 0
            while i < len(line)-2:
                el = float(line[i:i+7].strip())
                flux.append(el)
                i += 7

    wvl_range = np.arange(start, sstop+step, step)
    wvl = 0.5*(wvl_range[1:] + wvl_range[:-1])

    if gauss_sigma != 0.0:
        relflux = scipy.ndimage.filters.gaussian_filter(1.0 - flux, gauss_sigma)
    else:
        relflux = 1. - flux

    if file_ext == '.bin':
        return wvl, relflux, title, bin_gz_file[11:-7]
    else:
        return wvl, relflux, title, out_file[:-5]

def construct_title_filename(Teff=np.nan, Logg=np.nan, Feh=np.nan, Alphafe=np.nan,
file_ext='.bin', interp=False, Dlam=np.nan):

    #Redefine the parameters according to the file naming convention
    if not interp:
        teff = round(Teff/100.)*100
        logg = round(Logg*10.)
        feh = round(Feh*10.)
        alphafe = round(Alphafe*10.)
    else:
        teff = np.round(Teff, decimals=0)
        logg = np.round(Logg*10., decimals=2)
        feh = np.round(Feh*10., decimals=2)
        alphafe = np.round(Alphafe*10., decimals=2)
        dlam = np.round(Dlam*10.,decimals=2)

    if logg >= 0.:
        gsign = '_'
    else: gsign = '-'

    if feh >= 0.:
        fsign = '_'
    else: fsign = '-'

    if alphafe >= 0.:
        asign = '_'
    else: asign = '-'

    title = r'T$_{eff}$=%g, log(g)=%g, [Fe/H]=%g, [$\alpha$/Fe]=%g,\
    $\Delta\lambda$=%g'%(np.round(Teff, decimals=0), np.round(Logg, decimals=2),\
    np.round(Feh, decimals=2), np.round(Alphafe, decimals=2), np.round(Dlam, decimals=2))

    if file_ext == '.bin':

        bin_gz_file = "t%2i/g%s%2i/t%2ig%s%2if%s%2ia%s%2i.bin.gz"%(teff,gsign,abs(logg),teff,gsign,abs(logg),fsign,abs(feh),asign,abs(alphafe))
        bin_gz_file = bin_gz_file.replace(" ","0")
        filename = bin_gz_file
    else:
        out_file = "t%2ig%s%2if%s%2ia%s%2i.out2"%(teff,gsign,abs(logg),fsign,abs(feh),asign,abs(alphafe))
        out_file = out_file.replace(" ","0")
        filename = out_file

    return title, filename

def read_interp_synth(teff=np.nan, logg=np.nan, feh=np.nan, alphafe=np.nan,
fullres=False, data_path='', start=4100., sstop=6300., npar=4, gauss_sigma=0., hash=None):

    """
    Construct a synthetic spectrum in between grid points based on linear interpolation
    of synthetic spectra in the MOOGIE grid

    Parameters:
    -----------
    Teff: float: effective temperature (K) of synthetic spectrum
    Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
    Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
    Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
    fullres: boolean: if True, then use the unbinned, full-resolution version of the
                      synthetic spectrum
    filename: string: the filename of the desired synthetic spectrum. If the parameters
                       describing the synthetic spectrum are given, then a filename is
                       not necessary
    data_path: string: the path leading to the parent directory containing the synthetic
                       spectrum data
    verbose: boolean: if True, then print out statements during run time
    start: float: start wavelength of synthetic spectrum
    sstop: float: stop wavelength of synthetic spectrum
    file_ext: string: file extension of the filename to be read, default '.bin'
    npar: integer: number of parameters used to describe a synthetic spectrum
    hash: dict, optional: a dictionary to use to store memory concerning which synthetic
          spectra have been read in. Should be initliazed externally as an empty dict.

    Returns:
    --------
    wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                whether it is full resolution or binned, and on stop and start wavelengths
    relflux: array: the normalized flux of the synthetic spectrum
    """

    #Define the points of the 4D grid
    teff_arr = np.arange(3500., 5600., 100.).tolist() + np.arange(5600., 8200., 200.).tolist()
    teff_arr = np.round(np.array(teff_arr), decimals=0)

    logg_arr = np.round(np.arange(0., 5.5, 0.5), decimals=1)
    feh_arr = np.round(np.arange(-5., 0.1, 0.1), decimals=2)
    alphafe_arr = np.round(np.arange(-0.8, 1.3, 0.1), decimals=2)
    alphafe_arr[8] = 0.

    #First check that given synthetic spectrum parameters are in range
    #If not in range, throw an error
    #Consider implementing a random selection of nearby grid points if the parameters
    #do go out of range, for values near the edge of the grid

    #in_grid, key = enforce_grid_check(teff, logg, feh, alphafe)
    in_grid,_ = enforce_grid_check(teff, logg, feh, alphafe)
    if not in_grid: return

    """
    if not in_grid:

        sys.stderr.write('\nInput parameters {} = {} out of grid range\n'.format(['Teff',\
        'logg', '[Fe/H]', '[alpha/Fe]'], [teff, logg, feh, alphafe]))

        while not in_grid:

            if key == 0: teff = np.random.uniform(teff_arr[0], teff_arr[-1])
            if key == 1: logg = np.random.uniform(logg_arr[0], logg_arr[-1])
            if key == 2: feh = np.random.uniform(feh_arr[0], feh_arr[-1])
            if key == 3: alphafe = np.random.uniform(alphafe_arr[0], alphafe_arr[-1])

            sys.stderr.write('Selecting new parameters {} = {}\n'.format(['Teff', 'Logg',\
            '[Fe/H]', '[alpha/Fe]'], [teff, logg, feh, alphafe]))

            in_grid, key = enforce_grid_check(teff, logg, feh, alphafe)
    """

    params = np.array([teff, logg, feh, alphafe])
    params_grid = np.array([teff_arr, logg_arr, feh_arr, alphafe_arr])

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

    #Determine the number of pixels in a synthetic spectrum based on whether
    #the spectrum is binned or unbinned
    if fullres: step = 0.02
    else: step = 0.14

    #Calculate the number of pixels in the synthetic spectrum, and initialize the
    #interpolated synthetic spectrum array
    npixels = len(np.arange(start, sstop, step))
    synth_interp = np.zeros(npixels)

    #Function for loading a specified synthetic spectrum
    def load_synth(p):

        teffi, loggi, fehi, alphafei = p

        if hash is not None:

            #First construct the filename corresponding to the parameters, to use for
            #testing whether we should read in the specified synthetic spectrum

            _,filename = construct_title_filename(Teff=teff_arr[teffi],
            Logg=logg_arr[loggi], Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei])
            key = filename[11:-7]

            if key not in hash.keys():

                _,synthi,_,_ = read_synth(Teff=teff_arr[teffi], Logg=logg_arr[loggi],
                Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei], fullres=fullres,
                verbose=False, gauss_sigma=gauss_sigma, data_path=data_path)

                hash[key] = synthi

            #If the key is already present in the hash table, then find it and load the data
            else: synthi = hash[key]

        else:
            _,synthi,_,_ = read_synth(Teff=teff_arr[teffi], Logg=logg_arr[loggi],
            Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei], fullres=fullres,
            verbose=False, gauss_sigma=gauss_sigma, data_path=data_path)

        return synthi

    """
    #Function for calculating the product of the prefactors
    def dprod(q):
        ds0, ds1, ds2, ds3 = q
        return ds0*ds1*ds2*ds3

    #Load each nearby synthetic spectrum on the grid to linearly interpolate
    #to calculate the interpolated synthetic spectrum

    params_tup = list(itertools.product(*iparams))
    ds_tup = list(itertools.product(*ds))

    synthis = map(load_synth, params_tup)
    dprods = map(dprod, ds_tup)

    for i in range(len(dprods)):
        for m in range(npixels):
            synth_interp[m] += dprods[i]*synthis[i][m]
    """

    #Load each nearby synthetic spectrum on the grid to linearly interpolate
    #to calculate the interpolated synthetic spectrum
    for i in range(nspecs[0]):
        for j in range(nspecs[1]):
            for k in range(nspecs[2]):
                for l in range(nspecs[3]):

                    p = [iparams[0][i], iparams[1][j], iparams[2][k], iparams[3][l]]
                    synthi = load_synth(p)

                    """
                    _,synthi,_,_ = read_synth(Teff=teff_arr[iparams[0][i]],
                    Logg=logg_arr[iparams[1][j]], Feh = feh_arr[iparams[2][k]],
                    Alphafe = alphafe_arr[iparams[3][l]], fullres=fullres, verbose=False,
                    gauss_sigma=0., data_path=data_path)
                    """

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
