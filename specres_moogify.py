"""
@author: Ivanna Escala 2019-2022
"""

from astropy.table import Column,Table
from subprocess import check_call
from shutil import copyfile
from astropy.io import fits
import numpy as np
import data_redux
import glob
import os

def specres_moogify(slitmask, stacked=True, fit_type='ml',
                    slitmask_path='', moogify_path='',
                    data_path='', dlam_def=1.2):

    """
    Program to combine individual pickle files for a given slitmask run into
    a single FITS file. Run following completion of specabundie routine.

    Parameters
    ----------
    slitmask_name: str: name of the slitmask
    stacked: optional, boolean, default True. Whether the slitmask
    was observed over multiple nights, i.e. requiring stacking observations
    for a single object.
    fit_type: optional, string, default 'ml'. The fitting type that
    was used to measure the spectral resolution. Other option is 'mcmc'.
    slitmask_path: optional, string: The path to where the observational
    slitmask data is located.
    moogify_path: optional, string. The path to where the moogify file
    is located.
    data_path: optional, string. The path to where the output of the
    resolution code is located.
    dlam_def: optional, float, default 1.2 (600ZD). The value of dlam
    to adopt in the case that the spectral resolution measurement has
    failed for an individual star.

    Returns
    -------
    Version of the moogify file with a column for dlam. 
    """

    slit_strs = data_redux.get_slits(slitmask, slitmask_path=slitmask_path)

    def gaussian(x, mu, sigma, height):
        return height * np.exp(-0.5*(x-mu)**2/sigma**2) + 1.


    dlam = []
    for slit in slit_strs:

        print(slit)

        file = glob.glob(slitmask_path+slitmask+'/spec1d.'+slitmask+'*.'+slit+'.*.fits.gz')[0]

        hdu = fits.open(file)
        blue = hdu[1].data
        red = hdu[2].data

        hdu.close()

        x = np.concatenate((blue['lambda'][0], red['lambda'][0]), axis=0)

        try:

            if not stacked:

                if fit_type == 'mcmc':
                    res = np.load(data_path+slitmask+'/'+slit+'/'+slit+'.deltaLambdaFitMC.npy')

                if fit_type == 'ml':
                    res = np.load(data_path+slitmask+'/'+slit+'/'+slit+'.deltaLambdaFit.npy')[0]

                y = np.poly1d(res[:,0][:-1])(x)

            else:

                yavg = np.zeros(len(x)); counter = 0

                for dir in os.listdir(data_path+slitmask):
                    for subdir in os.listdir(data_path+slitmask+'/'+dir):

                        if fit_type == 'mcmc':
                            file = data_path+slitmask+'/'+dir+'/'+subdir+'/'+slit+'/'+slit+'.deltaLambdaFitMC.npy'
                            res = np.load(file)

                        if fit_type == 'ml':
                            file = data_path+slitmask+'/'+dir+'/'+subdir+'/'+slit+'/'+slit+'.deltaLambdaFit.npy'
                            res = np.load(file)[0]

                        y = np.poly1d(res[:,0][:-1])(x)

                        yavg += y
                        counter += 1

                yavg /= counter

        except IOError:

            #In the case that the spectral resolution could not be determined from
            #the arc and sky lines, assume a constant delta Lambda

            y = np.full(len(x), dlam_def)
            if stacked: yavg = y

        if not stacked: dlam.append(y)
        else: dlam.append(yavg)

    if not os.path.exists(moogify_path+slitmask+'/no_dlam/'):
        os.mkdir(moogify_path+slitmask+'/no_dlam/')

    if not os.path.exists(moogify_path+slitmask+'/no_dlam/moogify.fits.gz'):

        copyfile(moogify_path+slitmask+'/moogify.fits.gz',
                 moogify_path+slitmask+'/no_dlam/moogify.fits.gz')

    moog = fits.getdata(moogify_path+slitmask+'/no_dlam/moogify.fits.gz')

    slits = np.array([int(s) for s in slit_strs])

    arr = []
    for i,s in enumerate(moog['slit']):
        w = np.where(slits == s)[0]
        if len(w) > 0:
            arr.append( np.array(dlam)[w][0] )
        else:
            arr.append( np.full(8192, np.nan) )
    arr = np.array(arr)

    table = Table.read(moogify_path+slitmask+'/no_dlam/moogify.fits.gz')

    column = Column(arr, name='dlam')
    table.add_column(column)

    table.write(moogify_path+slitmask+'/moogify.fits', format='fits', overwrite=True)
    if os.path.exists(moogify_path+slitmask+'/moogify.fits.gz'):
        os.remove(moogify_path+slitmask+'/moogify.fits.gz')
    check_call(['gzip', moogify_path+slitmask+'/moogify.fits'])

    return
