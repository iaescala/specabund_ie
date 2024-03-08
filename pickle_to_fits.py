"""
@author: Ivanna Escala 2018-2022
"""

import pickle
import astropy.table as table
import numpy as np
import gzip
import glob2 as glob
import re

def pickle_to_fits(slitmask_name, save_path='', dlamfix=True, longslit=False,
                   resscale=False):
    """
    Program to combine individual pickle files for a given slitmask run into
    a single FITS file. Run following completion of specabundie routine.

    Parameters
    ----------
    slitmask_name: str: name of the slitmask
    save_path: optional, str: directory to save the output FITS file
    dlamfix: optional, boolean, default True. Whether the version of
    specabundie in which dlam was fixed was run.
    longslit: optional, boolean, default False. Whether the run
    was for longlist observations instead of a slitmask.
    resscale, optional, boolean, default False. Whether the version
    of specabundie used to estimate resscale was run.

    Returns
    -------
    Saves table with information from the specabund run into FITS file.
    """

    if dlamfix:
        specabund_files = glob.glob(save_path+slitmask_name+'/specabund_*_dlamfix.pkl')
    elif resscale:
        specabund_files = glob.glob(save_path+slitmask_name+'/specabund_*_resscale.pkl')
    else:
        if longslit:
            specabund_files = glob.glob(save_path+slitmask_name+'/specabund_'+slitmask_name+'*.pkl')
        else:
            specabund_files = glob.glob(save_path+slitmask_name+'_contiteralpha_v2/specabund_'+slitmask_name+'.[0-9][0-9][0-9].pkl')

    #Load in the information from a specabund pickle file
    with open(specabund_files[0], 'rb') as f:
        specabund = pickle.load(f)

    keys = specabund.keys()
    t = table.Table()

    slits = []; object_ids = []
    for i,key in enumerate(keys):

        #if key in ['cafe', 'mgfe', 'mgfeerr', 'cafeerr', 'chicafe', 'chimgfe', 'cafenpix', 'mgfenpix']:
        #    continue

        column = []
        for filename in specabund_files:

            if longslit and 'objname' not in keys and i == 0:
                object_id = re.search('\.2M\d{7}.\d{5}\_', filename).group()[1:-1]
                object_ids.append(object_id)

            if i == 0 and not longslit:
                if dlamfix: slit_str = re.search('\.\d\d\d\_', filename).group()[1:-1]
                elif resscale: slit_str = re.search('\.\d\d\d\_', filename).group()[1:-1]
                else: slit_str = re.search('\.\d\d\d\.', filename).group()[1:-1]
                slit_num = int(slit_str)
                slits.append(slit_num)

            #Load in the information from the specabund pickle file
            with open(filename, 'rb') as f:
                specabund = pickle.load(f)

            column.append(specabund[key])

        column_dtype = np.asarray(specabund[key]).dtype
        arr = np.array(column)

        try:
            column = table.Column(name=key, data=arr, dtype=column_dtype)

        except:

            lens = []
            for arr_i in arr:
                lens.append(np.shape(arr_i)[0])
            lens = np.array(lens)
            max_length = lens.max()

            new_arr = np.full((len(arr), max_length), np.nan)
            for i,arr_i in enumerate(arr):
                new_arr[i][:len(arr_i)] = arr_i

            column = table.Column(name=key, data=new_arr, dtype=column_dtype)

        t.add_column(column)

    if not longslit and 'slit' not in keys:
        slits = np.array(slits)
        t.add_column(table.Column(name='slit', data=slits))
    if longslit and 'objname' not in keys:
        object_ids = np.array(object_ids)
        t.add_column(table.Column(name='objname', data=object_ids))

    if not longslit:
        t.sort('slit')
    if longslit:
        t.sort('objname')

    if dlamfix:
        t.write(save_path+slitmask_name+'/specabundie_'+slitmask_name+'_dlamfix.fits', format='fits')
    elif resscale:
        t.write(save_path+slitmask_name+'/specabundie_'+slitmask_name+'_resscale.fits', format='fits')
    else:
         t.write(save_path+slitmask_name+'/specabundie_'+slitmask_name+'.fits', format='fits')

    return
