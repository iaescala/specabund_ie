from astropy.io import fits
import spec_abund_ie_dlamfix
import spec_abund_ie_resscale
import spec_abund_ie
from mpi4py import MPI
import numpy as np
import data_redux
import pickle

#########################################################
# PATH NAMES #
#########################################################

#Define directories
slitmask_path = '/central/groups/carnegie_poc/iescala/deimos/'
moogify_path = '/central/groups/carnegie_poc/iescala/moogify/'
synth_path_blue = '/central/groups/carnegie_poc/iescala/gridie/'
synth_path_red = '/central/groups/carnegie_poc/iescala/grid7/'
mask_path = '/home/iescala/specabundie-master/mask/'
save_path = '/central/groups/carnegie_poc/iescala/specabund/'
telluric_path = '/home/iescala/specabundie-master/'

###################################
###### Boolean parameters #######
##################################

dlamfix = True
resscale = False

wgood = True
replace = False

#######################################################

if resscale:
    dlamfix = False

if dlamfix:
    print('FIXING DELTA LAMBDA')
if resscale:
    print('DETERMINING THE RESOLUTION SCALE')
if replace:
    print('REPLACING PREVIOUS MEASUREMENTS')

######################################################
# Indicate desired slitmask and grating for analysis #
######################################################

## M31 mask examples

slitmask_name = 'H'; grating = '600ZD'

#slitmask_name = 'f130_1'; grating = '600ZD'
#slitmask_name = 'f123_1a'; grating = '1200G'

################################################
############## MASKED REGIONS ##################

#masking the Mg b 5167 line
#mask_ranges = [[5165, 5169]]
mask_ranges=None

################################################

#example resolution scale input
resscale_dict = {'f123_1a': 1.,
                 'f130_1': 0.822,
                 'H': 0.846,
                 }

#################################################
################################################

#Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Separate a list l into n parts of approximately equal length
def divide_list(l, n):

    avg = len(l)/float(n)
    out = []; last = 0.

    while last < len(l):
        out.append(l[int(last):int(last+avg)])
        last += avg

    if len(out) > n:
        ndiff = len(out) - n
        merge_el = []
        for i in range(ndiff+1): merge_el += out[-(ndiff+i)]
        out = [out[i] for i in range(len(out)-ndiff-1)]
        out.append(merge_el[::-1])

    return out

#If at root process, identify all slits for a given slitmask, and then based on the
#total number of processors, break up the list of slits into a separate list for each
#processor
if rank == 0:
    slit_strs_all = data_redux.get_slits(slitmask_name, slitmask_path=slitmask_path)
    slit_strs = divide_list(slit_strs_all, size)
    assert len(slit_strs) == size
else:
    slit_strs = None

#Send the information regarding the slit numbers that each processor should analyze
#to each individual processor, where the root process is rank 0
slit_strs = comm.scatter(slit_strs, root=0)

#### Perform the fitting to measure abundances from the spectrum ####

for slit_str in slit_strs:

    print(slit_str)

    if grating == '600ZD':

        if dlamfix:

            print('STARTING MEASUREMENT FOR FIXED DELTA LAMBDA')

            if slitmask_name in resscale_dict.keys():

                moog = fits.getdata(moogify_path+slitmask_name+'/moogify.fits.gz')
                dlam = moog['dlam']; slit = moog['slit']

                try: dlamarr = dlam[slit == int(slit_str)][0]
                except: continue

                dlamscale = np.array([resscale_dict[slitmask_name]*arr if len(np.unique(arr)) > 1\
                                      else arr for arr in dlamarr])

                specabund = spec_abund_ie_dlamfix.spec_abund(slitmask_name, slit_str=slit_str,
                synth_path_blue=synth_path_blue, slitmask_path=slitmask_path,
                moogify_path=moogify_path, mask_path=mask_path, save_path=save_path,
                replace=replace, synth_path_red=synth_path_red, grating=grating, dlam=dlamscale,
                telluric_path=telluric_path,
                wgood=wgood, mask_ranges=mask_ranges)

            else:

                specabund = spec_abund_ie_dlamfix.spec_abund(slitmask_name, slit_str=slit_str,
                synth_path_blue=synth_path_blue, slitmask_path=slitmask_path,
                moogify_path=moogify_path, mask_path=mask_path,
                save_path=save_path, replace=replace, synth_path_red=synth_path_red,
                grating=grating, dlam=1.2, telluric_path=slitmask_path,
                mask_ranges=mask_ranges)

        if resscale:

            print('STARTING MEASUREMENT FOR THE RESOLUTION SCALE')

            specabund = spec_abund_ie_resscale.spec_abund(slitmask_name, slit_str=slit_str,
            synth_path_blue=synth_path_blue, slitmask_path=slitmask_path, moogify_path=moogify_path,
            mask_path=mask_path, save_path=save_path, replace=replace,
            synth_path_red=synth_path_red, grating=grating, telluric_path=slitmask_path,
            wgood=wgood, mask_ranges=mask_ranges)

        if not dlamfix and not resscale:

            print('STARTING MEASUREMENT WITH VARIABLE DELTA LAMBDA')

            specabund = spec_abund_ie.spec_abund(slitmask_name, slit_str=slit_str,
            synth_path_blue=synth_path_blue, slitmask_path=slitmask_path, moogify_path=moogify_path,
            mask_path=mask_path, save_path=save_path, replace=replace,
            synth_path_red=synth_path_red, grating=grating, dlam_bounds=[1.0, 1.4],
            telluric_path=slitmask_path,
            mask_ranges=mask_ranges)


    if grating == '1200G':

        if slitmask_name in resscale_dict.keys():

            moog = fits.open(moogify_path+slitmask_name+'/moogify.fits.gz')
            dlam = moog[1].data['dlam']; slit = moog[1].data['slit']
            moog.close()

            try: dlamarr = dlam[slit == int(slit_str)][0]
            except: continue
            dlamscale = resscale_dict[slitmask_name]*dlamarr

            specabund = spec_abund_ie_dlamfix.spec_abund(slitmask_name, slit_str=slit_str,
            synth_path_blue=synth_path_blue, slitmask_path=slitmask_path,
            moogify_path=moogify_path, mask_path=mask_path, save_path=save_path,
            replace=replace, synth_path_red=synth_path_red, grating=grating, dlam=dlamscale,
            telluric_path=slitmask_path)

        else:

            specabund = spec_abund_ie_dlamfix.spec_abund(slitmask_name, slit_str=slit_str,
            synth_path_blue=synth_path_blue, slitmask_path=slitmask_path,
            moogify_path=moogify_path, mask_path=mask_path,
            save_path=save_path, replace=replace, synth_path_red=synth_path_red,
            grating=grating, dlam=0.5, telluric_path=telluric_path,
            mask_ranges=mask_ranges)

    print(specabund)
