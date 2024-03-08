from astropy.io import fits
#import spec_abund_ie_dlamfix_ts
import spec_abund_ie_dlamfix_ts_cub as spec_abund_ie_dlamfix_ts
import spec_abund_ie_resscale
import spec_abund_ie
from mpi4py import MPI
import numpy as np
import data_redux
import pickle
import getopt, sys

#for dynamic load balancing
WORKTAG = 1
DIETAG = 2

#########################################################
# PATH NAMES #
#########################################################

#Define directories
slitmask_path = '/central/groups/carnegie_poc/iescala/deimos/'
moogify_path = '/central/groups/carnegie_poc/iescala/moogify/'

##NOTE WHICH GRID IS BEING USED
synth_path = '/central/groups/carnegie_poc/iescala/tsgrid/'
#synth_path = '/central/groups/carnegie_poc/iescala/tsgrid_ch/'

mask_path = '/home/iescala/specabundie-master/mask/'
save_path = '/central/groups/carnegie_poc/iescala/specabund/'
telluric_path = '/home/iescala/specabundie-master/'

##############################
##### Get the arguments ######
##############################

args_list = sys.argv[1:]

options = "s:wrdzg:"

long_options = ["slitmask=", "wgood", "replace", "dlamfix", "resscale", "grating="]

try:

   args, vals = getopt.getopt(args_list, options, long_options)
   for arg_i, val_i in args:

      if arg_i in ("-s", "--slitmask"):
         slitmask_name = val_i

      if arg_i in ("-g", "--grating"):
          grating = val_i

      if arg_i in ("-w", "--wgood"):
          wgood = False
      else:
          wgood = True

      if arg_i in ("-r", "--replace"):
          replace = True
      else:
          replace = False

      if arg_i in ("-d", "--dlamfix"):
          dlamfix = False
      else:
          dlamfix = True

      if arg_i in ("-z", "--resscale"):
          resscale = True
      else:
          resscale = False

except getopt.error as err:
   print(str(err))

###################################
###### Boolean parameters #######
##################################

#dlamfix = True
#resscale = False

#wgood = True
#replace = False

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

#slitmask_name = 'H'; grating = '600ZD'
#slitmask_name = 'S'; grating = '600ZD'
#slitmask_name = 'D'; grating = '600ZD'

#slitmask_name = 'f130_2'; grating = '600ZD'

#slitmask_name = 'f130_1'; grating = '600ZD'
#slitmask_name = 'f123_1a'; grating = '1200G'

## 600ZD globular cluster masks

#slitmask_name = 'n2419c'; grating = '600ZD'
#slitmask_name = '1904l2'; grating = '600ZD'
#slitmask_name = 'n6864a'; grating = '600ZD'
#slitmask_name = '7078l1'; grating = '600ZD'
#slitmask_name = 'n6341b'; grating = '600ZD'

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
                 'S': 0.823,
                 'D': 0.861,
                 'f130_2': 0.844,
                 '1904l2': 0.865,
                 'n6864a': 0.918,
                 'n2419c': 0.868,
                 'n6341b': 0.901,
                 '7078l1': 0.881,
                 'm31d1': 0.839,
                 'm31d2': 0.877
                 }

###########################################################

def load_balance(tasks, comm, size): #tasks = slit str

    ntasks = len(tasks)
    workcount = 0
    recvcount = 0
    print("root sending first tasks", flush=True)

    #send out the first tups of stellar labels to each processor
    #rank = 0 is managing jobs
    for rank in range(1, size):
        if workcount < ntasks:
            work = tasks[workcount] #send one job to each processor
            comm.send(work, dest=rank, tag=WORKTAG)
            workcount += 1
            print(f"root sent {work} to processor {rank}", flush=True)

    ##while there is still work, receive a result from a working processor
    #which also signals they would like some new work
    while workcount < ntasks:

        #receive next finished result
        stat = MPI.Status()
        task = comm.recv(source=MPI.ANY_SOURCE, status=stat)
        recvcount += 1
        rank_id = stat.Get_source()
        print(f"root received {task} from processor {rank_id}", flush=True)

        #send next work
        work = tasks[workcount]
        comm.send(work, dest=rank_id, tag=WORKTAG)
        workcount += 1
        print(f"root send {work} to processor {rank_id}", flush=True)

    #receive results for outstanding work requests
    while recvcount < ntasks:
        stat = MPI.Status()
        task = comm.recv(source=MPI.ANY_SOURCE, status=stat)
        recvcount += 1
        rank_id = stat.Get_source()
        print(f"end: root received {task} from {rank_id}", flush=True)

    #tell all processors to stop!
    for rank in range(1, size):
        comm.send(-1, dest=rank, tag=DIETAG)

def do_work(comm):
    #keep receiving messages and do work, unless tagged to die

    while True:
        stat = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        print(f"processor {comm.Get_rank()} got {task}", flush=True)
        if stat.Get_tag() == DIETAG:
            print(f"processor {comm.Get_rank()} dying", flush=True)
            return
        else:
            ## do the work!!
            measure_abundance(task)

            #indicate done with work by sending to root
            comm.send(task, dest=0)

##################################################

def main():

    #Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #The root processor manages the assignment of jobs
    if rank == 0:
        slit_strs_all = data_redux.get_slits(slitmask_name, slitmask_path=slitmask_path)
        print(f"root created {len(slit_strs_all)} tasks", flush=True)
        load_balance(slit_strs_all, comm, size)
    #all other processors do the jobs
    else:
        do_work(comm)

##################################################

#### Perform the fitting to measure abundances from the spectrum ####

def measure_abundance(slit_str):

    print(slit_str)

    if grating == '600ZD':

        if dlamfix:

            print('STARTING MEASUREMENT FOR FIXED DELTA LAMBDA')

            if slitmask_name in resscale_dict.keys():

                moog = fits.getdata(moogify_path+slitmask_name+'/moogify.fits.gz')
                dlam = moog['dlam']; slit = moog['slit']

                try: dlamarr = dlam[slit == int(slit_str)][0]
                except: return

                dlamscale = np.array([resscale_dict[slitmask_name]*arr if len(np.unique(arr)) > 1\
                                      else arr for arr in dlamarr])

                try: specabund = spec_abund_ie_dlamfix_ts.spec_abund(slitmask_name, slit_str=slit_str,
                synth_path=synth_path, slitmask_path=slitmask_path,
                moogify_path=moogify_path, mask_path=mask_path, save_path=save_path,
                replace=replace, grating=grating, dlam=dlamscale,
                telluric_path=telluric_path,
                wgood=wgood, mask_ranges=mask_ranges)

                except ValueError: return

            else:

                specabund = spec_abund_ie_dlamfix_ts.spec_abund(slitmask_name, slit_str=slit_str,
                synth_path=synth_path, slitmask_path=slitmask_path,
                moogify_path=moogify_path, mask_path=mask_path,
                save_path=save_path, replace=replace,
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
            except: return
            dlamscale = resscale_dict[slitmask_name]*dlamarr

            specabund = spec_abund_ie_dlamfix_ts.spec_abund(slitmask_name, slit_str=slit_str,
            synth_path=synth_path, slitmask_path=slitmask_path,
            moogify_path=moogify_path, mask_path=mask_path, save_path=save_path,
            replace=replace, grating=grating, dlam=dlamscale,
            telluric_path=slitmask_path)

        else:

            specabund = spec_abund_ie_dlamfix_ts.spec_abund(slitmask_name, slit_str=slit_str,
            synth_path=synth_path, slitmask_path=slitmask_path,
            moogify_path=moogify_path, mask_path=mask_path,
            save_path=save_path, replace=replace,
            grating=grating, dlam=0.5, telluric_path=telluric_path,
            mask_ranges=mask_ranges)

    print(specabund)
    return

main()
