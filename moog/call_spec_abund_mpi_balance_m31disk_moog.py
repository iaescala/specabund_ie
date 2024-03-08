from astropy.io import fits
import spec_abund_ie_dlamfix
from mpi4py import MPI
import numpy as np
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

##NOTE WHICH GRID IS BEING USED
synth_path_blue = '/central/groups/carnegie_poc/iescala/gridie/'
synth_path_red = '/central/groups/carnegie_poc/iescala/grid7/'

mask_path = '/home/iescala/specabundie-master/mask/'
save_path = '/central/groups/carnegie_poc/iescala/specabund/'
telluric_path = '/home/iescala/specabundie-master/'

m31disk_path = '/central/groups/carnegie_poc/iescala/moogify/m31disk'

##############################
##### Get the arguments ######
##############################

mask_ranges = None

args_list = sys.argv[1:]

options = "R:s:rw"

long_options = ["region=", "subregon=", "replace", "wgood"]

try:

   args, vals = getopt.getopt(args_list, options, long_options)
   for arg_i, val_i in args:

      if arg_i in ("-R", "--region"):
         region = int(val_i)

      if arg_i in ("-s", "--subregion"):
          subregion = int(val_i)

      if arg_i in ("-w", "--wgood"):
          wgood = False
      else:
          wgood = True

      if arg_i in ("-r", "--replace"):
          replace = True
      else:
          replace = False

except getopt.error as err:
   print(str(err))

###########################################################

m31disk = fits.getdata(f'{m31disk_path}/mct_all_airmass_photpar_photfix_dustir.fits.gz')

is_good = m31disk['is_member'] & m31disk['is_rgb'] &\
         ~m31disk['duplicate'] & ~np.isnan(m31disk['f814w_v2'])

is_in_subregion = (m31disk['region'] == region) & (m31disk['subregion'] == subregion)

w = is_good & is_in_subregion

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
        niter_all = np.arange(0, len(m31disk['objname']))[w]
        print(f"root created {len(niter_all)} tasks", flush=True)
        load_balance(niter_all, comm, size)
    #all other processors do the jobs
    else:
        do_work(comm)

##################################################

#### Perform the fitting to measure abundances from the spectrum ####

def measure_abundance(index):

    try:
            specabund = spec_abund_ie_dlamfix.spec_abund(slit_num=index,
                    synth_path_red=synth_path_red, mask_path=mask_path, save_path=save_path,
                    replace=replace, telluric_path=telluric_path, m31disk=m31disk,
                    synth_path_blue=synth_path_blue)

            print(specabund)

    except:
        pass

    return

main()
