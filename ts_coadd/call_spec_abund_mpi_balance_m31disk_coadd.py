#from astropy.io import fits
from astropy.table import Table
from spec_abund_ie_coadd_ts import spec_abund_coadd
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
synth_path = '/central/groups/carnegie_poc/iescala/tsgrid/'

mask_path = '/home/iescala/specabundie-master/mask/'
save_path = '/central/groups/carnegie_poc/iescala/specabund/'
telluric_path = '/home/iescala/specabundie-master/'

m31disk_path = '/central/groups/carnegie_poc/iescala/moogify/m31disk'
m31disk = Table.read(f'{m31disk_path}/mct_all_airmass_photpar_photfix_dustir.fits.gz')

#Reduce the data table size
columns_keep = ['spec', 'ivar', 'lambda', 'wchip', 'airmass', 'grating', 'objname',
                'slitmask', 'slit', 'teffphot', 'teffphoterr', 'loggphot', 'loggphoterr']
m31disk.keep_columns(columns_keep)

##############################
##### Get the arguments ######
##############################

wgood = False
replace = False
mask_ranges = None

#####################################################
###### Load the slit list in for the coadds #########
#####################################################

coadd_path = '/central/groups/carnegie_poc/iescala/coadd'

grating = '600ZD'

slit_list_all = np.load(f'{coadd_path}/R34_disk_{grating}.npy', allow_pickle=True)

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
        print(f"root created {len(slit_list_all)} tasks", flush=True)
        load_balance(slit_list_all, comm, size)
    #all other processors do the jobs
    else:
        do_work(comm)

##################################################

#### Perform the fitting to measure abundances from the spectrum ####

def measure_abundance(slit_list):

    try:
        if grating == '1200G':
            dlam = 0.45
        if grating == '600ZD':
            dlam = 1.2

        specabund = spec_abund_coadd(slit_list_str=slit_list, synth_path=synth_path,
        mask_path=mask_path, save_path=save_path, replace=replace,
        telluric_path=telluric_path, m31disk=m31disk, spec1dfile=False,
        grating=grating, dlam=dlam)

        print(specabund)


    except:
        print('Abundance measurement FAILED')

    return

main()
