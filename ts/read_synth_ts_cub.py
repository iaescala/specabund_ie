import numpy as np
from numba import jit
import itertools
import gzip

def read_synth(Teff = np.nan, Logg = np.nan, Feh = np.nan, Alphafe = np.nan, data_path=''):

    """
    Read the ".bin.gz" file containing the synthetic spectrum information
    that it output by running MOOGIE

    Parameters:
    ------------
    Teff: float: effective temperature (K) of synthetic spectrum
    Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
    Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
    Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
    data_path: string: the path leading to the parent directory containing the synthetic
                       spectrum data

    Returns:
    ---------
    wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                whether it is full resolution or binned, and on stop and start wavelengths
    flux: array: the normalized flux of the synthetic spectrum

    """

    data_path = f"{data_path}synths"

    if Logg <= 3.0:
        model_vt = 2.
        geo = 'sph'
    else:
        model_vt = 1.
        geo = 'pp'

    #Redefine the parameters according to the file naming convention
    modelfn = construct_model_filename(Teff, Logg, Feh, Alphafe,
    path=data_path, grid_type='apogee', micro_turb_vel=model_vt, geometry=geo)

    #Open and read the contents of the gzipped binary file without directly
    #unzipping, for enhanced performance
    with gzip.open(modelfn, 'rb') as f:
        bstring = f.read()
        result = np.fromstring(bstring, dtype=float, sep='\n')
        wave, flux = result[0::3], result[1::3]

    return wave, flux

def construct_model_filename(teffi, loggi, fehi, alphai, geometry='sph',
model_mass=1.0, micro_turb_vel=2., path='.', grid_type='default'):

        #The user is allowed to define the geometry to some extent,
        #but there are limits as described above
        if loggi < 3.0: geometry = 'sph' #override user input if necessary
        if loggi > 3.5: geometry = 'pp' #only plane parallel models

        if geometry == 'sph':
            geostr = 's'
        if geometry == 'pp':
            geostr = 'p'
            model_mass = 0.0 #overwrite default parameters

        #if its the default MARCS grid, then check the zclass
        if grid_type == 'default':
            try:
                zclass = check_zclass(fehi, alphai)
            except:
                if alphai >= 0.4: zclass = 'ae'
                elif alphai <= -0.4: zclass = 'an'
                else: zclass = 'ap'

        #if using the MARCS-APOGEE grid, then just use 'x3'
        if grid_type == 'apogee':
            zclass = 'x3'

        if np.sign(loggi) == 1 or np.sign(loggi) == 0: gsign = '+'
        else: gsign = '-'

        if np.sign(fehi) == 1 or np.sign(fehi) == 0: fsign = '+'
        else: fsign = '-'

        if np.sign(alphai) == 1 or np.sign(alphai) == 0: asign = '+'
        else: asign = '-'

        feh_str = str(np.abs(fehi))
        if len(feh_str) != 4: feh_str += '0'

        alpha_str = str(np.abs(alphai))
        if len(alpha_str) != 4: alpha_str += '0'

        vt_str = f'0{int(micro_turb_vel)}'

        gstr = str(int(loggi*10.))
        if len(gstr) != 2: gstr = '0'+gstr

        modelfn = (f"{path}/t{int(teffi)}/g_{gstr}/{geostr}{int(teffi)}_g{gsign}{np.abs(loggi)}"
                   f"_m{model_mass}_t{vt_str}_{zclass}_z{fsign}{feh_str}"
                   f"_a{asign}{alpha_str}_c+0.00_n+0.00_o{asign}{alpha_str}"
                   f"_r+0.00_s+0.00.bin.gz")

        return modelfn

#Function for loading a specified synthetic spectrum
def load_synth(p, hash=None, data_path=None):

    teffi, loggi, fehi, alphai = p

    if loggi <= 3.0:
        model_vt = 2.
        geo = 'sph'
    else:
        model_vt = 1.
        geo = 'pp'

    if hash is not None:

        #First construct the filename corresponding to the parameters, to use for
        #testing whether we should read in the specified synthetic spectrum

        modelfn = construct_model_filename(teffi, loggi, fehi, alphai, path=data_path,
        micro_turb_vel=model_vt, grid_type='apogee', geometry=geo)

        key = modelfn.split('/')[-1][:-7]

        if key not in hash.keys():

            _,synthi = read_synth(Teff=teffi, Logg=loggi,
            Feh=fehi, Alphafe=alphai, data_path=data_path)

            hash[key] = synthi

        #If the key is already present in the hash table, then find it and load the data
        else: synthi = hash[key]

    else:
        _,synthi = read_synth(Teff=teffi, Logg=loggi,
        Feh=fehi, Alphafe=alphai, data_path=data_path)

    return synthi


def getee(ndim=4):

    """
    Based on getee.f90 from FERRE

    generates a matrix with ndim columns and base**ndim rows
    which can be used to transform ndim nested loops (all indices
    running 0 to 1) into a single loop

    Case inter = 3 is cubic bezier interpolation (=> base 4)
    """

    base = 4
    #NOTE THAT FORTRAN SWAPS ROWS AND COLUMNS RELATIVE TO PYTHON
    ee = np.zeros((base**ndim, ndim)).astype('int')

    for i in range(ndim): #note that Fortran indexing starts at 1, Python at 0
        nr = base**i
        c1 = 0
        for j in range(nr):
            c2 = 0
            ratio = int(base**ndim/nr)
            for k in range(ratio):
                ee[c1, i] = c2/base**(ndim-(i+1))
                c1 += 1
                c2 += 1

    powers = np.zeros(ndim).astype('int')
    for i in range(ndim):
        powers[i] = base**(ndim-(i+1))

    #from load_control.f90 in FERRE
    #initialize indi with default values when not set in the control file
    #indi=[ndim,ndim-1,ndim-2 ...,1,...]
    indi = np.zeros(ndim).astype('int')
    for i in range(ndim):
        indi[i] = ndim-i

    #copy ee to eeindi and swap columns to match interpolation order
    eeindi = np.zeros((base**ndim, ndim)).astype('int')
    for i in range(ndim):
        eeindi[:,i] = ee[:, ndim-indi[i]]

    imap = np.zeros(base**ndim).astype('int')
    for i in range(base**ndim):
        imap[i] = np.dot(eeindi[i,:], powers)

    return ee, imap


def cub(p, params_all, ndim=4, n_p=[21, 11, 15, 9], hash=None, npix=35715,
f_access=1, synth_path='/central/groups/carnegie_poc/iescala/tsgrid_ch/'):

    """
    Based on cub.f90 from FERRE

    Multidimensional cubic interpolation.

    p(ndim) is ndim vector of wanted pars [0-1], where ndim is dimension of grid (ndim = 4)
    flux(npix) is n vector of output flux, npix is initialized at 0 in share.f90
               but should be size of wavelength array (int)
    n_p(maxndim) is the grid's size, where maxndim (int) is the limit to the number of dimensions
                 maxndim=20 in FERRE, but maxndim = 4 for our purposes

    integer,allocatable     :: ntimes(:) !ntimes(i)=Prod_(i+1)^ndim(n_p(i)) from share.f90
    integer,allocatable	:: ee(:,:) see getee from share.f90

    offset(ndim) !offset=0  [-1,0,1,2] (cubic), integer
    findex(4**ndim) !indices of rows in f needed for the interpolation, integer

    params to share between read_f and lin/qua/cub
    real(dp), allocatable :: f(:,:) synth grid
    """

    #Call this function to allocate these arrays
    ee, imap = getee(ndim=ndim)

    n_p = np.array(n_p)

    #get t's from p's. The p's run between 0-1, while t's run
    #from the min to the max value of the indices of f (1<ti<n_pi) in Fortran
    #0<ti<n_pi-1 in Python
    t = np.zeros(ndim)
    t[:ndim] = p[:ndim]*(n_p[:ndim] - 1)

    #This is to deal with interpolation at the grid edges
    #offset = 0 means use cubic, use other methods otherwise
    offset = np.zeros(ndim).astype('int')
    for i in range(ndim):
        if (t[i] < 1): offset[i] = 1
        if (t[i] >= ((n_p[i]-1) - 1 )): offset[i] = -1

    #Allocate ntimes, based on read_f.f90 in FERRE
    #ntot is the total number of models in the grid
    #ntimes is the number of models fixing a given element of an array
    #for example, for a given teff value, there are nlogg*nfeh*nalpha options
    ntimes = np.ones(ndim).astype('int')
    ntot = n_p[ndim-1]
    for j in range(2, ndim+1):
        ntimes[ndim-j] = ntot
        ntot *= n_p[ndim-j]

    #Find the indices of the models in the binned ASCII file
    findex = np.zeros(4**ndim).astype('int')
    for i in range(4**ndim):
        findex[i] = np.dot( ntimes, (t[:ndim].astype('int') + ee[i, :ndim] - 1 + offset[:ndim]) )

    #Load the working matrix
    #NOTE: Fortran is (columns, rows)
    #Python is (rows, columns)

    wrk = np.zeros((4**ndim, npix))
    if f_access == 0:
        for i in range(4**ndim):
            #wrk[i, :npix] = f[findex[i], :npix]
            wrk[i,:npix] = f[findex[imap[i]],:npix]
    else:
        for i in range(4**ndim):
            wrk[i,:npix] = load_synth(params_all[findex[imap[i]]], data_path=synth_path,
                                   hash=hash)

    #Actually interpolate
    @jit(nopython=True)
    def cbezier_loop(t,wrk,offset,npix,ndim=4):

        delta = t - t.astype('int')
        delta = delta[::-1]
        omdelta = 1. - delta

        for i in range(ndim):

            #delta = t[indi[i]-1] - int(t[indi[i]-1])
            #omdelta = 1. - delta

            delta4 = np.zeros(4)
            delta3 = np.zeros(3)

            if offset[ndim-(i+1)] == 0:
                delta4[0] = omdelta[i]**3
                delta4[1] = delta[i]**3
                delta4[2] = 3.*delta[i]*omdelta[i]**2
                delta4[3] = 3.*delta[i]**2*omdelta[i]

                for j in range(4**(ndim-(i+1))):
                    for l in range(npix):

                        #yp = cbezier(wrk[4*(j+1)-4,l], wrk[4*(j+1)-3,l],
                        #             wrk[4*(j+1)-2,l], wrk[4*(j+1)-1,l], delta4)

                        y1 = wrk[4*(j+1)-4,l]
                        y2 = wrk[4*(j+1)-3,l]
                        y3 = wrk[4*(j+1)-2,l]
                        y4 = wrk[4*(j+1)-1,l]

                        yprime0 = 0.5*(y3-y1)
                        c0 = y2 + yprime0/3.
                        yprime1 = 0.5*(y4-y2)
                        c1 = y3 - yprime1/3.

                        yp = y2*delta4[0] + y3*delta4[1] + c0*delta4[2] + c1*delta4[3]

                        wrk[j,l] = yp

            else:

                if offset[ndim-(i+1)] == 1:
                    qoffset=1
                else:
                    qoffset=0

                delta3[0]=omdelta[i]**2
                delta3[1]=delta[i]**2
                delta3[2]=2.*delta[i]*omdelta[i]

                for j in range(4**(ndim-(i+1))):
                    for l in range(npix):

                        y1 = wrk[4*(j+1)-3-qoffset,l]
                        y2 = wrk[4*(j+1)-2-qoffset,l]
                        y3 = wrk[4*(j+1)-1-qoffset,l]

                        if qoffset == 0:
                            yprime = 0.5*(y3-y1)
                            c0 = y2+0.5*yprime
                            yp = y2*delta3[0] + y3*delta3[1] + c0*delta3[2]

                        else:
                            yprime=0.5*(y3-y1)
                            c0=y2-0.5*yprime
                            yp=y1*delta3[0] + y2*delta3[1] + c0*delta3[2]

                        wrk[j,l] = yp

        return yp

    yp = cbezier_loop(t,wrk,offset,npix)

    flux=wrk[0,:]

    return flux

def convert_params(praw):

    """
    Convert the input raw model grid parameters to
    the parameter range [0-1] for use with the interpolation
    function based on the original Fortran code from FERRE
    """

    (teffi, loggi, fehi, alphafei) = praw

    teff_arr = np.concatenate((np.arange(3000, 4100, 100), np.arange(4250, 6750, 250)))
    logg_arr = np.arange(0., 5.5, 0.5)

    #no interp grid
    #feh_arr = np.arange(-2.5, 1.25, 0.25)
    #alphafe_arr = np.arange(-1., 1.25, 0.25)

    #interp grid
    feh_arr = np.round(np.arange(-2.5, 1.1, 0.1), decimals=2)
    alphafe_arr = np.round(np.arange(-1.0, 1.1, 0.1), decimals=2)

    params_grid = np.array([teff_arr, logg_arr, feh_arr, alphafe_arr], dtype='object')
    params = np.array([teffi, loggi, fehi, alphafei])

    params_all = np.asarray(list(itertools.product(teff_arr, logg_arr, feh_arr, alphafe_arr)))

    p = np.zeros(params.size)
    n_p = np.zeros(params.size)
    for i in range(params.size):
        scaled_arr = np.linspace(0, 1, params_grid[i].size)
        pi = np.interp(params[i], params_grid[i], scaled_arr)
        p[i] = pi
        n_p[i] = params_grid[i].size

    return p, n_p, params_all

def read_interp_synth(teff=np.nan, logg=np.nan, feh=np.nan, alphafe=np.nan, hash=None,
                      start=4100, sstop=9100, step=0.14,
                      data_path='/central/groups/carnegie_poc/iescala/tsgrid_ch'):

    p, n_p, params_all = convert_params((teff, logg, feh, alphafe))

    wave = np.arange(start, sstop, step)
    npix = wave.size

    flux = cub(p, params_all, n_p=n_p, hash=hash, npix=npix, synth_path=data_path,
               f_access=1)

    return wave, flux


#######################################################################################


"""
def read_grid(indices, data_path='/central/groups/carnegie_poc/iescala/tsgrid_ch',
npix=35715, hash=None):

    gridfn = 'tsgrid_ch_nointerp.dat.gz'
    full_filename = f'{data_path}/{gridfn}'

    def read_f(indices_subset):

        grid_subset = np.zeros((len(indices_subset), npix))

        nn = 0
        with gzip.open(full_filename, 'r') as f:
            for i,line in enumerate(f):
                if i in indices_subset:
                    line_split = line.decode().split()
                    line_float = np.array([float(l) for l in line_split])
                    grid_subset[nn,:] = line_float
                    nn += 1

        return grid_subset

    if hash is not None:

        grid = np.zeros((len(indices), npix))

        count = np.arange(0, len(indices), 1).astype('int')
        w_in_hash = np.asarray([ii in hash.keys() for ii in indices])

        if len(indices[w_in_hash]) != 0:
            for ii in range(indices[w_in_hash].size):
                grid[count[w_in_hash][ii],:] = hash[indices[w_in_hash][ii]]

        if len(indices[~w_in_hash]) != 0:
            grid_subset = read_f(indices[~w_in_hash])
            for ii in range(indices[~w_in_hash].size):
                hash[indices[~w_in_hash][ii]] = grid_subset[ii,:]
            #for ii,n in enumerate(count[~w_in_hash]):
            #    grid[n,:] = grid_subset[ii,:]
            grid[count[~w_in_hash],:] = grid_subset

    else:

        grid = read_f(indices)

    return grid
"""
