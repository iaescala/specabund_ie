#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:22:55 2019

@author: kevinm
Modifications by I. Escala 2019-2020
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib

plt.ioff()

from tqdm import tqdm
import numpy as np
from scipy import interpolate
import emcee
import scipy.optimize as op
import time
import corner
import os
import spectrum_class
import fnmatch
import sys

font = {'family' : 'serif',
        'size'   : 16,}
matplotlib.rc('font', **font)


light = 2.99792458e5 #speed of light in km/s
#classic wavelengths
Halpha = 6562.81
Hbeta = 4861.34
Magnesium = [5167.33, 5172.70, 5183.62]
Calcium = [8498.03, 8542.09, 8662.14]
lines = {'H_Alpha':Halpha,'H_Beta':Hbeta,'Ca_Triplet':Calcium,'Mg_Triplet':Magnesium}


def find(pattern, path):
    '''works like command line's find function'''
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def collapseArray(wavelength,arcflux,slitflux,numRows=5,starRow=True):
    '''Uses specified 2d wavelength and fluxes to cut out a 1d spectrum.
    Finds starRow from slitflux, then takes the numRows around starRow
    and interpolates onto a final wavelength to make 1d arcflux spectrum.'''
    startWave,endWave = np.max(wavelength[:,0]),np.min(wavelength[:,-1])
    firstWave = wavelength[0][(wavelength[0] >= startWave) & (wavelength[0] <= endWave)]

    interp = np.zeros((numRows,len(firstWave)))

    if starRow: #find the peak of the star's flux in row values
        ignore = 5
        centerRow = (np.mean(arcflux,axis=1)[ignore:-ignore]).argmax()+ignore
    else: #use the center Row
        centerRow = wavelength.shape[0]//2

    for i in range(centerRow-numRows//2,centerRow+numRows//2+1):
        f = interpolate.interp1d(wavelength[i],arcflux[i])
        interp[i-centerRow-numRows//2] += f(firstWave)

    results = np.average(interp,axis=0)
    errors = np.std(interp,axis=0)
    return firstWave,results,errors


def gaussian(x, mu, sigma, height):
    '''unnormalized, with a vertical shift/ offset.
    used to fit peaks in spectra'''
    if type(mu) == type(np.ndarray(0)):
        x = np.array([x])
        return (height * np.exp(-0.5*np.power((x.T-mu)/sigma,2)) + 1).T
    else:
        return height * np.exp(-0.5*(x-mu)**2/float(sigma)**2) + 1


#def fitGaussians(num,field,date,mask,submask,plotter=False,path='.',out='.',fittype='emcee'):
#def fitGaussians(num,field,mask,subfolder,plotter=False,path='.',out='.',fittype='emcee'):
def fitGaussians(num=None,field=None,obsdate=None,submask=None, subfolder=None,
plotter=False,path='.',out='.',fittype='emcee', stacked=False):
    '''Takes calibfile and finds peaks in arclamp spectrum. Fits gaussians to arclamp spectrum
    and saves the fit gaussian parameters. Takes a slit number with field, date, mask, and submask'''

    if not stacked: savePath = f'{out}{field}/{submask}/'
    else: savePath = f'{out}{field}/{submask}/{obsdate}/{subfolder}/'

    if not(os.path.isdir(savePath)):
        os.makedirs(savePath)

    filepath = path

    starNum = ''
    print(filepath)
    for filename in os.listdir(filepath):

        if not stacked: spec1dfile = f'spec1d.%s.%03d.'%(field,num)
        else: spec1dfile = f'spec1d.%s.%03d.'%(submask,num)

        if spec1dfile in filename and '.fits.gz' in filename and 'zspec1d' not in filename:
            try:
                starNum = filename.split('.')[3]
            except:
                continue
            break

    if starNum == '':
        if not stacked: raise ValueError(f'ERROR: Cannot locate slit {num} along {filepath}')
        else:
            sys.stderr.write(f'ERROR: Cannot locate slit {num} along {filepath}')
            return

    #first locate the slit and calib files for a star
    calibLocation =  f'{path}'

    if not stacked: shortCalibName = "calibSlit.%s.%03d"%(field,num)
    else: shortCalibName = "calibSlit.%s.%03d"%(submask,num)
    calibFile = ''.join([calibLocation,shortCalibName])

    if not stacked: shortSlitName = "slit.%s.%03d"%(field,num)
    else: shortSlitName = "slit.%s.%03d"%(submask,num)
    slitFile = ''.join([calibLocation,shortSlitName])

    blueCalib = ''.join([calibFile,'B.fits.gz'])
    redCalib = ''.join([calibFile,'R.fits.gz'])
    blueSlit = ''.join([slitFile,'B.fits.gz'])
    redSlit = ''.join([slitFile,'R.fits.gz'])

    try: #if the calib file is local
        calibB = fits.open(blueCalib)
    except:
        if not stacked: raise ValueError(f"ERROR: Can't find {blueCalib}")
        else:
            sys.stderr.write(f"ERROR: Can't find {blueCalib}")
            return
    try: #if the calib file is local
        calibR = fits.open(redCalib)
    except:
        raise ValueError(f"ERROR: Can't find {redCalib}")
    try: #if the calib file is local
        slitB = fits.open(blueSlit)
    except:
        raise ValueError(f"ERROR: Can't find {blueSlit}")
    try: #if the calib file is local
        slitR = fits.open(redSlit)
    except:
        raise ValueError(f"ERROR: Can't find {redSlit}")

    try:
        rawarcB = calibB[1].data['RAWARC'][0]
    except:
        calibB[1].header['X0SYNTH'] = 0
        calibB[1].header['X1SYNTH'] = 0
        rawarcB = calibB[1].data['RAWARC'][0]
    try:
        rawarcR = calibR[1].data['RAWARC'][0]
    except:
        calibR[1].header['X0SYNTH'] = 0
        calibR[1].header['X1SYNTH'] = 0
        rawarcR = calibR[1].data['RAWARC'][0]
    #if there are two arclamp exposures, combine them
    if len(rawarcB.shape) == 3:
        rawarcB = np.average(rawarcB,axis=0)
    if len(rawarcR.shape) == 3:
        rawarcR = np.average(rawarcR,axis=0)
    slitfluxB = slitB[1].data['FLUX'][0]
    slitfluxR = slitR[1].data['FLUX'][0]

    #changes the x=[0,1,2,3,...] indices to lambda=[4333,...] \AA values
    waveB = slitB[1].data['DLAMBDA'][0]+slitB[1].data['LAMBDA0'][0]
    waveR = slitR[1].data['DLAMBDA'][0]+slitR[1].data['LAMBDA0'][0]

    #returns the interp wavelengths, arcfluxes, and standard errors
    collapsedB = collapseArray(waveB,rawarcB,slitfluxB,numRows=5,starRow=True)
    collapsedR = collapseArray(waveR,rawarcR,slitfluxR,numRows=5,starRow=True)

    xsubset,ysubset,ivarsubset = np.copy(collapsedB)
    sigmaThreshold = (10,20) #number of sigma allowed below, above data
    splineB = 0
    splineChanging = True
    i,minTimes = 0,3
    window = 25
#    Fit the background continuum
    while (i < minTimes) or splineChanging: #keep looping until spline stops changing, min 3 times
        if i <= 1:
            s = len(xsubset)+np.sqrt(2*len(xsubset)) #smoothing value
        else: #smoothing gets smaller over time
            s = len(xsubset)-np.sqrt(2*len(xsubset)) #smoothing value
    #        t = np.arange(xsubset[1],xsubset[-1],window) #knots locations
        deltaWave = xsubset[1:]-xsubset[:-1]
        t = np.array([])
        knotInds = np.insert(np.where(deltaWave >= window),0,0)
        knotInds = np.append(knotInds,len(xsubset)-1)
        for j in range(1,len(knotInds)):
            t = np.append(t,np.arange(xsubset[knotInds[j-1]+1],xsubset[knotInds[j]],window))
        try: t, c, k = interpolate.splrep(xsubset, ysubset, k=3, s=s, w=np.power(ysubset,-2), task=-1, t=t)
        except: return
        newSpline = interpolate.BSpline(t, c, k, extrapolate=True)

        if i >= minTimes: #then start spline comparisons
            testSplineChange = np.abs((splineB(xsubset)-newSpline(xsubset))/splineB(xsubset))
            if np.all(testSplineChange < 0.001): #if no points in spline are changing by more than 0.1%, stop
                splineChanging = False
        splineB = newSpline
        #make a list of the points that are within sigmaThreshold of the spline
        distance = (ysubset-splineB(xsubset))*np.sqrt(ivarsubset)
        notTooFar = np.logical_or((-sigmaThreshold[0] < distance) & (distance <= 0),(sigmaThreshold[1] > distance) & (distance >= 0))
        notTooFar[0] = True
        notTooFar[-1] = True
        #remove points too far away
        xsubset,ysubset,ivarsubset = xsubset[notTooFar],ysubset[notTooFar],ivarsubset[notTooFar]
        i += 1

    xsubset,ysubset,ivarsubset = np.copy(collapsedR)

    sigmaThreshold = (10,20) #number of sigma allowed below, above data
    splineR = 0
    splineChanging = True
    i,minTimes = 0,3
    window = 25
    while (i < minTimes) or splineChanging: #keep looping until spline stops changing, min 3 times
        if i <= 1:
            s = len(xsubset)+np.sqrt(2*len(xsubset)) #smoothing value
        else: #smoothing gets smaller over time
            s = len(xsubset)-np.sqrt(2*len(xsubset)) #smoothing value
        deltaWave = xsubset[1:]-xsubset[:-1]
        t = np.array([])
        knotInds = np.insert(np.where(deltaWave >= window),0,0)
        knotInds = np.append(knotInds,len(xsubset)-1)
        for j in range(1,len(knotInds)):
            t = np.append(t,np.arange(xsubset[knotInds[j-1]+1],xsubset[knotInds[j]],window))
        t, c, k = interpolate.splrep(xsubset, ysubset, k=3, s=s, w=np.power(ysubset,-2), task=-1, t=t)
        newSpline = interpolate.BSpline(t, c, k, extrapolate=True)

        if i >= minTimes: #then start spline comparisons
            testSplineChange = np.abs((splineR(xsubset)-newSpline(xsubset))/splineR(xsubset))
            if np.all(testSplineChange < 0.001): #if no points in spline are changing by more than 0.1%, stop
                splineChanging = False
        splineR = newSpline
        #make a list of the points that are within sigmaThreshold of the spline
        distance = (ysubset-splineR(xsubset))*np.sqrt(ivarsubset)
        notTooFar = np.logical_or((-sigmaThreshold[0] < distance) & (distance <= 0),(sigmaThreshold[1] > distance) & (distance >= 0))
        notTooFar[0] = True
        notTooFar[-1] = True
        #remove points too far away
        xsubset,ysubset,ivarsubset = xsubset[notTooFar],ysubset[notTooFar],ivarsubset[notTooFar]
        i += 1

    normalizedB = np.array((collapsedB[0],collapsedB[1]/splineB(collapsedB[0]),collapsedB[-1]*np.power(splineB(collapsedB[0]),2)))
    normalizedR = np.array((collapsedR[0],collapsedR[1]/splineR(collapsedR[0]),collapsedR[-1]*np.power(splineR(collapsedR[0]),2)))

    #start finding peaks and then fitting them with emcee
    #change emcee fitting to likelihood one to speed this up
    normalizedAll = [normalizedB,normalizedR]
    parameters = []
    for ind,normalized in enumerate(normalizedAll):
        deltaX = normalized[0][1:]-normalized[0][:-1]
        deltaY = normalized[1][1:]-normalized[1][:-1]
        derivs = np.zeros(len(normalized[0]))
        derivs[-1] = deltaY[-1]/deltaX[-1]
        derivs[:-1] = (deltaY/deltaX)
        derivs[1:] = (derivs[1:]+derivs[:-1])*0.5

        peaks = np.where(deltaY > 0)
        peaks = peaks[0][np.where(peaks[0][1:] - peaks[0][:-1] > 1)]+1

        peakInds = []

        row = normalized[1]
        medianVal = np.median(normalized[1])

        for peak in peaks:
            results = [peak]
            height = row[peak]
            if height < medianVal or height - medianVal < 0.5:
                continue
            #go forward first
            i = peak+1
            nearPeak = True #test whether near peak still
            while (i < len(row)-1) and (row[i] > max(1.05,0.9*medianVal)):
                #while above the median, keep adding
                if abs((height-row[i])/(height)) < 0.01 and nearPeak:
                    results.append(i)
                elif abs((height-row[i])/(height)) >= 0.01 and nearPeak:
                    nearPeak = False
                    results.append(i)
                elif (row[i] > row[i+1]):
                    #if we are decreasing as marching forward, add values
                    results.append(i)
                    slope = (row[i]-row[i+1])/(normalized[0][i]-normalized[0][i+1])
                    if slope > -0.1:
                        break
                else:
                    break
                i += 1
            #go backward
            i = peak-1
            nearPeak = True #test whether near peak still
            while (i > 0) and (row[i] > max(1.05,0.9*medianVal)):
                if abs((height-row[i])/(height)) < 0.01 and nearPeak:
                    results.insert(0,i)
                elif abs((height-row[i])/(height)) >= 0.01 and nearPeak:
                    nearPeak = False
                    results.insert(0,i)
                elif (row[i] > row[i-1]):
                    #if we are decreasing as marching backward, add values
                    results.insert(0,i)
                    slope = (row[i]-row[i-1])/(normalized[0][i]-normalized[0][i-1])
                    if slope < 0.1:
                        break
                else:
                    break
                i -= 1
            if len(results) <= 3:
                continue
            results = np.array(results)
            peakInds.append(results)
        peakInds = np.array(peakInds)

        if plotter:
            plt.plot(collapsedB[0],collapsedB[1])
            plt.plot(collapsedB[0],splineB(collapsedB[0]))
            plt.gca().set_yscale('log')
#            plt.show()
            plt.plot(collapsedR[0],collapsedR[1])
            plt.plot(collapsedR[0],splineR(collapsedR[0]))
            plt.gca().set_yscale('log')
#            plt.show()
            plt.plot(normalized[0],normalized[1],c='b')
            [plt.plot(normalized[0][peakInd],normalized[1][peakInd]) for peakInd in peakInds]
            plt.gca().set_yscale('log')
#            plt.show()
            plt.close('all')

        def lnprior(theta,bounds):
            for i in range(len(theta)):
                if not (bounds[i][0] <= theta[i] <= bounds[i][1]):
                    return -np.inf
            return 0.0

        def lnlike(theta, x, y, yerr):
            mu, sigma, height, lnf = theta
            model = gaussian(x, mu, sigma, height)
            inv_sigma2 = 1.0/(np.power(yerr,2) + np.power(model,2)*np.exp(2*lnf))
            return -0.5*(np.sum(np.power(y-model,2)*inv_sigma2 - np.log(inv_sigma2)))
#            return -0.5*(np.sum((y-model)**2))

        def lnprob(theta, x, y, yerr, bounds):
            lp = lnprior(theta,bounds)
            if not np.isfinite(lp):
                return -np.inf
            ll = lnlike(theta, x, y, yerr)
            return lp + ll

        paramResult = []
        ndim, nwalkers, nsteps = 4, 200, 500
        burnin = 300

        for i,peakInd in enumerate(tqdm(peakInds,total=len(peakInds))):
            x = normalized[0][peakInd]
            y = normalized[1][peakInd]
            yerr = 1/np.sqrt(normalized[2][peakInd])

            peakVal = np.max(y)
            mu = x[np.where(y == peakVal)[0][0]]
            height = peakVal-1
            sigma = np.std(x)

            guesses = np.array([mu,sigma,height]) #first guesses of params
            bounds = np.array(((x.min(),0,0),(x.max(),3,peakVal*2)))
            if guesses[1] > bounds[1][1]:
                guesses[1] = bounds[1][1]
            def nll(theta):
                return -lnlike(theta,x,y,yerr)

            popt,pcov = op.curve_fit(gaussian,x,y,sigma=yerr,p0=guesses,bounds=bounds,absolute_sigma=True)

            mu_ml, sigma_ml, height_ml = popt
            newGuess = np.array(popt)
            errors = np.sqrt(np.diag(pcov))

            newGuess = np.append(newGuess,np.log(0.05))
            errors = np.append(errors,0)
            if fittype != 'emcee':
                paramResult.append([(newGuess[i],errors[i]) for i in range(len(newGuess))])
                continue #SKIP THE EMCEE FITTING, JUST MAXIMUM LIKELIHOOD

            bounds = np.array([(x[0],x[-1]),(0,4),(0,2*peakVal),(-10,1)])

            pos = np.array([newGuess + 1e-2*np.random.randn(ndim)*np.array([100,2,2,2]) for j in range(nwalkers)])

            badpos = np.logical_not(np.all((pos-bounds[:,0] > 0) & (bounds[:,1]-pos > 0 ),axis=1))
            while np.sum(badpos) > 0:
                pos[badpos] = np.array([newGuess + 1e-2*np.random.randn(ndim)*np.array([100,2,2,2]) for j in range(np.sum(badpos))])
                badpos = np.logical_not(np.all((pos-bounds[:,0] > 0) & (bounds[:,1]-pos > 0 ),axis=1))

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, bounds), threads = 1)

            sampler.run_mcmc(pos, nsteps)

            dimLabels = ['$\mu$','$\sigma$','A','$\ln$f']
            samplerChain = sampler.chain
            if plotter:
                plt.figure(figsize=[9,7])
                for dim in range(ndim):
                    plt.subplot(ndim,1,dim+1)
                    plt.plot(samplerChain[:,:,dim].T,alpha=0.25)
                    if dim != ndim-1:
                        plt.xticks([])
                    else:
                        plt.xticks(np.arange(0, samplerChain.shape[1]+1, samplerChain.shape[1]/10))
                    plt.ylabel(dimLabels[dim])
                plt.xlabel('Step Number')
                plt.tight_layout()
                plt.close('all')

            samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

            if plotter:
                plt.figure(figsize=[9,7])
                corner.corner(samples, labels=dimLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                 title_kwargs={"fontsize": 12})
                plt.close('all')
            emceeResult = map(lambda v: (v[1], v[0], v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
            emceeResult = np.array(list(emceeResult))

            newGuess = 0.5*(emceeResult[:,1]+emceeResult[:,2])
            errors = 0.5*(emceeResult[:,1]+emceeResult[:,2])-emceeResult[:,1]

            paramResult.append([(newGuess[i],errors[i]) for i in range(len(newGuess))])

            if plotter:
                deltaX = x[-1]-x[0]
                fitx = np.linspace(x[0]-0.25*deltaX,x[-1]+0.25*deltaX,num=50)

                predictedVals = gaussian(fitx,samples.T[0],samples.T[1],samples.T[2])
                predictedBounds = np.array(list(map(lambda v: [v[0], v[1], v[2], v[3]],zip(*np.percentile(predictedVals, [2.5,16,84,97.5],axis=0)))))

                plt.figure()
                plt.errorbar(x,y,yerr=yerr,label='Data')
                plt.plot(fitx,gaussian(fitx,guesses[0],guesses[1],guesses[2]),label='Initial Guess')
                plt.plot(fitx,gaussian(fitx,mu_ml,sigma_ml,height_ml),label='Min Chi-Sq')
                plt.plot(fitx,gaussian(fitx,emceeResult[0][0],emceeResult[1][0],emceeResult[2][0]),label='MCMC')
                plt.fill_between(fitx,predictedBounds[:,0],predictedBounds[:,1],interpolate=True,alpha=0.25,color='grey')
                plt.fill_between(fitx,predictedBounds[:,1],predictedBounds[:,2],interpolate=True,alpha=0.25,color='red')
                plt.fill_between(fitx,predictedBounds[:,2],predictedBounds[:,3],interpolate=True,alpha=0.25,color='grey')
                plt.legend(loc='best')
                plt.close('all')
        paramResult = np.array(paramResult)
        parameters.extend(paramResult)

        if plotter:
            [plt.plot(normalized[0][peakInd],normalized[1][peakInd]) for peakInd in peakInds]
            plt.gca().set_yscale('log')
            xlim = plt.xlim()
            plt.show()
            plt.errorbar(paramResult[:,0,0],paramResult[:,1,0],yerr=paramResult[:,1,1],xerr=paramResult[:,0,1],fmt='o',ms=1)
            plt.errorbar(paramResult[:,0,0],paramResult[:,1,0],yerr=paramResult[:,1,1],xerr=paramResult[:,0,1],fmt='o',ms=1)
            plt.axhline(np.median(paramResult[:,1,0]),label='Median',ls='--',c='C1')
            plt.xlim(xlim)
            plt.show()

    parameters = np.array(parameters)
    #save the fit parameters
    print(savePath)
    if not stacked: np.save(savePath+'%s.%03d.gaussian_params.npy'%(field,num),parameters)
    else: np.save(savePath+'%s.%03d.gaussian_params.npy'%(submask,num),parameters)

    return


def skylineFitter(num=None,field=None,submask=None,obsdate=None,subfolder=None,
plotter=False, path='',out='',fittype='emcee', stacked=False, site='keck'):
    '''Takes spec1d and finds peaks in night skyline spectrum. Fits gaussians to skyline spectrum
    and saves the fit gaussian parameters. Takes a slit number with field, date, mask, and submask'''

    #savepath = f'{out}{field}/{mask}/{date}/{submask}/'

    if not stacked: savepath = f'{out}{field}/{submask}/'
    else: savepath = f'{out}{field}/{submask}/{obsdate}/{subfolder}/'

    if not(os.path.isdir(savepath)):
        os.makedirs(savepath)

    filepath =  path
    #temp = path.split('/')
    #subfolder = temp[-2]

    starNum = ''
    for filename in os.listdir(filepath):

        if not stacked:
            if site == 'keck':
                spec1dfile = f'spec1d.%s.%03d.'%(field,num)
            if site == 'magellan':
                spec1dfile = f'spec1d.%s.%s.'%(field,str(num))
        else: spec1dfile = f'spec1d.%s.%03d.'%(submask,num)

        if spec1dfile in filename and '.fits.gz' in filename and 'zspec1d' not in filename:

            try:
                starNum = filename.split('.')[3]
            except:
                continue
            break

    if starNum == '':
        print(f'ERROR: Cannot locate slit {num} along {filepath}')
        return

    #reads in 1d spectrum
    spec = spectrum_class.Spectrum(f'{filepath}{filename}', site=site)

    wave,flux,ivar = spec.lam,spec.sky,spec.ivar
    xsubset,ysubset,ivarsubset = np.copy(wave),np.copy(flux),np.copy(ivar)
    keep = (ivarsubset > 0) & (ysubset > 0)
    xsubset,ysubset,ivarsubset = xsubset[keep],ysubset[keep],ivarsubset[keep]

    sigmaThreshold = (3,3) #number of sigma allowed below, above data
    spline = 0
    splineChanging = True
    i,minTimes = 0,3
    window = 20

    #fit the background continuum
    while (i < minTimes) or splineChanging: #keep looping until spline stops changing, min 3 times

        if i <= 1:
            s = len(xsubset)+np.sqrt(2*len(xsubset)) #smoothing value
        else: #smoothing gets smaller over time
            s = len(xsubset)-np.sqrt(2*len(xsubset)) #smoothing value

        deltaWave = xsubset[1:]-xsubset[:-1]

        t = np.array([])
        knotInds = np.insert(np.where(deltaWave >= window),0,0)
        knotInds = np.append(knotInds,len(xsubset)-1)

        for j in range(1,len(knotInds)):
            t = np.append(t,np.arange(xsubset[knotInds[j-1]+1],xsubset[knotInds[j]],window))

        t, c, k = interpolate.splrep(xsubset, ysubset, k=3, s=s, w=np.power(ysubset,-2), task=-1, t=t)
        newSpline = interpolate.BSpline(t, c, k, extrapolate=True)

        if i >= minTimes: #then start spline comparisons
            testSplineChange = np.abs((spline(xsubset)-newSpline(xsubset))/spline(xsubset))
            if np.all(testSplineChange < 0.001): #if no points in spline are changing by more than 0.1%, stop
                splineChanging = False
        spline = newSpline
        #make a list of the points that are within sigmaThreshold of the spline
        distance = (ysubset-spline(xsubset))*np.sqrt(ivarsubset)
        notTooFar = np.logical_or((-sigmaThreshold[0] < distance) & (distance <= 0),(sigmaThreshold[1] > distance) & (distance >= 0))
        notTooFar[0] = True
        notTooFar[-1] = True
        #remove points too far away
        xsubset,ysubset,ivarsubset = xsubset[notTooFar],ysubset[notTooFar],ivarsubset[notTooFar]
        i += 1
    normalized = np.array((wave,flux/spline(wave),ivar*np.power(spline(wave),2)))

    deltaX = normalized[0][1:]-normalized[0][:-1]
    deltaY = normalized[1][1:]-normalized[1][:-1]
    derivs = np.zeros(len(normalized[0]))
    derivs[-1] = deltaY[-1]/deltaX[-1]
    derivs[:-1] = (deltaY/deltaX)
    derivs[1:] = (derivs[1:]+derivs[:-1])*0.5

    peaks = np.where(deltaY > 0)
    peaks = peaks[0][np.where(peaks[0][1:] - peaks[0][:-1] > 1)]+1

    peakInds = []

    #identify the peaks
    row = normalized[1]
    medianVal = np.median(normalized[1])
    for peak in peaks:
        height = row[peak]
        if height < medianVal or height - medianVal < 0.5:
            continue
        results = [peak]
        #go forward first
        i = peak+1
        nearPeak = True #test whether near peak still
        while (i < len(row)-1) and (row[i] > max(1.05,0.9*medianVal)):
            #while above the median, keep adding
            if abs((height-row[i])/(height)) < 0.05 and nearPeak:
                results.append(i)
            elif abs((height-row[i])/(height)) >= 0.05 and nearPeak:
                nearPeak = False
                results.append(i)
            elif (row[i] > row[i+1]):
                #if we are decreasing as marching forward, add values
                results.append(i)
                slope = (row[i]-row[i+1])/(normalized[0][i]-normalized[0][i+1])
                if slope > -0.1:
                    break
            else:
                break
            i += 1
        #go backward
        i = peak-1
        nearPeak = True #test whether near peak still
        while (i > 0) and (row[i] > max(1.05,0.9*medianVal)):
            if abs((height-row[i])/(height)) < 0.05 and nearPeak:
                results.insert(0,i)
            elif abs((height-row[i])/(height)) >= 0.05 and nearPeak:
                nearPeak = False
                results.insert(0,i)
            elif (row[i] > row[i-1]):
                #if we are decreasing as marching backward, add values
                results.insert(0,i)
                slope = (row[i]-row[i-1])/(normalized[0][i]-normalized[0][i-1])
                if slope < 0.1:
                    break
            else:
                break
            i -= 1
        if len(results) < 5:
            continue
        results = np.array(results)
        peakInds.append(results)
    peakInds = np.array(peakInds)


    if plotter:
        plt.plot(normalized[0],normalized[1],c='b')
        [plt.plot(normalized[0][peakInd],normalized[1][peakInd]) for peakInd in peakInds]
        plt.gca().set_yscale('log')
        plt.show()
        for peakInd in peakInds:
            if normalized[1][peakInd].max() > 15:
                plt.plot(normalized[1][peakInd])
        plt.show()

    def lnprior(theta,bounds):
        for i in range(len(theta)):
            if not (bounds[i][0] <= theta[i] <= bounds[i][1]):
                return -np.inf
        return 0.0

    def lnlike(theta, x, y, yerr):
        mu, sigma, height, lnf = theta
        model = gaussian(x, mu, sigma, height)
        inv_sigma2 = 1.0/(np.power(yerr,2) + np.power(model,2)*np.exp(2*lnf))
        return -0.5*(np.sum(np.power(y-model,2)*inv_sigma2 - np.log(inv_sigma2)))
#            return -0.5*(np.sum((y-model)**2))

    def lnprob(theta, x, y, yerr, bounds):
        lp = lnprior(theta,bounds)
        if not np.isfinite(lp):
            return -np.inf
        ll = lnlike(theta, x, y, yerr)
        return lp + ll

    paramResult = []
    ndim, nwalkers, nsteps = 4, 200, 500
    burnin = 300

    #start fitting peaks with emcee
    #change emcee fitting to likelihood one to speed up
    for i,peakInd in enumerate(tqdm(peakInds,total=len(peakInds))):
        x = normalized[0][peakInd]
        y = normalized[1][peakInd]
        yerr = np.ones_like(normalized[2][peakInd])*0.1

        peakVal = np.max(y)
        mu = x[np.where(y == peakVal)[0][0]]
        height = peakVal-1
        sigma = np.std(x)

        guesses = np.array([mu,sigma,height]) #first guesses of params
        bounds = np.array(((x.min(),0,0),(x.max(),3,peakVal*2)))
        if guesses[1] > bounds[1][1]:
            guesses[1] = bounds[1][1]
        def nll(theta):
            return -lnlike(theta,x,y,yerr)

        popt,pcov = op.curve_fit(gaussian,x,y,sigma=yerr,p0=guesses,bounds=bounds,absolute_sigma=True)

        mu_ml, sigma_ml, height_ml = popt
        newGuess = np.array(popt)
        errors = np.sqrt(np.diag(pcov))

        newGuess = np.append(newGuess,np.log(0.05))
        errors = np.append(errors,0)
        if fittype != 'emcee':
            paramResult.append([(newGuess[i],errors[i]) for i in range(len(newGuess))])
            continue #SKIP THE EMCEE FITTING, JUST MAXIMUM LIKELIHOOD

        bounds = np.array([(x[0],x[-1]),(0,4),(0,2*peakVal),(-10,1)])

        pos = np.array([newGuess + 1e-2*np.random.randn(ndim)*np.array([100,2,2,2]) for j in range(nwalkers)])

        badpos = np.logical_not(np.all((pos-bounds[:,0] > 0) & (bounds[:,1]-pos > 0 ),axis=1))
        while np.sum(badpos) > 0:
            pos[badpos] = np.array([newGuess + 1e-2*np.random.randn(ndim)*np.array([100,2,2,2]) for j in range(np.sum(badpos))])
            badpos = np.logical_not(np.all((pos-bounds[:,0] > 0) & (bounds[:,1]-pos > 0 ),axis=1))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, bounds), threads = 1)

        sampler.run_mcmc(pos, nsteps)

        dimLabels = ['$\mu$','$\sigma$','A','$\ln$f']
        samplerChain = sampler.chain
        if plotter:
            plt.figure(figsize=[9,7])
            for dim in range(ndim):
                plt.subplot(ndim,1,dim+1)
                plt.plot(samplerChain[:,:,dim].T,alpha=0.25)
                if dim != ndim-1:
                    plt.xticks([])
                else:
                    plt.xticks(np.arange(0, samplerChain.shape[1]+1, samplerChain.shape[1]/10))
                plt.ylabel(dimLabels[dim])
            plt.xlabel('Step Number')
            plt.tight_layout()
            plt.show()

        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

        if plotter:
            plt.figure(figsize=[9,7])
            corner.corner(samples, labels=dimLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                             title_kwargs={"fontsize": 12})
            plt.show()
        emceeResult = map(lambda v: (v[1], v[0], v[2]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
        emceeResult = np.array(list(emceeResult))


        newGuess = 0.5*(emceeResult[:,1]+emceeResult[:,2])
        errors = 0.5*(emceeResult[:,1]+emceeResult[:,2])-emceeResult[:,1]

        paramResult.append([(newGuess[i],errors[i]) for i in range(len(newGuess))])

        if plotter:
            deltaX = x[-1]-x[0]
            fitx = np.linspace(x[0]-0.25*deltaX,x[-1]+0.25*deltaX,num=50)

            predictedVals = gaussian(fitx,samples.T[0],samples.T[1],samples.T[2])
            predictedBounds = np.array(list(map(lambda v: [v[0], v[1], v[2], v[3]],zip(*np.percentile(predictedVals, [2.5,16,84,97.5],axis=0)))))

            plt.figure()
            plt.errorbar(x,y,yerr=yerr,label='Data')
            plt.plot(fitx,gaussian(fitx,guesses[0],guesses[1],guesses[2]),label='Initial Guess')
            plt.plot(fitx,gaussian(fitx,mu_ml,sigma_ml,height_ml),label='Min Chi-Sq')
            plt.plot(fitx,gaussian(fitx,emceeResult[0][0],emceeResult[1][0],emceeResult[2][0]),label='MCMC')
            plt.fill_between(fitx,predictedBounds[:,0],predictedBounds[:,1],interpolate=True,alpha=0.25,color='grey')
            plt.fill_between(fitx,predictedBounds[:,1],predictedBounds[:,2],interpolate=True,alpha=0.25,color='red')
            plt.fill_between(fitx,predictedBounds[:,2],predictedBounds[:,3],interpolate=True,alpha=0.25,color='grey')
            plt.legend(loc='best')
            plt.show()
    paramResult = np.array(paramResult)

    if plotter:
        [plt.plot(normalized[0][peakInd],normalized[1][peakInd]) for peakInd in peakInds]
        plt.gca().set_yscale('log')
        xlim = plt.xlim()
        plt.show()
        plt.errorbar(paramResult[:,0,0],paramResult[:,1,0],yerr=paramResult[:,1,1],xerr=paramResult[:,0,1],fmt='o',ms=1)
        plt.errorbar(paramResult[:,0,0],paramResult[:,1,0],yerr=paramResult[:,1,1],xerr=paramResult[:,0,1],fmt='o',ms=1)
        plt.axhline(np.median(paramResult[:,1,0]),label='Median',ls='--',c='C1')
        plt.xlim(xlim)
        plt.show()

    if not stacked:

        if site == 'keck':
            np.save(savepath+'%s.%03d.gaussian_params_sky.npy'%(field,num),paramResult)

        if site == 'magellan':
            np.save(savepath+'%s.%s.gaussian_params_sky.npy'%(field,num),paramResult)

    else:
        np.save(savepath+'%s.%03d.gaussian_params_sky.npy'%(submask,num),paramResult)

    return

def fitDeltaLambda(star,field=None,submask=None,obsdate=None,subfolder=None,plotter=True,
path='',out='',fittype='emcee',stacked=False):
    '''Takes a star's name and the field its in, then looks for all available gaussian parameters
    (from sky and arclamps) and then fits a quadratic polynomial to it.'''

    calibLocation = f'{path}'
    gaussLocation = f'{out}'

    parameters = []
    errors = []

    if not stacked: savePath = f'{out}{field}/%s/'%(star)
    else: savePath = f'{out}{field}/{submask}/{obsdate}/{subfolder}/'

    if not(os.path.isdir(savePath)):
        os.makedirs(savePath)

    if fittype == 'emcee':
        if os.path.isfile(savePath+'%s.deltaLambdaFitMC.npy'%(star)):
            print('delta Lambda MC parameters already fit')
            return
    if fittype == 'ml':
        if os.path.isfile(savePath+'%s.deltaLambdaFit.npy'%(star)):
            print('delta Lambda ML parameters already fit')
            return

    try:
        if not stacked: params = np.load('%s%s/%s/%s.%s.gaussian_params.npy'%(gaussLocation,field,star,field,star))
        else: params = np.load('%s%s/%s/%s/%s/%s.%s.gaussian_params.npy'%(gaussLocation,field,submask,obsdate,subfolder,submask,star))
        parameters.extend(params[:,:2,0])
        errors.extend(params[:,:2,1])
    except:
        if not stacked: print('Missing ARCLINE gaussian fits for %s (slit = %s)'%(field,star))
        else: print('Missing ARCLINE gaussian fits for %s.%s.%s.%s (slit = %s)'%(field,submask,obsdate,subfolder,star))
    try:
        if not stacked: params = np.load('%s%s/%s/%s.%s.gaussian_params_sky.npy'%(gaussLocation,field,star,field,star))
        else: params = np.load('%s%s/%s/%s/%s/%s.%s.gaussian_params_sky.npy'%(gaussLocation,field,submask,obsdate,subfolder,submask,star))
        parameters.extend(params[:,:2,0])
        errors.extend(params[:,:2,1])
    except:
        if not stacked: print('Missing SKYLINE gaussian fits for %s (slit = %s)'%(field,star))
        else: print('Missing SKYLINE gaussian fits for %s.%s.%s.%s (slit = %s)'%(field,submask,obsdate,subfolder,star))

    if len(parameters) == 0:
        raise ValueError('ERROR: No data found for %s'%(field,star))

    parameters = np.array(parameters)
    errors = np.array(errors)

    #if the errors are unrealistically small, make them a more normal size
    errors[errors<0.1] = 0.1

    #remove the points that are too far from the mean or have bad S/N from fitting (plotted in orange later)
    ave = np.average(parameters[:,1],weights=1/np.power(errors[:,1],2))
    keepInds = np.logical_and(np.abs(parameters[:,1]-ave) < 1,parameters[:,1]/errors[:,1] > 3)
    removeInds = np.logical_not(keepInds)

    #first likelihood fit guess
#    deltaLambdaFitML,deltaLambdaFitCov = np.polyfit(parameters[:,0][keepInds],parameters[:,1][keepInds],deg=2,w=1/errors[:,1][keepInds],cov=True) #use 1/sigma weights, not 1/sigma^2 as per documentation
    deltaLambdaFitML,deltaLambdaFitCov = np.polyfit(parameters[:,0][keepInds],parameters[:,1][keepInds],deg=2,w=1/errors[:,1][keepInds],cov=True) #use 1/sigma weights, not 1/sigma^2 as per documentation

    if plotter:
        fitx = np.linspace(4000,11000,num=200)
        fity = np.poly1d(deltaLambdaFitML)(fitx)

    def quad(x,*params):
        lna,b,c,lnf = params
#        return a*x**2 + b*x + c
        return np.exp(lna)*np.power(x-b,2) + c

    firstGuess = deltaLambdaFitML

    #change parameter values to vertext form
    h = -firstGuess[1]/(2*firstGuess[0])
    k = np.poly1d(deltaLambdaFitML)(h)

    firstGuess = np.append(firstGuess,np.log(0.5))
    firsterrors = np.sqrt(np.diag(deltaLambdaFitCov))

    firsterrors = np.append(firsterrors,0)
    if fittype != 'emcee':
        paramResult = []
        paramResult.append([(firstGuess[i],firsterrors[i]) for i in range(len(firstGuess))])
        paramResult = np.array(paramResult)
        np.save(savePath+'%s.deltaLambdaFit.npy'%(star),paramResult)

        if plotter:
            samples = np.random.multivariate_normal(mean=deltaLambdaFitML,cov=deltaLambdaFitCov,size=1000)
            predictedVals = np.polynomial.polynomial.polyval(fitx,samples.T[[2,1,0],:],tensor=True)
            predictedBounds = np.array(list(map(lambda v: [v[0], v[1], v[2], v[3]],zip(*np.percentile(predictedVals, [2.5,16,84,97.5],axis=0)))))

            predictedResVals = fitx/np.polynomial.polynomial.polyval(fitx,samples.T[[2,1,0],:],tensor=True)
            predictedResBounds = np.array(list(map(lambda v: [v[0], v[1], v[2], v[3]],zip(*np.percentile(predictedResVals, [2.5,16,84,97.5],axis=0)))))

            fity = np.poly1d(deltaLambdaFitML)(fitx)

            plt.figure(figsize=[9,6])
            plt.subplot(2,1,1)
            plt.errorbar(parameters[:,0][keepInds],parameters[:,1][keepInds],yerr=errors[:,1][keepInds],xerr=errors[:,0][keepInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.errorbar(parameters[:,0][removeInds],parameters[:,1][removeInds],yerr=errors[:,1][removeInds],xerr=errors[:,0][removeInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.plot(fitx,fity,label='emcee Fit',color='red')
            plt.fill_between(fitx,predictedBounds[:,0],predictedBounds[:,1],interpolate=True,alpha=0.25,color='grey')
            plt.fill_between(fitx,predictedBounds[:,1],predictedBounds[:,2],interpolate=True,alpha=0.25,color='red')
            plt.fill_between(fitx,predictedBounds[:,2],predictedBounds[:,3],interpolate=True,alpha=0.25,color='grey')

            plt.xticks([])
            ax0 = plt.gca()
            ax0.xaxis.set_ticklabels([])
            ax0.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True)
            plt.ylabel("Linewidth ($\AA$)")
            plt.ylim(0.75,2.75)
            plt.xlim(4000,11000)

            plt.subplot(2,1,2)
            resolution = parameters[:,0]/parameters[:,1]
            resolutionErr = np.sqrt(np.power(resolution/parameters[:,0]*errors[:,0],2)+np.power(resolution/parameters[:,1]*errors[:,1],2))
            plt.errorbar(parameters[:,0][keepInds],resolution[keepInds],yerr=resolutionErr[keepInds],xerr=errors[:,0][keepInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.errorbar(parameters[:,0][removeInds],resolution[removeInds],yerr=resolutionErr[removeInds],xerr=errors[:,0][removeInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.plot(fitx,fitx/fity,label='emcee Fit',color='red')
            plt.fill_between(fitx,predictedResBounds[:,0],predictedResBounds[:,1],interpolate=True,alpha=0.25,color='grey')
            plt.fill_between(fitx,predictedResBounds[:,1],predictedResBounds[:,2],interpolate=True,alpha=0.25,color='red')
            plt.fill_between(fitx,predictedResBounds[:,2],predictedResBounds[:,3],interpolate=True,alpha=0.25,color='grey')
            plt.xlabel("Wavelength ($\AA$)")
            plt.ylabel("Resolution")
            plt.ylim(800,7200)
            plt.xlim(4000,11000)
            ax1 = plt.gca()
            ax1.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,top=True)
            plt.subplots_adjust(hspace=0)
            plt.savefig(savePath+'%s.deltaLambdaFit.png'%(star))
            plt.close('all')
        return
    else:

        firstGuess[1],firstGuess[2] = h,k
        def lnl(theta, x, y, yerr):
            lnf = theta[-1]
            model = quad(x,*theta)

            inv_sigma2 = 1.0/(np.power(yerr,2) + np.power(yerr,2)*np.exp(2*lnf))
            return -0.5*(np.sum(np.power((y-model),2)*inv_sigma2 - np.log(inv_sigma2)))

        def lnp(theta):
            lna,b,c,lnf = theta
            if (-30 <= lna <= np.log(5)) and (5000 <= b <= 9000) and (0 <= c <= 3) and (-10.0 <= lnf <= 1.0):
                return 0.0
            else:
                return -np.inf

        def lnpost(theta, x, y, yerr):
            lp = lnp(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnl(theta, x, y, yerr)

        ndim, nwalkers, nsteps = len(firstGuess), 500, 3000

        pos = (np.random.rand(nwalkers,ndim)-np.array([0,0,0,0]))*np.array([2,4000,2,-10])+np.array([0,5000,0,0])
        pos[:,0] = np.log(pos[:,0])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(parameters[:,0][keepInds], parameters[:,1][keepInds], errors[:,1][keepInds]), threads = 1)

        firstTime = time.time()
        for i, result in enumerate(tqdm(sampler.sample(pos, iterations=nsteps),total=nsteps)):
            pass
        print('Full Run:\t%.3f sec'%(time.time()-firstTime))

        burnin = nsteps//2
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        a_samples = np.exp(np.copy(samples[:,0]))
        h_samples = np.copy(samples[:,1])
        k_samples = np.copy(samples[:,2])
        samples[:,0] = a_samples
        samples[:,1] = -2*a_samples*h_samples
        samples[:,2] = k_samples+a_samples*np.power(h_samples,2)

        #save polynomial function in y=ax^2+b*x+c with (a,b,c,lnf)
        emceeResult = map(lambda v: (v[1], v[1]-v[0], v[2]-v[1]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))
        emceeResult = np.array(list(emceeResult))
        np.save(savePath+'%s.deltaLambdaFitMC.npy'%(star),emceeResult)

        if plotter:
            dimLabels = ['$\ln (a)$','$h$','$k$','$\ln (f)$']
            samplerChain = sampler.chain
            plt.figure(figsize=[9,7])
            for dim in range(ndim):
                plt.subplot(ndim,1,dim+1)
                plt.plot(samplerChain[:,:,dim].T,alpha=0.25)
        #        plt.locator_params(axis='y',nticks=3)
                if dim != ndim-1:
                    plt.xticks([])
                else:
                    plt.xticks(np.arange(0, samplerChain.shape[1]+1, samplerChain.shape[1]/10))
                plt.ylabel(dimLabels[dim])
            plt.xlabel('Step Number')
            plt.tight_layout()
            plt.savefig(savePath+'%s.deltaLambdaFit_walkers.png'%(star))
            plt.close('all')
    #        plt.show()

            dimLabels = ['$a$','$b$','$c$','$\ln (f)$']
            plt.figure(figsize=[9,7])
            corner.corner(samples, labels=dimLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                             title_kwargs={"fontsize": 12},bins=100)
            plt.savefig(savePath+'%s.deltaLambdaFit_corner.png'%(star))
            plt.close('all')

            predictedVals = np.polynomial.polynomial.polyval(fitx,samples.T[[2,1,0],:],tensor=True)
            predictedBounds = np.array(list(map(lambda v: [v[0], v[1], v[2], v[3]],zip(*np.percentile(predictedVals, [2.5,16,84,97.5],axis=0)))))

            predictedResVals = fitx/np.polynomial.polynomial.polyval(fitx,samples.T[[2,1,0],:],tensor=True)
            predictedResBounds = np.array(list(map(lambda v: [v[0], v[1], v[2], v[3]],zip(*np.percentile(predictedResVals, [2.5,16,84,97.5],axis=0)))))


            usefulEmcee = np.array([emceeResult[i][0] for i in range(len(emceeResult))])

            fity = np.poly1d(usefulEmcee[:-1])(fitx)

            plt.figure(figsize=[9,6])
            plt.subplot(2,1,1)
            plt.errorbar(parameters[:,0][keepInds],parameters[:,1][keepInds],yerr=errors[:,1][keepInds],xerr=errors[:,0][keepInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.errorbar(parameters[:,0][removeInds],parameters[:,1][removeInds],yerr=errors[:,1][removeInds],xerr=errors[:,0][removeInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.plot(fitx,fity,label='emcee Fit',color='red')
            plt.fill_between(fitx,predictedBounds[:,0],predictedBounds[:,1],interpolate=True,alpha=0.25,color='grey')
            plt.fill_between(fitx,predictedBounds[:,1],predictedBounds[:,2],interpolate=True,alpha=0.25,color='red')
            plt.fill_between(fitx,predictedBounds[:,2],predictedBounds[:,3],interpolate=True,alpha=0.25,color='grey')

            plt.xticks([])
            ax0 = plt.gca()
            ax0.xaxis.set_ticklabels([])
            ax0.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True)
            plt.ylabel("Linewidth ($\AA$)")
            plt.ylim(0.75,2.75)
            plt.xlim(4000,11000)

            plt.subplot(2,1,2)
            resolution = parameters[:,0]/parameters[:,1]
            resolutionErr = np.sqrt(np.power(resolution/parameters[:,0]*errors[:,0],2)+np.power(resolution/parameters[:,1]*errors[:,1],2))
            plt.errorbar(parameters[:,0][keepInds],resolution[keepInds],yerr=resolutionErr[keepInds],xerr=errors[:,0][keepInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.errorbar(parameters[:,0][removeInds],resolution[removeInds],yerr=resolutionErr[removeInds],xerr=errors[:,0][removeInds],fmt='o',ms=1.5,alpha=0.5,zorder=-100)
            plt.plot(fitx,fitx/fity,label='emcee Fit',color='red')
            plt.fill_between(fitx,predictedResBounds[:,0],predictedResBounds[:,1],interpolate=True,alpha=0.25,color='grey')
            plt.fill_between(fitx,predictedResBounds[:,1],predictedResBounds[:,2],interpolate=True,alpha=0.25,color='red')
            plt.fill_between(fitx,predictedResBounds[:,2],predictedResBounds[:,3],interpolate=True,alpha=0.25,color='grey')
            plt.xlabel("Wavelength ($\AA$)")
            plt.ylabel("Resolution")
            plt.ylim(800,7200)
            plt.xlim(4000,11000)
            ax1 = plt.gca()
            ax1.tick_params(axis='both',direction='inout',length=5,bottom=True,left=True,top=True)
            plt.subplots_adjust(hspace=0)
            plt.savefig(savePath+'%s.deltaLambdaFit.png'%(star))
            plt.close('all')

        return


np.random.seed(101)

def singleStarResolution(star, field, spectra_path='.',run_fit_only=False,fittype='emcee',
out='.', stacked=False, fit_sky_only=False, site='keck'):

    '''Takes a star's name in a field, finds all useful files (snr of spectrum > 0) and
    finds/ fits all arclamp and night sky peaks, then fits quadratic resolution function

    Assuming the file heirarchy is as expected, should be able to run the entire fitting process
    by calling: singleStarResolution(STAR,FIELD,path=fullpathways['spectrapath'],run_fit_only=False)

    Should read in the resulting resolution file with:

    resolutionfilename = '%s.deltaLambdaFit.npy'%(STAR)
    deltaLambPolyVals = np.load(resolutionfilename)
    deltaLambPolyVals = deltaLambPolyVals[:,0] #keep only the median parameter values from best fit

    deltaLambFunc = np.poly1d(deltaLambPolyVals[:-1]) #remove lnf parameter because it isn't part of polynomial
    '''

    gaussLocation = f"{out}"

    paths = {}
    starSlit = ''

    if not stacked:

        filepath =  f'{spectra_path}{field}/{star}/'
        for filename in os.listdir(f'{filepath}'):

            if 'spec1d' in filename and 'serendip' not in filename:

                if site == 'keck':
                    starSlit = int(filename.split('.')[2])
                if site == 'magellan':
                    starSlit = star

                spec_tmp=spectrum_class.Spectrum(f'{filepath}{filename}', site=site)

                #h_alpha=(spec_tmp.lam>6500.)&(spec_tmp.lam<6650.)
                #snr=np.median(spec_tmp.flux[h_alpha]*(spec_tmp.ivar[h_alpha])**(1./2.))

                cat = (spec_tmp.lam > 8450.) & (spec_tmp.lam < 8700.)
                snr = np.median(spec_tmp.flux[cat]*(spec_tmp.ivar[cat])**(0.5))

                print(snr)
                if snr>0.:
                    paths[f'{filepath}{filename}'] = starSlit
                else:
                    print(f'ERROR: S/N  = 0 for star {star} on {field}')
                    return

    else:

        filepath = f'{spectra_path}{field}'
        for submask in os.listdir(filepath):
            if submask == '.DS_Store':
                continue
            for obsdate in os.listdir(f'{filepath}/{submask}/'):
                if obsdate == '.DS_Store':
                    continue

                #for subfolder in os.listdir(f'{filepath}/{submask}/{obsdate}'):
                #    if subfolder == '.DS_Store':
                #        continue
                for filename in os.listdir(f'{filepath}/{submask}/{obsdate}/{star}/'):

                    if 'spec1d' in filename and 'serendip' not in filename:
                        starSlit = int(filename.split('.')[2])

                        spec_tmp=spectrum_class.Spectrum(f'{filepath}/{submask}/{obsdate}/{star}/{filename}')
                        h_alpha=(spec_tmp.lam>6500.)&(spec_tmp.lam<6650.)
                        snr=np.median(spec_tmp.flux[h_alpha]*(spec_tmp.ivar[h_alpha])**(1./2.))
                        print(snr)
                        if snr>0.:
                            paths[f'{filepath}/{submask}/{obsdate}/{star}/{filename}'] = starSlit
                        else:
                            print(f'ERROR: S/N  = 0 for star {star} on {field}')
                            pass


    if starSlit == '':
        print(f'ERROR: Cannot locate star {star} along {filepath}')
        return

    if run_fit_only:
        print(f'Fitting resolution fuction of star {star} on {field}')
        fitDeltaLambda(star,field,plotter=False,path=spectra_path,out=out,fittype=fittype)
    else:

        if not stacked:

            datapath = f'{spectra_path}{field}/{star}/'
            for file in paths:

                if not fit_sky_only:

                    if not(os.path.isfile('%s%s/%s/%s.%s.gaussian_params.npy'%(gaussLocation,\
                        field,star,field,star))):

                        print(f'Fitting arclines of star {star} on {field}')
                        fitGaussians(num=paths[file],field=field,submask=star,plotter=False,
                            path=datapath,fittype=fittype, out=out)
                    else:
                        print(f'Arclines of star {star} on {field} already fit.')

            for file in paths:

                if not(os.path.isfile('%s%s/%s/%s.%s.gaussian_params_sky.npy'%(gaussLocation,\
                        field,star,field,star))):

                    print(f'Fitting skylines of star {star} on {field}')
                    skylineFitter(num=paths[file],field=field,submask=star,plotter=False,
                        path=datapath,fittype=fittype, out=out, site=site)
                else:
                    print(f'Skylines of star {star} on {field} already fit.')

            print(f'Fitting resolution fuction of star {star} on {field}')

            fitDeltaLambda(star, field=field, submask=star, plotter=False,
                path=datapath, out=out, fittype=fittype, stacked=stacked)

            return

        else:

            for file in paths:

                print(file)

                temp = file.split('/')
                obsdate = temp[-3]
                subfolder = temp[-2]
                submask = temp[-4]
                slitNum = temp[-1].split('.')[2]
                datapath = '%s%s/%s/%s/'%(spectra_path,field,submask,obsdate)

                if not(os.path.isfile('%s%s/%s/%s/%s/%s.%s.gaussian_params.npy'%(gaussLocation,\
                    field,submask,obsdate,subfolder,submask,slitNum))):

                     print(f'Fitting arclines of star {star} on {submask}/{obsdate}/{subfolder}')

                     fitGaussians(num=paths[file],field=field,obsdate=obsdate,submask=submask,
                        subfolder=subfolder,plotter=False, path=f'{datapath}{subfolder}/',
                        fittype=fittype, stacked=stacked, out=out)
                else:
                    print(f'Arclines of star {star} on {submask}/{obsdate}/{subfolder} already fit.')
                    pass

                if not(os.path.isfile('%s%s/%s/%s/%s/%s.%s.gaussian_params_sky.npy'%(gaussLocation,\
                    field,submask,obsdate,subfolder,submask,slitNum))):

                    print(f'Fitting skylines of star {star} on {submask}/{obsdate}/{subfolder}')

                    skylineFitter(num=paths[file],field=field,obsdate=obsdate,submask=submask,
                        subfolder=subfolder,plotter=False, path=f'{datapath}{subfolder}/',
                        fittype=fittype, stacked=stacked, out=out)

                else:
                    print(f'Skylines of star {star} on {submask}/{obsdate}/{subfolder} already fit.')
                    pass

                print(f'Fitting resolution fuction of star {star} on {submask}/{obsdate}/{subfolder}')
                fitDeltaLambda(star,field=field,obsdate=obsdate,submask=submask,subfolder=subfolder,
                    plotter=False,path=datapath,out=out,fittype=fittype,stacked=stacked)

            return
