from pathlib import Path as _Path
from typing import List, Tuple

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy import wcs
from photutils import detect_threshold, detect_sources, CircularAperture
from scipy import ndimage
from scipy import optimize

from .apertures import distarr
from .gaussfitter import twodgaussian, moments


__all__ = ["skybgr", "cleanimg", "maskstarsSEG", "maskstarsPSF"]


class _Error(Exception):
    '''Base class for custom exceptions'''
    pass


class _SkyError(_Error):
    '''Class of exception where the sky background has been
       improperly calculated '''

    def __init__(self, value):
        print(value)
        raise AttributeError


def inBbox(extent: List[float], point: List[float]) -> bool:
    ''' Check if a point is in a box defined by the bounds extent

    Parameters
    ----------

    extent: List[float]
        The extent of the bounding box. indices 0, 1 refer to xmin, xmax.
        2, 3 refer to ymin, ymax.
    point: List[float]
        The point that is to be checked.

    Returns
    -------

    bool

    '''

    if point[0] > extent[0] and point[1] > extent[2]:
        if point[0] < extent[1] and point[1] < extent[3]:
            return True

    return False


def skybgr(img: np.ndarray, imgsize: int, file, args) -> Tuple[float]:
    '''Helper function for calculating skybgr

    Parameters
    ----------

    img: np.ndarray

    imgsize: int

    file: Path object

    args: argpasre object

    Returns
    -------

    sky, sky_err: Tuple[float]

    '''

    if args.largeimage:
        filename = file.name
        if args.imgsource == "sdss":
            filename = filename.replace("sdss", "sdssl", 1)
        elif args.imgsource == "hsc":
            filename = filename.replace("hsc", "hscl", 1)

        infile = file.parents[0] / _Path(filename)
        try:
            largeimg = fits.getdata(infile)
            # clean image so that skybgr not over estimated
            largeimg = maskstarsSEG(largeimg)
            sky, sky_err = _calcSkybgr(largeimg, largeimg.shape[0], img)
        except IOError:
            print(f"Large image of {filename}, does not exist!")
            sky, sky_err = _calcSkybgr(img, imgsize)

    else:
        sky, sky_err = _calcSkybgr(img, imgsize)

    return sky, sky_err


def _calcSkybgr(img: np.ndarray, imgsize: int, smallimg=None) -> Tuple[float, float, int]:
    '''Function that calculates the sky background value.

    Parameters
    ----------
    img : np.ndarray
        Image data from which the background will be calculated.
    imgsize : int
        Size of the image.
    smallimg : np.ndarray, optional
        Smaller cutout of object. If provided Gaussian is fitted on this
        instead of larger cutout as the fitter sometimes fails on larger images.

    Returns
    -------

    sky, sky_err : float, float, int
        sky is the sky background measurement.
        sky_err is the error in that measurement.
    '''

    npix = imgsize
    cenpix = np.array([int(npix/2) + 1, int(npix/2) + 1])
    distarrvar = distarr(npix, npix, cenpix)

    # Define skyregion by fitting a Gaussian to the galaxy and computing on all
    # outwith this
    try:
        if smallimg is not None:
            yfit = _gauss2dfit(smallimg, smallimg.shape[0])
        else:
            yfit = _gauss2dfit(img, imgsize)
        fact = 2 * np.sqrt(2 * np.log(2))
        fwhm_x = fact * np.abs(yfit[4])
        fwhm_y = fact * np.abs(yfit[5])
        r_in = 2. * max(fwhm_x, fwhm_y)
    except RuntimeError:
        yfit = _altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)

    skyregion = distarrvar < r_in
    skymask = np.ma.array(img, mask=skyregion)

    if skymask.count() < 300:
        # if skyregion too small try with more robust Gaussian fitter
        yfit = _altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)
        skyregion = distarrvar < r_in
        skymask = np.ma.array(img, mask=skyregion)

    if skymask.count() < 100:
        raise _SkyError(f"Error! Sky region too small {skymask.count()}")

    # Flag the measurement if sky region smaller than 20000 pixels
    # (Simard et al. 2011)
    if skymask.count() < 20000:
        print(f"Warning! skyregion smaller than optimal {skymask.count()}")

    mean_sky = np.ma.mean(skymask)
    median_sky = np.ma.median(skymask)
    sigma_sky = np.ma.std(skymask)

    if mean_sky <= median_sky:
        # non crowded region. Use mean for background measurement
        sky = mean_sky
        sky_err = sigma_sky
    else:
        mean_sky, median_sky, sigma_sky = sigma_clipped_stats(img, mask=skyregion, sigma=3.)
        sky = 3.*median_sky - 2.*mean_sky
        sky_err = sigma_sky
    return sky, sky_err


def _gauss2dfit(img: np.ndarray, imgsize: int) -> List[float]:
    '''Function that fits a 2D Gaussian to image data.

    Parameters
    ----------

    img : np.ndarray
        Image array to be fitted to.
    imgsize : int
        Size of image in x and y directions.

    Returns
    -------

    popt : List[float]
        List of calculated parameters from curve_fit.

    '''

    x = np.linspace(0, imgsize-1, imgsize)
    y = np.linspace(0, imgsize-1, imgsize)
    X, Y = np.meshgrid(x, y)

    # guess intital parameters from calculation of moments
    initial_guess = moments(img)

    # convert image to 1D array for fitting purposes
    imgravel = img.ravel()

    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pconv = optimize.curve_fit(twodgaussian, (X, Y), imgravel, p0=initial_guess)

    return np.abs(popt)


def _altgauss2dfit(img: np.ndarray, imgsize: float) -> models:
    '''Alternative slower but more robust Gaussian fitter.

    img : np.ndarray
        Image array to be fitted to.
    imgsize : int
        Size of image in x and y directions.

    Returns
    -------

    g : astropy.modeling.functional_models.Gaussian2D class
        Model parameters of fitted Gaussian.

    '''

    imgold = np.copy(img)

    fit_w = fitting.LevMarLSQFitter()

    total = np.abs(imgold).sum()
    Y, X = np.indices(imgold.shape)  # python convention: reverse x,y np.indices
    mean = np.mean(imgold)
    xmid = int(imgsize / 2)
    ymid = int(imgsize / 2)

    # Mask out bright pixels not near object centre
    while True:
        x, y = np.unravel_index(np.argmax(imgold), shape=imgold.shape)
        if abs(x - xmid) > 20 or abs(y - ymid) > 20:
            imgold[x, y] = mean
        else:
            break

    # subtracting the mean sometimes helps
    mean = np.mean(imgold)
    imgold -= mean

    # make intial guess
    total = np.abs(imgold).sum()
    y0 = int(img.shape[0] / 2)
    x0 = int(img.shape[1] / 2)

    col = imgold[int(y0), :]
    sigmax = np.sqrt(np.abs((np.arange(col.size)-y0)**2*col).sum() / np.abs(col).sum())

    row = imgold[:, int(x0)]
    sigmay = np.sqrt(np.abs((np.arange(row.size)-x0)**2*row).sum() / np.abs(row).sum())

    height = np.median(imgold.ravel())
    amp = imgold.max() - height
    angle = 3.14 / 2.

    # fit model
    w = models.Gaussian2D(amp, x0, y0, sigmax, sigmay, angle)
    yi, xi = np.indices(imgold.shape)
    g = fit_w(w, xi, yi, imgold)

    return g


def cleanimg(img: np.ndarray, pixmap: np.ndarray, filter=False) -> np.ndarray:
    '''Function that cleans the image of non overlapping sources, i.e outside
       the objects binary pixelmap


    Parameters
    ----------

    img : np.ndarray
        Input image to be cleaned.
    pixmap : np.ndarray
        Binary pixelmap of object.
    filter : bool, optional
        Flag for whether the output image should be filtered before return.
        Helps wth "salt and pepper noise"

    Returns
    -------

    imgclean : np.ndarray
        Cleaned image.

    '''

    imgsize = img.shape[0]
    imgravel = img.ravel()

    # Dilate pixelmap
    element = np.ones((9, 9))
    mask = ndimage.morphology.binary_dilation(pixmap, structure=element)

    skyind = np.nonzero(mask.ravel() != 1)[0]

    # find a threshold for defining sky pixels
    meansky = np.mean(imgravel[skyind])
    mediansky = np.median(imgravel[skyind])

    if meansky <= mediansky:
        thres = meansky
    else:
        # begin sigma clip
        sigmasky = np.std(imgravel[skyind])

        mode_old = 3.*mediansky - 2.*meansky
        mode_new = 0.0
        w = 0
        clipsteps = imgravel.size

        while w < clipsteps:

            skyind = np.nonzero(np.abs(imgravel[skyind] - meansky) < 3.*sigmasky)
            meansky = np.mean(imgravel[skyind])
            mediansky = np.median(imgravel[skyind])
            sigmasky = np.std(imgravel[skyind])

            mode_new = 3.*mediansky - 2.*meansky
            mode_diff = abs(mode_old - mode_new)

            if mode_diff < 0.01:
                modesky = mode_new.copy()
                w = clipsteps
            else:
                w += 1

            mode_old = mode_new.copy()

        thres = modesky

    # Mask out sources with random sky pixels
    # element wise boolean AND
    skypix = np.nonzero((pixmap.ravel() != 1) & (imgravel > thres))

    skypixels = imgravel[skypix]
    allpixels = skypixels

    if skypixels.size >= imgravel.size:
        print("ERROR! No sources detected! Check image and pixel map.")
    else:
        pixfrac = imgravel.size / skypixels.size
        if pixfrac == float(round(pixfrac)):
            for p in range(1, int(pixfrac)):
                allpixels = np.append(allpixels, skypixels)
        else:
            for p in range(1, int(pixfrac)):
                allpixels = np.append(allpixels, skypixels)
            diff = imgravel.shape[0] - allpixels.shape[0]
            allpixels = np.append(allpixels, skypixels[0:diff])

    # shuffle sky pixels randomly
    # np.random.seed(0)
    np.random.shuffle(allpixels)

    imgclean = np.where((pixmap.ravel() != 1) & (imgravel >= thres), allpixels, imgravel)
    imgclean = imgclean.reshape((imgsize, imgsize))

    # convert imgclean from object dtype to float dtype
    imgclean[imgclean == None] = 0.
    imgclean = imgclean.astype(np.float)

    if filter:
        imgclean = ndimage.uniform_filter(imgclean, size=(3, 3), mode="reflect")

    return imgclean


def maskstarsSEG(img):

    cenpix = np.array([int(img.shape[0]/2) + 1, int(img.shape[1]/2) + 1])
    mean, median, std = sigma_clipped_stats(img, sigma=3.)

    imgclean = np.copy(img)
    threshold = detect_threshold(img, 1.5)
    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(img, threshold, npixels=8, filter_kernel=kernel)

    stars = []
    for i, segment in enumerate(segm.segments):

        if not inBbox(segment.bbox.extent, cenpix):
            stars.append(i)

    for i in stars:
        masked = segm.segments[i].data_ma
        masked = np.where(masked > 0., 1., 0.) * np.random.normal(mean, std, size=segm.segments[i].data.shape)
        imgclean[segm.segments[i].bbox.slices] = masked

        imgclean = np.where(imgclean == 0, img, imgclean)

    return imgclean


def circle_mask(shape, centre, radius):
    '''Return a boolean mask for a circle.'''

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx, y-cy)

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= 2.*np.pi

    return circmask*anglemask


def calcR(psf, counts, sigma):
    return sigma * np.sqrt(2. * np.log(counts / psf))


def maskstarsPSF(img, objs, header, skyCount):

    s1 = header["PSF_S1"]
    s2 = header["PSF_S2"]
    exptime = header["EXPTIME"]
    aa = header["PHT_AA"]
    kk = header["PHT_KK"]
    airmass = header["AIRMASS"]
    skyCount = header["SKY"]

    fact = 0.4*(aa + kk*airmass)

    try:
        bzero = header["BZERO"]
    except KeyError:
        bzero = 0

    w = wcs.WCS(header)
    if len(objs) > 0:
        mask = np.zeros_like(img)
    else:
        mask = np.ones_like(img)

    for obj in objs:
        pos = SkyCoord(obj[0], obj[1], unit="deg")
        pixelPos = wcs.utils.skycoord_to_pixel(pos, wcs=w)
        x, y = pixelPos

        x = round(float(x))
        y = round(float(y))

        # objectCount = np.amax(img[y-1:y+1, x-1:x+1])
        # objectFluxRatio = (objectCount / exptime) * 10**(fact)
        objectMag = obj[3]#-2.5 * np.log10(objectFluxRatio)

        skyFluxRatio = ((skyCount - bzero) / exptime) * 10**(fact)
        skyMag = -2.5 * np.log10(skyFluxRatio)

        sigma = max(s1, s2)
        radius = calcR(objectMag, skyMag, sigma)

        aps = CircularAperture(pixelPos, r=5*radius)
        aperMask = np.zeros_like(img)
        masks = aps.to_mask(method="subpixel")
        aperMask = np.where(masks.to_image(img.shape) > 0., 1., 0.)
        # aperMask = ndimage.morphology.binary_dilation(aperMask, border_value=0, iterations=3)

        newMask = circle_mask(mask.shape, (pixelPos[1], pixelPos[0]), 5.*radius)
        # newMask = ndimage.morphology.binary_dilation(newMask, border_value=0, iterations=3)
        mask = np.logical_or(mask, newMask)
        mask = np.logical_or(mask, aperMask)

    if len(objs) > 0:
        mask = ~mask

    return mask
