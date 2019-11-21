from pathlib import Path as _Path
from typing import List, Tuple

import numpy as np
from astropy import wcs
from astropy import units
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import CircularAperture, detect_threshold, detect_sources, IRAFStarFinder
from scipy import ndimage
from scipy import optimize

from .apertures import distarr
from .gaussfitter import twodgaussian, moments


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


def skybgr(img, imgsize, file, args):

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
            sky, sky_err, flag = _calcSkybgr(largeimg, largeimg.shape[0], img)
        except IOError:
            print(f"Large image of {filename}, does not exist!")
            sky, sky_err, flag = _calcSkybgr(img, imgsize)

    else:
        sky, sky_err, flag = _calcSkybgr(img, imgsize)

    return sky, sky_err, flag


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

    sky, sky_err, and flag : float, float, int
        sky is the sky background measurement.
        sky_err is the error in that measurement.
        flag indicates that the image size was less than ideal.
    '''

    flag = 0
    npix = imgsize
    cenpix = np.array([int(npix/2) + 1, int(npix/2) + 1])
    distarrvar = distarr(npix, npix, cenpix)

    # Define skyregion by fitting a Gaussian to the galaxy and computing on all
    # outwith this
    try:
        if smallimg is not None:
            yfit = gauss2dfit(smallimg, smallimg.shape[0])
        else:
            yfit = gauss2dfit(img, imgsize)
        fact = 2 * np.sqrt(2 * np.log(2))
        fwhm_x = fact * np.abs(yfit[4])
        fwhm_y = fact * np.abs(yfit[5])
        r_in = 2. * max(fwhm_x, fwhm_y)
    except RuntimeError:
        yfit = altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)

    skyind = np.nonzero(distarrvar > r_in)
    skyregion = img[skyind]

    if skyregion.shape[0] < 300:
        # if skyregion too small try with more robust Gaussian fitter
        yfit = altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)
        skyind = np.nonzero(distarrvar > r_in)
        skyregion = img[skyind]
        if skyregion.shape[0] < 100:
            return -99, -99, 1

    if skyregion.shape[0] > 100:
        # Flag the measurement if sky region smaller than 20000 pixels
        # (Simard et al. 2011)
        if skyregion.shape[0] < 20000:
            print(f"Warning! skyregion too small {skyregion.shape[0]}")

        mean_sky = np.mean(skyregion)
        median_sky = np.median(skyregion)
        sigma_sky = np.std(skyregion)

        if mean_sky <= median_sky:
            # non crowded region. Use mean for background measurement
            sky = mean_sky
            sky_err = sigma_sky
        else:
            # crowded region. Use mode for background measurement
            mode_old = 3.*median_sky - 2.*mean_sky
            mode_new = 0.0
            w = 0
            clipsteps = skyregion.shape[0]

            # Begin sigma clipping until convergence
            while w < clipsteps:

                skyind = np.nonzero(np.abs(skyregion - mean_sky) < 3. * sigma_sky)
                skyregion = skyregion[skyind]

                mean_sky = np.mean(skyregion)
                median_sky = np.median(skyregion)
                sigma_sky = np.std(skyregion)
                mode_new = 3.*median_sky - 2.*mean_sky
                mode_diff = np.abs(mode_old - mode_new)

                if mode_diff < 0.01:
                    mode_sky = mode_new
                    w = clipsteps
                else:
                    w += 1
                mode_old = mode_new

            sky = mode_sky
            sky_err = sigma_sky
    else:
        sky = -99
        sky_err = -99
        flag = 1

    return sky, sky_err, flag


def gauss2dfit(img: np.ndarray, imgsize: int) -> List[float]:
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


def altgauss2dfit(img: np.ndarray, imgsize: float) -> models:
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


def maskstarsfinder(img):

    cenpix = np.array([int(img.shape[0]/2) + 1, int(img.shape[1]/2) + 1])
    mean, median, std = sigma_clipped_stats(img, sigma=3.)

    imgclean = np.copy(img)
    daofind = IRAFStarFinder(fwhm=2.0, threshold=5.*std)
    sources = daofind(img - median)
    imgclean = np.copy(img)

    if sources is None:
        return imgclean

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    for i in range(0, positions.shape[0]):
        if abs(positions[i][1] - cenpix[1]) > 10 and abs(positions[i][0] - cenpix[0]) > 10:
            apertures = CircularAperture(positions[i], r=2.2*sources[i]["fwhm"])
            mask = apertures.to_mask(method="center").to_image(img.shape)

            maskvals = mask * np.random.normal(mean, std, size=img.shape)
            imgclean = np.where(mask == 1., maskvals, imgclean)

    return imgclean
