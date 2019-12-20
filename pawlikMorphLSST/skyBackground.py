from pathlib import Path as _Path
from typing import List, Tuple
import warnings

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.utils.exceptions import AstropyWarning
from photutils import aperture, detect_sources, detect_threshold
from scipy import optimize

from .apertures import distarr
from .gaussfitter import twodgaussian, moments
from .imageutils import maskstarsSEG

__all__ = ["skybgr"]


class _Error(Exception):
    '''Base class for custom exceptions'''
    pass


class _SkyError(_Error):
    '''Class of exception where the sky background has been
       improperly calculated '''

    def __init__(self, value):
        print(value)
        raise AttributeError


# TODO make file optional
def skybgr(img: np.ndarray, file=None, largeImage=False,
           imageSource=None) -> Tuple[float, float, List[float], float]:
    '''Helper function for calculating skybgr

    Parameters
    ----------

    img: np.ndarray
        image from which a sky background will be calculated

    file: Path object, optional
        Path to image

    largeImage : bool, optional
        If true algorithm uses larger image to estimate sky background.
        If False uses provided image.

    imageSource : str, optional
        Telescope source of the image. Default is SDSS

    Returns
    -------

    sky : float
        Estimation of the sky background value in counts

    sky_err : float
        Error in sky background value in counts

    fwhms : List[float]
        FWHM in x and y direction of the fitted Gaussian.

    theta : float
        Angle of the fitted Gaussian in radians measured from the +ive x axis
        anticlockwise

    '''

    imgsize = img.shape[0]

    if largeImage:
        filename = file.name
        if imageSource != "none":
            filename = filename.replace(imageSource, imageSource+"l", 1)
        else:
            filename = filename.replace("cutout", "cutoutl", 1)

        infile = file.parents[0] / _Path(filename)
        try:
            with warnings.catch_warnings():
                # ignore invalid card warnings
                warnings.simplefilter('ignore', category=AstropyWarning)
                largeimg = fits.getdata(infile)
            # clean image so that skybgr not over estimated
            largeimg = maskstarsSEG(largeimg)
            sky, sky_err, fwhms, theta = _calcSkybgr(largeimg, largeimg.shape[0], img)
        except IOError:
            print(f"Large image of {filename}, does not exist!")
            sky, sky_err, fwhms, theta = _calcSkybgr(img, imgsize)

    else:
        sky, sky_err, fwhms, theta = _calcSkybgr(img, imgsize)

    return sky, sky_err, fwhms, theta


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

    # check if there are any objects in the image
    sigma = 3.0 * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    threshold = detect_threshold(img, 1.5)
    segm = detect_sources(img, threshold, npixels=8, filter_kernel=kernel)

    if segm is None:
        raise _SkyError("No object in image")

    # Define skyregion by fitting a Gaussian to the galaxy and computing on all
    # outwith this
    try:
        r_in, fwhms, theta = _fitGauss(img, smallimg)
    except RuntimeError:
        r_in, fwhms, theta = _fitAltGauss(img, smallimg)

    skyMask, skyRegion = _getSkyRegion(img, distarrvar, r_in, fwhms, cenpix, theta)

    # if r_in massive then image may contain on object
    if skyMask.count() < 300 or r_in > imgsize * 3:
        # if skyregion too small try with more robust Gaussian fitter
        r_in, fwhms, theta = _fitAltGauss(img, smallimg)
        skyMask, skyRegion = _getSkyRegion(img, distarrvar, r_in, fwhms, cenpix, theta)

    if skyMask.count() < 100:
        raise _SkyError(f"Error! Sky region too small {skyMask.count()}")
        sys.exit()
    # Flag the measurement if sky region smaller than 20000 pixels
    # (Simard et al. 2011)
    if skyMask.count() < 20000:
        print(f"Warning! skyregion smaller than optimal {skyMask.count()}")

    mean_sky = np.ma.mean(skyMask)
    median_sky = np.ma.median(skyMask)
    sigma_sky = np.ma.std(skyMask)

    if mean_sky <= median_sky:
        # non crowded region. Use mean for background measurement
        sky = mean_sky
        sky_err = sigma_sky
    else:
        mean_sky, median_sky, sigma_sky = sigma_clipped_stats(img, mask=skyRegion, sigma=3.)
        sky = 3.*median_sky - 2.*mean_sky
        sky_err = sigma_sky
    return sky, sky_err, fwhms, theta


def _fitGauss(img: np.ndarray, smallimg: bool) -> Tuple[float, List[float], float]:
    '''Helper function that calls a fast, non robust Gaussian fitting routine

    Parameters
    ----------

    img : np.ndarray
        img to which a Gaussian will be fitted in to determine sky bgr value

    smallimg : bool
        Boolean that determines if a larger image should be used to calculate
        the skybackground value

    Returns
    -------

    r_in : float
        Radius is equal to 2 times the largest FWHM.

    fwhms : List[float]
        FWHM for the x and y direction of the fitted Gaussian.

    theta : float
        Angle of fitted Gaussian, measured in radian from the +ive x axis anticlockwise.

    '''

    if smallimg is not None:
        yfit = _gauss2dfit(smallimg, smallimg.shape[0])
    else:
        imgsize = img.shape[0]
        yfit = _gauss2dfit(img, imgsize)

    factor = 2 * np.sqrt(2 * np.log(2))  # to convert sigma to a fwhm
    fwhm_x = factor * np.abs(yfit[4])
    fwhm_y = factor * np.abs(yfit[5])
    # define skyregion as everything outwith 2 time the largest FWHM.
    r_in = 2. * max(fwhm_x, fwhm_y)
    theta = yfit[6]

    # convert theta to radians and correct quadrant for Sersic fit
    # Theta is now measured anticlockwise from +ive x axis
    if theta > (np.pi/2.):
        theta = np.pi - theta
    else:
        thetaDeg = theta / (2.*np.pi)
        n = int(thetaDeg)
        r = (thetaDeg) - n
        theta = (r * 2.*np.pi) - (np.pi/2.)

    return r_in, [fwhm_x, fwhm_y], theta


def _fitAltGauss(img: np.ndarray, smallimg: bool) -> Tuple[float, List[float], float]:
    '''Helper function that calls a slower, robust Gaussian fitting routine

    Parameters
    ----------

    img : np.ndarray
        img to which a Gaussian will be fitted in to determine sky bgr value

    smallimg : bool
        Boolean that determines if a larger image should be used to calculate
        the skybackground value

    Returns
    -------

    r_in : float
        Radius is equal to 2 times the largest FWHM.

    fwhms : List[float]
        FWHM for the x and y direction of the fitted Gaussian.

    theta : float
        Angle of fitted Gaussian, measured in radian from the +ive x axis anticlockwise.

    '''

    if smallimg is not None:
        yfit = _altgauss2dfit(smallimg, smallimg.shape[0])
    else:
        imgsize = img.shape[0]
        yfit = _altgauss2dfit(img, imgsize)

    fwhm_x = yfit.x_fwhm
    fwhm_y = yfit.y_fwhm
    r_in = 2. * max(fwhm_x, fwhm_y)
    theta = yfit.theta.value

    return r_in, [fwhm_x, fwhm_y], theta


def _getSkyRegion(image: np.ndarray, distarrvar: np.ndarray, radius: float, fwhms, cenpix, theta) -> Tuple[np.ndarray, np.ndarray]:
    '''Function to get sky region as a numpy mask array

    Parameters
    ----------

    image : np.ndarray
        image to mask sky region on

    distarrvar : np.ndarray
        array of distances from centre of object

    radius : float
        Calculated radius of object.

    Returns
    -------

    skymask : np.ndarray mask
        mask that defines the skyregion as True, and object region as False

    skyregion : np.ndarray
        Array of bools that describes where the object is


    '''

    # Get sky region and mask
    a = 2. * min(fwhms)
    b = 2. * max(fwhms)

    ellipAper = aperture.EllipticalAperture(cenpix, a, b, theta=theta)
    ellipMask = ellipAper.to_mask(method="center")
    skyregion = (ellipMask.to_image(image.shape).astype(bool))

    skymask = np.ma.array(image, mask=skyregion)

    return skymask, skyregion


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

    # guess initial parameters from calculation of moments
    initial_guess = moments(img)

    # convert image to 1D array for fitting purposes
    imgravel = img.ravel()

    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pconv = optimize.curve_fit(twodgaussian, (X, Y), imgravel, p0=initial_guess)

    return np.abs(popt)


def _altgauss2dfit(image: np.ndarray, imgsize: float) -> models:
    '''Alternative slower but more robust Gaussian fitter.

    image : np.ndarray
        Image array to be fitted to.
    imgsize : int
        Size of image in x and y directions.

    Returns
    -------

    fit_parameters : astropy.modeling.functional_models.Gaussian2D class
        Model parameters of fitted Gaussian.

    '''

    imageOld = np.copy(image)

    fit_w = fitting.LevMarLSQFitter()

    total = np.abs(imageOld).sum()
    Y, X = np.indices(imageOld.shape)  # python convention: reverse x,y np.indices
    mean = np.mean(imageOld)
    xmid = int(imgsize / 2)
    ymid = int(imgsize / 2)

    # Mask out bright pixels not near object centre
    while True:
        x, y = np.unravel_index(np.argmax(imageOld), shape=imageOld.shape)
        if abs(x - xmid) > 20 or abs(y - ymid) > 20:
            imageOld[x, y] = mean
        else:
            break

    # subtracting the mean sometimes helps
    mean = np.mean(imageOld)
    imageOld -= mean

    # make initial guess
    total = np.abs(imageOld).sum()
    y0 = int(image.shape[0] / 2)
    x0 = int(image.shape[1] / 2)

    col = imageOld[int(y0), :]
    sigmax = np.sqrt(np.abs((np.arange(col.size)-y0)**2*col).sum() / np.abs(col).sum())

    row = imageOld[:, int(x0)]
    sigmay = np.sqrt(np.abs((np.arange(row.size)-x0)**2*row).sum() / np.abs(row).sum())

    height = np.median(imageOld.ravel())
    amp = imageOld.max() - height
    angle = 3.14 / 2.

    # fit model
    w = models.Gaussian2D(amp, x0, y0, sigmax, sigmay, angle)
    yi, xi = np.indices(imageOld.shape)
    fit_parameters = fit_w(w, xi, yi, imageOld)

    return fit_parameters
