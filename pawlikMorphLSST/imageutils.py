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


__all__ = ["skybgr", "maskstarsSEG", "maskstarsPSF"]


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


def skybgr(img: np.ndarray, imgsize: int, file, largeImage: bool,
           imageSource: str) -> Tuple[float]:
    '''Helper function for calculating skybgr

    Parameters
    ----------

    img: np.ndarray

    imgsize: int

    file: Path object

    largeImage : bool

    imgSource : str


    Returns
    -------

    sky, sky_err: Tuple[float]

    '''

    if largeImage:
        filename = file.name
        if imageSource == "sdss":
            filename = filename.replace("sdss", "sdssl", 1)
        elif imageSource == "hsc":
            filename = filename.replace("hsc", "hscl", 1)

        infile = file.parents[0] / _Path(filename)
        try:
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

    # Define skyregion by fitting a Gaussian to the galaxy and computing on all
    # outwith this
    # try:
    #     if smallimg is not None:
    #         yfit = _gauss2dfit(smallimg, smallimg.shape[0])
    #     else:
    #         yfit = _gauss2dfit(img, imgsize)
    #     fact = 2 * np.sqrt(2 * np.log(2))
    #     fwhm_x = fact * np.abs(yfit[4])
    #     fwhm_y = fact * np.abs(yfit[5])
    #     r_in = 2. * max(fwhm_x, fwhm_y)
    #     theta = yfit[6]
    # except RuntimeError:
    if smallimg is not None:
        yfit = _altgauss2dfit(smallimg, smallimg.shape[0])
    else:
        yfit = _altgauss2dfit(img, imgsize)
    fwhm_x = yfit.x_fwhm
    fwhm_y = yfit.y_fwhm
    r_in = 2. * max(fwhm_x, fwhm_y)
    theta = yfit.theta

    skyregion = distarrvar > r_in
    skymask = np.ma.array(img, mask=skyregion)

    if skymask.count() < 300:
        # if skyregion too small try with more robust Gaussian fitter
        if smallimg is not None:
            yfit = _altgauss2dfit(smallimg, smallimg.shape[0])
        else:
            yfit = _altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)
        skyregion = distarrvar < r_in
        skymask = np.ma.array(img, mask=skyregion)
        theta = yfit.theta

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
    return sky, sky_err, [fwhm_x, fwhm_y], theta.value


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

    g : astropy.modeling.functional_models.Gaussian2D class
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
    g = fit_w(w, xi, yi, imageOld)

    return g


def maskstarsSEG(image: np.ndarray):
    '''Function that cleans image of external sources. Uses segmentation map
       to achieve this.

    Parameters
    ----------

    image : np.ndarray
        Image to be cleaned.

    Returns
    -------

    imageClean : np.ndarray
        Image cleaned of external sources.

    '''

    cenpix = np.array([int(image.shape[0]/2) + 1, int(image.shape[1]/2) + 1])
    mean, median, std = sigma_clipped_stats(image, sigma=3.)

    # create segmentation map
    imageClean = np.copy(image)
    threshold = detect_threshold(image, 1.5)
    sigma = 3.0 * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(image, threshold, npixels=8, filter_kernel=kernel)

    # Save potions of segmentation map outwith object of interest
    stars = []
    for i, segment in enumerate(segm.segments):

        if not inBbox(segment.bbox.extent, cenpix):
            stars.append(i)

    # clean image of external sources
    for i in stars:
        masked = segm.segments[i].data_ma
        masked = np.where(masked > 0., 1., 0.) * np.random.normal(mean, std, size=segm.segments[i].data.shape)
        imageClean[segm.segments[i].bbox.slices] = masked

        imageClean = np.where(imageClean == 0, image, imageClean)

    return imageClean


def _circle_mask(shape: Tuple[int], centre: Tuple[int], radius: float):
    '''Return a boolean mask for a circle.

    Parameters
    ----------

    shape : Tuple[int]
        Dimensions of image on what to create a circular mask.

    centre : Tuple[int]
        Centre of where mask should be created.

    radius : float
        radius of circular mask to be created.

    Returns
    -------

    Circular mask : np.ndarray

    '''

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


def _calculateRadius(psf, counts, sigma):
    return sigma * np.sqrt(2. * np.log(counts / psf))


def maskstarsPSF(image: np.ndarray, objs: List, header, skyCount: float, 
                 numSigmas=5.) -> np.ndarray:
    '''Function that uses the PSF to mask stars.

    Parameters
    ----------

    image : np.ndarray
        Image that is to be masked of nuisance stars.

    objs : List[float, float, str, float]
        List of objects. [RA, DEC, type, psfMag_r]

    header : astropy.io.fits.header.Header
        The header of the current image. Contains information on PSF and
        various other parameters.

    skyCount : float
        Sky background in counts.

    numSigmas : optional, float
        Number of sigmas that the stars radius should extend to

    Returns
    -------

    mask : np.ndarray
        Array that masks stars on original image.

    '''

    sigma1 = header["PSF_S1"]          # Sigma of Gaussian fit of PSF
    sigma2 = header["PSF_S2"]          # Sigma of Gaussian fit of PSF
    expTime = header["EXPTIME"]        # Exposure time of image
    aa = header["PHT_AA"]              # zero point
    kk = header["PHT_KK"]              # extinction coefficient
    airMass = header["AIRMASS"]        # airmass
    softwareBias = header["SOFTBIAS"]  # software bias added to pixel counts

    skyCount -= softwareBias

    # https://classic.sdss.org/dr7/algorithms/fluxcal.html#counts2mag
    factor = 0.4*(aa + kk*airMass)
    skyFluxRatio = ((skyCount) / expTime) * 10**(factor)
    skyMag = -2.5 * np.log10(skyFluxRatio)

    wcsFromHeader = wcs.WCS(header)

    # if no objects create empty mask that wont interfere with future calculations
    if len(objs) > 0:
        mask = np.zeros_like(image)
    else:
        mask = np.ones_like(image)

    for obj in objs:
        # convert object RA, DEC to pixels
        pos = SkyCoord(obj[0], obj[1], unit="deg")
        pixelPos = wcs.utils.skycoord_to_pixel(pos, wcs=wcsFromHeader)
        x, y = pixelPos

        # get object psfMag_r
        objectMag = obj[3]

        # calculate radius of star
        sigma = max(sigma1, sigma2)
        radius = numSigmas * _calculateRadius(objectMag, skyMag, sigma)

        # mask out star
        aps = CircularAperture(pixelPos, r=radius)
        masks = aps.to_mask(method="subpixel")
        aperMask = np.where(masks.to_image(image.shape) > 0., 1., 0.)

        newMask = _circle_mask(mask.shape, (pixelPos[1], pixelPos[0]), radius)
        mask = np.logical_or(mask, newMask)
        mask = np.logical_or(mask, aperMask)

    # invert calculated mask so that future calculations work
    if len(objs) > 0:
        mask = ~mask

    return mask
