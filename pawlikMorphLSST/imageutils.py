import warnings
from typing import List, Tuple

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
from astropy import wcs
from photutils import detect_threshold, detect_sources, CircularAperture


__all__ = ["maskstarsSEG", "maskstarsPSF"]


def _inBbox(extent: List[float], point: List[float]) -> bool:
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


def maskstarsSEG(image: np.ndarray) -> np.ndarray:
    """ 'Cleans' image of external sources.
        For example will remove all stars that do not interfere with the object of interest.

    Parameters
    ----------

    image : np.ndarray
        Image to be cleaned.

    Returns
    -------

    imageClean : np.ndarray
        Image cleaned of external sources.

    """

    cenpix = np.array([int(image.shape[0]/2) + 1, int(image.shape[1]/2) + 1])
    mean, median, std = sigma_clipped_stats(image, sigma=3.)

    # create segmentation map
    # TODO give user option to specify kernel, size of kernel and threshold?
    imageClean = np.copy(image)
    threshold = detect_threshold(image, 1.5)
    sigma = 3.0 * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(image, threshold, npixels=8, filter_kernel=kernel)

    # if no sources return
    if segm is None:
        return imageClean

    # Save potions of segmentation map outwith object of interest
    stars = []
    for i, segment in enumerate(segm.segments):

        if not _inBbox(segment.bbox.extent, cenpix):
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


def _calculateRadius(y: float, A: float, sigma: float) -> float:
    r''' Function that calculates the radius of a Gaussian for a given y
        Standard Gaussian
        .. math:
            y = A * exp(x^2/(s*sig^2))
        Rearange for s
        .. math:
            \sigma * \sqrt{\left|2 log_e(\frac{A}{y})\right|}


    Parameters
    ----------

    y : float
        y parameter of Gaussian function.

    A : float
        Amplitude of Gaussian function.

    sigma : float
        Sigma parameters for Gaussian function.

    Returns
    -------

    Radius : float
        Radius of Gaussian

    '''

    return sigma * np.sqrt(abs(2. * np.log(A / y)))


def maskstarsPSF(image: np.ndarray, objs: List, header, skyCount: float,
                 numSigmas=5., adaptive=True, sky_err=0.0) -> np.ndarray:
    '''Use the PSF to estimate stars radius, then masks them out.

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
    b = header["PHT_B"]                # softening parameter

    skyCount -= softwareBias

    # https://classic.sdss.org/dr7/algorithms/fluxcal.html#counts2mag
    factor = 0.4*(aa + kk*airMass)
    skyFluxRatio = ((skyCount) / expTime) * 10**(factor)
    skyMag = -(2.5 / np.log(10.)) * (np.arcsinh((skyFluxRatio) / (2*b)) + np.log(b))

    with warnings.catch_warnings():
        # ignore invalid card warnings
        warnings.simplefilter('ignore', category=AstropyWarning)
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

        # get object psfMag_r
        objectMag = obj[3]

        # calculate radius of star
        sigma = max(sigma1, sigma2)
        radius = numSigmas * _calculateRadius(skyMag, objectMag, sigma)

        # mask out star
        # aps = CircularAperture(pixelPos, r=radius)
        # masks = aps.to_mask(method="subpixel")
        # aperMask = np.where(masks.to_image(image.shape) > 0., 1., 0.)

        newMask = _circle_mask(mask.shape, (pixelPos[1], pixelPos[0]), radius)
        mask = np.logical_or(mask, newMask)
        # mask = np.logical_or(mask, aperMask)
        if adaptive:
            extraExtent = _calculateAdaptiveRadius(mask, pixelPos, image, skyCount, sky_err)

            aps = CircularAperture(pixelPos, r=radius+extraExtent)
            masks = aps.to_mask(method="subpixel")
            aperMask = np.where(masks.to_image(image.shape) > 0., 1., 0.)
            mask = np.logical_or(mask, aperMask)
            tmp = image*~mask + ((skyCount+1000)*mask)  # not used??

    # invert calculated mask so that future calculations work
    if len(objs) > 0:
        mask = ~mask

    return mask


def _calculateAdaptiveRadius(mask: np.ndarray, pixelPos: List[float],
                             image: np.ndarray, skyCount: float,
                             sky_err: float) -> int:
    '''Function calculates the "real" radius of the star to mask out to.
        Achieves this by moving from the centre of the star in the opposite
        direction of the object of interest, and checking if there is a large
        positive discontinuity in this line profile of the flux.

    Parameters
    ----------

    mask : np.ndarray
        Pixelmap of object

    pixelPos : List[float]
        x and y position of star to be masked.

    image : np.ndarray
        Image in which the star is to be masked.

    skyCount : float
        Sky background value.

    sky_err : float
        Sky background error.

    Returns
    -------

    extraExtent : int
        Extra radius in pixels to mask star to.

    '''

    # calculate masked image by giving the stars mask the value of the sky (unsoftware biased)
    maskedImage = image*~mask + ((skyCount+1000)*mask)
    yinit = maskedImage[int(pixelPos[1]), int(pixelPos[0])]
    imgsize = image.shape[0]

    # get direction to move in
    if pixelPos[1] > int(imgsize/2):
        delta = 1
    else:
        delta = -1

    # if on edge return
    if pixelPos[1] >= imgsize-1 or pixelPos[1] <= 0:
        return 0

    ypos = int(pixelPos[1]) + delta
    ys = [yinit]

    # move along line away from analysis
    for i in range(int(imgsize / 2)):

        ys.append(maskedImage[ypos, int(pixelPos[0])])
        ypos += delta
        if ypos >= imgsize or ypos < 0:
            break

    # if flux profile is above threshold then we mask out to this distance
    thres = np.mean(ys) + sky_err
    arr = np.where(ys > thres)[0]
    for i, val in enumerate(arr):
        if i+1 > arr.shape[0]-1:
            break
        if val + 1 != arr[i+1]:
            break

    extraExtent = i

    return extraExtent
