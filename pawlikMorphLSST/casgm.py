'''Module contains routines to calculate CAS parameters, as well as Gini, M20,
   R20, R80.
'''

from typing import List, Tuple

import numpy as np
from photutils import CircularAperture, CircularAnnulus
from photutils.morphology import gini as giniPhotutils
import scipy.ndimage as ndi
from scipy.optimize import brentq
from skimage.measure import moments_central, moments

from . apertures import aperpixmap
from . asymmetry import minapix
from . pixmap import calcRmax

__all__ = ["gini", "m20", "concentration", "smoothness", "calcR20_R80", "calculateCSGM"]


def calculateCSGM(image: np.ndarray, mask: np.ndarray, skybgr: float) -> Tuple[float]:
    """Helper function that calculates the CSGM parameters

    Parameters
    ----------

    image : np.ndarray, 2D float
        Image of a galxy for which the CSGM parameters are to be calculated

    mask : np.ndarray, 2D uint8
        Image for which only the pixels that belong to the galaxy in "image" are "hot".

    skybgr : float
        The value of the sky background in the given image

    Returns
    -------

    C, S, gini, m20: Tuple[float]
        The CSGM parameters

    """

    Rmax = calcRmax(image, mask)
    aperturepixmap = aperpixmap(image.shape[0], Rmax, 9, 0.1)

    starmask = np.ones_like(image)
    apix = minapix(image, mask, aperturepixmap, starmask)
    r20, r80 = calcR20_R80(image, apix, Rmax)
    C = concentration(r20, r80)
    g = gini(image, mask)
    S = smoothness(image, mask, apix, Rmax, r20, skybgr)
    m = m20(image, mask)

    return C, S, g, m


def _getCircularFraction(image: np.ndarray, centroid: List[float],
                         radius: float, fraction: float) -> float:
    '''Function determines the radius of a circle that encloses a fraction of
       the total light.

    Parameters
    ----------

    image : float, 2d np.ndarray
        Image of galaxy

    centroid : List[float]
        Location of the brightest pixel

    radius : float
        Radius in which to measure galaxy light out to. This is usually Rmax.

    fraction : float
        The fraction of the total galaxy light for the circle to enclose.


    Returns
    -------

    radiusOut : float
        The radius of the circle that encloses fraction of light

    '''

    apeture_total = CircularAperture(centroid, radius)
    totalSum = apeture_total.do_photometry(image, method="exact")[0][0]

    # number of points for grid
    npoints = float(200)

    # grid spacing
    deltaRadius = (radius / npoints) * 2.
    radiusCurrent = radius - deltaRadius
    radiusMin = 0
    radiusMax = 0

    # loop until bracketing values are found for the root at which is
    # our desired radius.
    while True:
        apCur = CircularAperture(centroid, radiusCurrent)
        currentSum = apCur.do_photometry(image, method="exact")[0][0]
        currentFraction = currentSum / totalSum
        if currentFraction <= fraction:
            radiusMin = radiusCurrent
            radiusMax = radiusCurrent + deltaRadius
            break
        radiusCurrent -= deltaRadius

    radiusOut = brentq(_fractionTotalFLuxCircle, radiusMin, radiusMax,
                       args=(image, centroid, totalSum, fraction))

    return radiusOut


def _fractionTotalFLuxCircle(radius: float, image: np.ndarray,
                             centroid: List[float], totalSum: float,
                             fraction: float) -> float:
    r'''Helper function to help find the radius of a circle that encloses a
       fraction of the total light in the galaxy of interest.


    Parameters
    ----------

    radius : float
        Radius in which to measure galaxy light out to. This is usually Rmax.

    image : float, 2d np.ndarray
        Image of galaxy

    centroid : List[float]
        Location of the brightest pixel

    totalSum : float
        The total amount of light in the galaxy.

    fraction : float
        The fraction of the total galaxy light for the circle to enclose.


    Returns
    -------

    root : float
        The fraction of light at the current radius minus the fraction,
        so that a root at 0 can be found.

    '''

    apetureCur = CircularAperture(centroid, radius)
    currentSum = apetureCur.do_photometry(image, method="exact")[0][0]

    root = (currentSum/totalSum) - fraction

    return root


def calcR20_R80(image: np.ndarray, centroid: List[float],
                radius: float) -> Tuple[float, float]:
    r'''Calculation of :math:`r_{20}`, and :math:`r_{80}`


    Parameters
    ----------

    image : float, 2d np.ndarray
        Image of galaxy

    centroid : List[float]
        Location of the brightest pixel

    radius : float
        Radius in which to measure galaxy light out to.

    Returns
    -------

    r20, r80 : Tuple[float, float]
        The radii where 20% and 80% light falls within

    '''

    r20 = _getCircularFraction(image, centroid, radius, 0.2)
    r80 = _getCircularFraction(image, centroid, radius, 0.8)

    return r20, r80


def concentration(r20: float, r80: float) -> float:
    r'''Calculation of the concentration of light in a galaxy.

    .. math::
        C = 5log_{10}(\frac{r_{80}}{r_{20}})

    see Lotz et al. 2004 https://doi.org/10.1086/421849


    Parameters
    ----------

    r20 : float
        Radius at 20% of light
    r80 : float
        Radius at 80% of light

    Returns
    -------
    C : float
        The concentration index

    '''

    C = 5.0 * np.log10(r80 / r20)

    return C


def gini(image: np.ndarray, mask: np.ndarray) -> float:
    r'''Calculation of the Gini index of a Galaxy.

    .. math:: g = \frac{1}{2 \bar{X} n(n-1)} \sum (2i - n - 1) \left|X_i\right|

    Where :math:`\bar{X}` is the mean over all intensities
    n is the total number of pixels
    :math:`X_i` are the pixel intensities in increasing order

    see Lotz et al. 2004 https://doi.org/10.1086/421849

    Parameters
    ----------

    image : float, 2d np.ndarray
        Image from which the Gini index shall be calculated

    mask : int, 2D np.ndarray
        TMask which contains the galaxies pixels

    Returns
    -------

    G : float
        The Gini index.
    '''

    # Only calculate the Gini index on pixels that belong to the galaxy
    img = image[mask > 0]
    G = giniPhotutils(img)

    return G


def m20(image: np.ndarray, mask: np.ndarray) -> float:
    r'''Calculate the M20 statistic.

    .. math:: M_{20} = log_{10} \left(\frac{\sum M_i}  {M_{tot}}\right)
    .. math:: While \sum f_i < 0.2 f_{tot}
    .. math:: M_{tot} = \sum M_i = \sum f_i [(x - x_c)^2 + (y - y_c)^2]

    see Lotz et al. 2004 https://doi.org/10.1086/421849

    Adapted from statmorph: https://github.com/vrodgom/statmorph

    Parameters
    ----------

    image : float, 2d np.ndarray
        Image of galaxy

    mask : float [0. - 1.], 2d np.ndarray
        Mask which contains the pixels belonging to the galaxy of interest.

    Returns
    -------

    m20 : float
        M20 statistic

    '''

    # use the same image as used in Gini calculation.
    img = np.where(mask > 0, image, 0.)

    # Calculate centroid from moments
    M = moments(img, order=1)
    centroid = (M[1, 0] / M[0, 0],  M[0, 1] / M[0, 0])

    # Calculate 2nd order central moment
    Mcentral = moments_central(img, center=centroid, order=2)
    secondMomentTotal = Mcentral[2, 0] + Mcentral[0, 2]

    # sort pixels, then take top 20% of brightest pixels
    sortedPixels = np.sort(img.ravel())
    fluxFraction = np.cumsum(sortedPixels) / np.sum(sortedPixels)
    thresh = sortedPixels[fluxFraction > 0.8][0]

    # Select pixels from the image that are the top 20% brightest
    # then compute M20
    img20 = np.where(img >= thresh, img, 0.0)
    Mcentral20 = moments_central(img20, center=centroid, order=2)
    secondMoment20 = Mcentral20[0, 2] + Mcentral20[2, 0]

    m20 = np.log10(secondMoment20 / secondMomentTotal)

    return m20


def smoothness(image: np.ndarray, mask: np.ndarray, centroid: List[float],
               Rmax: float, r20: float, sky: float) -> float:
    r'''Calculate the smoothness or clumpiness of the galaxy of interest.

    .. math:: S = \frac{\sum \left|I - I_s\right| - B_s} {\sum \left|I\right|}

    Where I is the image
    :math:`I_s` is the smoothed image
    :math:`B_s` is the background smoothness

    see Lotz et al. 2004 https://doi.org/10.1086/421849


    Parameters
    ----------

    image : float, 2d np.ndarray
        Image of galaxy

    mask : float [0. - 1.], 2d np.ndarray
        Mask which contains the pixels belonging to the galaxy of interest.

    centroid : List[float]
        Pixel location of the brightest pixel in galaxy.

    Rmax : float
        Distance from brightest pixel to furthest pixel in galaxy

    r20 : float
        Distance from brightest pixel to radius at which 20% of light of galaxy
        is enclosed.

    sky : float
        Value of the sky background.

    Returns
    -------

    Result: float
        The smoothness or clumpiness parameter, S.

    '''

    r_in = r20
    r_out = Rmax

    # Exclude inner 20% of light. See Concelice 2003
    imageApeture = CircularAnnulus(centroid, r_in, r_out)

    # smooth image
    imageSmooth = ndi.uniform_filter(image, size=int(r20))

    # calculate residual, setting any negative pixels to 0.
    imageDiff = image - imageSmooth
    imageDiff[imageDiff < 0.] = 0.

    # calculate S, accounting for the background smoothness.
    imageFlux = imageApeture.do_photometry(image, method="exact")[0][0]
    diffFlux = imageApeture.do_photometry(imageDiff, method="exact")[0][0]
    backgroundSmooth = _getBackgroundSmoothness(image, mask, sky, r20)
    S = (diffFlux - imageApeture.area*backgroundSmooth) / imageFlux

    return S


def _getBackgroundSmoothness(image: np.ndarray, mask: np.ndarray, sky: float,
                             boxcarSize: float) -> float:
    '''Calculate the background smoothness.

    Parameters
    ----------

    image : float, 2d np.ndarray
        The image.

    mask : float [0-1], 2d np.ndarray
        mask that contains the galaxy objects pixels

    sky : float
        Value of the sky background

    boxcarSize: float
        Size of the box to use in median filter

    Returns
    -------

    The smoothness of the background. Float

    '''

    # Set the galxy pixels to -99 so that we dont include them in background
    # smoothness calculation
    maskCopy = ~np.array(mask, dtype=np.bool)
    imageCopy = image * maskCopy
    imageCopy[imageCopy == 0.] = -99

    boxSize = 32
    posX, posY = 0, 0

    # move box of size boxSize over image till we find an area that is
    # only background.
    # start at boxSize=32 then half if not found
    # check box does not overlap galaxy object
    # slide right/up 2 pixels at a time if not found
    while True:
        skyPixels = imageCopy[posY:posY+boxSize, posX:posX+boxSize]
        boxMean = np.sum(skyPixels) / (boxSize**2)

        if np.abs(sky) > np.abs(boxMean) and not np.any(skyPixels) == -99.:
            break
        else:
            posX += 2
            if posX >= image.shape[1] - boxSize:
                posX = 0
                posY += 2
            if posY >= image.shape[0] - boxSize:
                if boxSize > 8:
                    posY = 0
                    posX = 0
                    boxSize //= 2
                else:
                    break

    # Calculate smoothness
    bkg = image[posY:posY+boxSize, posX:posX+boxSize]
    bkgSmooth = ndi.uniform_filter(bkg, size=int(boxcarSize))
    bkgDiff = bkg - bkgSmooth
    bkgDiff[bkgDiff < 0.] = 0.

    return np.sum(bkgDiff) / float(bkg.size)
