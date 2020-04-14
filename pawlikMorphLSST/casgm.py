'''Module contains routines to calculate CAS parameters, as well as Gini, M20,
   R20, R80, and Petrosian radii.
'''

from typing import List, Tuple

import numpy as np
from photutils import EllipticalAperture, EllipticalAnnulus, CircularAperture, CircularAnnulus
from photutils.morphology import gini as giniPhotutils
import scipy.ndimage as ndi
from scipy.optimize import brentq
from skimage.measure import moments_central, moments

__all__ = ["gini", "m20", "concentration", "smoothness", "calcPetrosianRadius",
           "calcR20_R80"]


def _getCircularFraction(image: np.ndarray, centroid: List[float],
                         radius: float, fraction: float) -> float:

    apeture_total = CircularAperture(centroid, radius)
    totalSum = apeture_total.do_photometry(image, method="exact")[0][0]

    deltaRadius = (radius / 100.) * 2.
    radiusCurrent = radius - deltaRadius
    radiusMin = 0
    radiusMax = 0

    while True:
        apCur = CircularAperture(centroid, radiusCurrent,)
        currentSum = apCur.do_photometry(image, method="exact")[0][0]
        currentFraction = currentSum / totalSum
        if currentFraction <= fraction:
            radiusMin = radiusCurrent
            radiusMax = radiusCurrent + deltaRadius
            break
        radiusCurrent -= deltaRadius

    r = brentq(_fractionTotalFLuxCircle, radiusMin, radiusMax, args=(image, centroid,
               totalSum, fraction))

    return r


def _fractionTotalFLuxCircle(radius: float, image: np.ndarray,
                             centroid: List[float], totalSum: float,
                             fraction: float) -> float:

    apetureCur = CircularAperture(centroid, radius)
    currentSum = apetureCur.do_photometry(image, method="exact")[0][0]

    return (currentSum/totalSum) - fraction


def calcR20_R80(image: np.ndarray, centroid: List[float],
                radius: float) -> Tuple[float, float]:

    r20 = _getCircularFraction(image, centroid, radius, 0.2)
    r80 = _getCircularFraction(image, centroid, radius, 0.8)

    return r20, r80


def _fractionTotalFLuxEllipse(a: float, image: np.ndarray,
                              centroid: List[float], ellip: float,
                              theta: float) -> float:

    b = a / ellip
    a_in = a - 0.5
    a_out = a + 0.5

    b_out = a_out / ellip

    ellip_annulus = EllipticalAnnulus(centroid, a_in, a_out, b_out, theta)
    ellip_aperture = EllipticalAperture(centroid, a, b, theta)

    ellip_annulus_mean_flux = np.abs(ellip_annulus.do_photometry(image, method="exact")[0][0]) / ellip_annulus.area
    ellip_aperture_mean_flux = np.abs(ellip_aperture.do_photometry(image, method="exact")[0][0]) / ellip_aperture.area

    ratio = ellip_annulus_mean_flux / ellip_aperture_mean_flux

    return ratio - 0.2


def calcPetrosianRadius(image: np.ndarray, centroid: List[float],
                        fwhms: List[float],
                        theta: float) -> Tuple[float, float]:

    ellip = max(fwhms)/min(fwhms)

    npoints = 100
    a_inner = 1.
    a_outer = np.sqrt(image.shape[0]**2 + image.shape[1]**2)

    da = a_outer - a_inner / float(npoints)
    a_min, a_max = None, None
    a = a_inner

    while True:
        curval = _fractionTotalFLuxEllipse(a, image, centroid, ellip, theta)
        if curval == 0:
            return a
        elif curval > 0:
            a_min = a
        elif curval < 0:
            a_max = a
            break
        a += da

    rp = brentq(_fractionTotalFLuxEllipse, a_min, a_max,
                args=(image, centroid, ellip, theta))

    a_in = rp - 0.5
    a_out = rp + 0.5
    b_out = a_out / ellip

    ellip_annulus = EllipticalAnnulus(centroid, a_in, a_out, b_out, theta)
    meanflux = np.abs(ellip_annulus.do_photometry(image, method="exact")[0][0]) / ellip_annulus.area

    return rp, meanflux


def concentration(r20: float, r80: float) -> float:
    '''Function calculates the concentration index from the growth curve radii
       R20 and R80.

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
    ''' Function calculates the Gini index of a Galaxy.

    Parameters
    ----------

    image : image, 2d np.ndarray
        Image from which the Gini index shall be calculated

    flux : float
        The elliptical Petrosian flux, at which is defined as the cutoff for
        calculating the Gini index.


    Returns
    -------

    G : float
        The Gini index.

    '''

    # Only calculate the Gini index on pixels that belong to the galaxy where
    # the flux is greater than the elliptical Petrosian radius.
    img = image[mask > 0]
    G = giniPhotutils(img)

    return G


def m20(image: np.ndarray, mask: np.ndarray) -> float:
    '''Calculate the M20 statistic.

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
    Mc = moments_central(img, center=centroid, order=2)
    secondMomentTotal = Mc[0, 2] + Mc[2, 0]

    # sort pixels, then take top 20% of brightest pixels
    sortedPixels = np.sort(img.ravel())
    fluxFraction = np.cumsum(sortedPixels) / np.sum(sortedPixels)
    sortedPixels20 = sortedPixels[fluxFraction > 0.8]
    thresh = sortedPixels20[0]

    # Select pixels from the image that are the top 20% brightest
    # then compute M20
    img20 = np.where(img >= thresh, img, 0.0)
    Mc20 = moments_central(img20, center=centroid, order=2)
    secondMoment20 = Mc20[0, 2] + Mc20[2, 0]

    m20 = np.log10(secondMoment20 / secondMomentTotal)

    return m20


def smoothness(image: np.ndarray, mask: np.ndarray, centroid: List[float],
               Rmax: float, r20: float, sky: float) -> float:
    '''Calculate the smoothness of clumpiness of the galaxy of interest.

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

    bgr = -99.

    # move box of size boxSize over image till we find an area that is
    # only background.
    # start at boxSize=32 then half if not found
    # check box does not overlap galaxy object
    # slide right/up 2 pixels at a time if not found
    while True:
        skyPixels = imageCopy[posY:posY+boxSize, posX:posX+boxSize]
        boxMean = np.sum(skyPixels) / (boxSize**2)

        if np.abs(sky) > np.abs(boxMean) and not np.any(skyPixels) == -99.:
            bgr = boxMean
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
                    bgr = -99.
                    break

    # Calculate smoothness
    bkg = image[posY:posY+boxSize, posX:posX+boxSize]
    bkgSmooth = ndi.uniform_filter(bkg, size=int(boxcarSize))
    bkgDiff = bkg - bkgSmooth
    bkgDiff[bkgDiff < 0.] = 0.

    return np.sum(bkgDiff) / float(bkg.size)
