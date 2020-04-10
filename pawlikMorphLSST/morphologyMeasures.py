from typing import List

import numpy as np
from photutils.morphology import gini as giniPhotutils
from photutils import EllipticalAperture, EllipticalAnnulus, CircularAperture, CircularAnnulus
from scipy.optimize import brentq
import scipy.ndimage as ndi

__all__ = ["gini", "m20", "concentration", "smoothness", "calcPetrosianRadius"]


def _getCircularFraction(image, centroid, radius, fraction):

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


def _fractionTotalFLuxCircle(radius, image, centroid, totalSum, fraction):
    apetureCur = CircularAperture(centroid, radius)
    currentSum = apetureCur.do_photometry(image, method="exact")[0][0]

    return (currentSum/totalSum) - fraction


def calcR20_R80(image, centroid, radius):

    r20 = _getCircularFraction(image, centroid, radius, 0.2)
    r80 = _getCircularFraction(image, centroid, radius, 0.8)

    return r20, r80


def _fractionTotalFLuxEllipse(a, image, centroid, ellip, theta):

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


def calcPetrosianRadius(image, centroid, fwhms, theta):

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

    rp = brentq(_fractionTotalFLuxEllipse, a_min, a_max, args=(image, centroid, ellip, theta))

    a_in = rp - 0.5
    a_out = rp + 0.5
    b_out = a_out / ellip

    ellip_annulus = EllipticalAnnulus(centroid, a_in, a_out, b_out, theta)
    meanflux = np.abs(ellip_annulus.do_photometry(image, method="exact")[0][0]) / ellip_annulus.area

    return rp, meanflux


def concentration(r20, r80):
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


def gini(image, flux):
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
    img = image[np.abs(image) >= flux]
    G = giniPhotutils(img)

    return G


def m20(image, pixelmap):
    pass


def smoothness(image, mask, centroid, Pr):

    pr = 33.21485364762297

    r_in = 0.25 * pr
    r_out = 1.5 * pr

    imageApeture = CircularAnnulus(centroid, r_in, r_out)

    boxcarSize = int(pr * .25)
    imageSmooth = ndi.uniform_filter(image, size=boxcarSize)

    imageDiff = image - imageSmooth
    imageDiff[imageDiff < 0.] = 0.

    imageFlux = imageApeture.do_photometry(image, method="exact")[0][0]
    diffFlux = imageApeture.do_photometry(imageDiff, method="exact")[0][0]

    Bs = _skySmoothness(image, mask, pr)

    top = diffFlux - imageApeture.area*Bs
    bottom = imageFlux

    print(f"Bs:{Bs}")
    return top / bottom


def _skySmoothness(image, mask, Pr):

    maskCopy = mask.copy()
    bkg = image * ~np.array(maskCopy, dtype=np.bool)

    boxcarSize = int(Pr * .25)
    bkgSmooth = ndi.uniform_filter(bkg, size=boxcarSize)

    bkgDiff = bkg - bkgSmooth
    bkgDiff[bkgDiff < 0] = 0.0

    return np.sum(bkgDiff) / float(bkg.size)
