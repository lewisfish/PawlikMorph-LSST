import numpy as np
from photutils.morphology import gini as giniPhotutils
from photutils import EllipticalAperture, EllipticalAnnulus, CircularAperture
from typing import List
from scipy.optimize import brentq

__all__ = ["gini", "m20", "concentration", "clumpiness", "calcPetrosianRadius"]


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

    C = 5.0 * np.log10(r80 / r20)

    return C


def gini(image):

    # imgMasked = np.abs(image * pixelmap)
    G = giniPhotutils(image > 0)

    return G


def m20(image, pixelmap):
    pass


def clumpiness():
    pass
