import numpy as np
from photutils.morphology import gini as giniPhotutils

__all__ = ["gini", "m20", "concentration", "clumpiness"]


def _getCircularFraction(image, centroid, radius, fraction):

    apeture_total = CircularAperture(centroid, radius)
    totalSum = ap_total.do_photometry(image, method="exact")[0][0]

    deltaRadius = (radius / 100.) * 2.
    radiusCurrent = radius - deltaRadius
    radiusMin = 0
    radiusMax = 0
    while True:
        apCur = EllipticalAperture(centroid, radiusCurrent,)
        currentSum = apCur.do_photometry(image, method="exact")[0][0]
        currentFraction = currentSum / totalSum
        if currentFraction <= fraction:
            radiusMin = radiusCurrent
            radiusMax = radiusCurrent + deltaRadius
            break
        radiusCurrent -= deltaRadius

    r = brentq(_fractionTotalFLuxCircle, radiusMin, radiusMax, args=(image, centroid,
               centroid, radius, totalSum, fraction))

    return r


def _fractionTotalFLuxCircle(image, centroid, radius, totalSum, fraction):
    apetureCur = CircularAperture(centroid, radius)
    currentSum = apCur.do_photometry(image, method="exact")[0][0]

    return (curSum/totalsum) - fraction


def calcR20_R80(image, centroid, radius):

    r20 = _getCircularFraction(image, centroid, radius, 0.2)
    r80 = _getCircularFraction(image, centroid, radius, 0.8)

    return r20, r80


def concentration(r20, r80):

    C = 5.0 * np.log10(r80 / r20)

    return C


def gini(image, pixelmap):

    imgMasked = image * pixelmap
    G = giniPhotutils(imgMasked)

    return G


def m20(image, pixelmap):
    pass


def clumpiness():
    pass
