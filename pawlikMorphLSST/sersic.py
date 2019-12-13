from typing import List, Tuple
import sys
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import photutils
import numpy as np
from astropy.modeling import models, fitting

__all__ = ["fitSersic"]


def fractionTotalFLuxEllipse(a, image, b, theta, centre, totalsum):

    apCur = photutils.EllipticalAperture(centre, a, b, theta)
    curSum = apCur.do_photometry(image, method="exact")[0][0]

    return (curSum/totalsum) - 0.5


def fitSersic(image: np.ndarray, centroid: List[float], fwhms: List[float],
              theta: float, starMask: np.ndarray):
    '''Function that fits a 2D sersic function to an image of a Galaxy.

    Parameters
    ----------

    image : np.ndarray
        image to which a 2D Sersic function will be fit
    centroid : List[float]
        Centre of object of interest
    fwhms : List[float]
        Full width half maximums of object
    theta : float
        rotation of object anticlockwise from positive x axis
    starMask : np.ndarray
        Mask contains star locations.

    Returns
    -------

    Parameters : astropy.modeling.Model object

        Collection of best fit parameters for the 2D sersic function

    '''

    imageCopy = image * starMask
    fit_p = fitting.LevMarLSQFitter()

    # amplitude => Surface brightness at r_eff
    # r_eff => Effective (half-light) radius
    # n => sersic index, guess 1.?

    b = 2. * min(fwhms)
    a = 2. * max(fwhms)
    ellip = 1. - (b/a)

    ap_total = photutils.EllipticalAperture(centroid, a, b, theta)
    totalSum = ap_total.do_photometry(image, method="exact")[0][0]

    # get bracketing values for root finder to find r_eff
    deltaA = (a / 100.) * 3.
    aCurrent = a - deltaA
    aMin = 0
    aMax = 0
    while True:
        apCur = photutils.EllipticalAperture(centroid, aCurrent, b, theta)
        currentSum = apCur.do_photometry(image, method="exact")[0][0]
        currentFraction = currentSum / totalSum
        if currentFraction <= .5:
            aMin = aCurrent
            aMax = aCurrent + deltaA
            break
        aCurrent -= deltaA

    # get root
    r_eff = brentq(fractionTotalFLuxEllipse, aMin, aMax, args=(image, b, theta,
                   centroid, totalSum))

    # calculate amplitude at r_eff
    a_in = r_eff - 0.5
    a_out = r_eff + 0.5
    b_out = a_out - (1. * ellip)
    ellip_annulus = photutils.EllipticalAnnulus(centroid, a_in, a_out, b_out, theta)
    totalR_effFlux = ellip_annulus.do_photometry(image, method="exact")[0][0]
    meanR_effFlux = totalR_effFlux / ellip_annulus.area

    sersic_init = models.Sersic2D(amplitude=meanR_effFlux,
                                  r_eff=r_eff, n=2.0,
                                  x_0=centroid[0], y_0=centroid[1],
                                  ellip=ellip, theta=theta)

    ny, nx = imageCopy.shape
    y, x = np.mgrid[0:ny, 0:nx]

    Parameters = fit_p(sersic_init, x, y, imageCopy, maxiter=1000, acc=1e-8)

    return Parameters
