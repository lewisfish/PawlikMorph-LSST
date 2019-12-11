from typing import List, Tuple

import numpy as np
from astropy.modeling import models, fitting

__all__ = ["fitSersic"]


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

    ellip = (max(fwhms) - min(fwhms)) / max(fwhms)

    # TODO better guess of intial values?
    sersic_init = models.Sersic2D(amplitude=np.mean(imageCopy),
                                  r_eff=max(fwhms), n=1.5,
                                  x_0=centroid[0], y_0=centroid[1],
                                  ellip=ellip, theta=theta)

    ny, nx = imageCopy.shape
    y, x = np.mgrid[0:ny, 0:nx]

    Parameters = fit_p(sersic_init, x, y, imageCopy, maxiter=500, acc=1e-5)

    return Parameters
