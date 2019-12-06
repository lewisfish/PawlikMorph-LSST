from typing import List, Tuple

import numpy as np
from astropy.modeling import models, fitting

__all__ = ["fitSersic"]


def fitSersic(image: np.ndarray, centroid: List[float], fwhms: List[float], theta: float):
    '''Function that fits a 2D sersic function to an image.

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

    Returns
    -------

    Parameters : astropy.modeling.Model object

        Collection of best fir parameters for the 2D sersic function

    '''

    fit_p = fitting.LevMarLSQFitter()

    # amplitude => Surface brightness at r_eff
    # r_eff => Effective (half-light) radius
    # n => sersic index, guess 1.?

    ellip = (max(fwhms) - min(fwhms)) / max(fwhms)

    sersic_init = models.Sersic2D(amplitude=np.mean(image), r_eff=max(fwhms), n=1.5, x_0=centroid[0], y_0=centroid[1],
                                  ellip=ellip, theta=theta)
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]

    Parameters = fit_p(sersic_init, x, y, image, maxiter=500, acc=1e-5)

    return Parameters
