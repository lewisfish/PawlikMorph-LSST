from typing import List, Tuple

import numpy as np
from astropy.modeling import models, fitting

__all__ = ["fitSersic"]


def fitSersic(image: np.ndarray, centroid: List[float], fwhms: List[float], theta: float):
    '''

    Parameters
    ----------

    image : np.ndarray

    centroid : List[float]

    fwhms : List[float]

    theta : float

    Returns
    -------

    '''

    fit_p = fitting.LevMarLSQFitter()

    # amplitude => Surface brightness at r_eff
    # r_eff => Effective (half-light) radius
    # n => sersic index, guess 1.?
    # theta, rotation angle in radians, counterclockwise from +ive x axis

    ellip = (max(fwhms) - min(fwhms)) / max(fwhms)

    sersic_init = models.Sersic2D(amplitude=np.mean(image), r_eff=max(fwhms), n=1.5, x_0=centroid[0], y_0=centroid[1],
                                  ellip=ellip, theta=theta)
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]

    p = fit_p(sersic_init, x, y, image, maxiter=10000)

    return p
