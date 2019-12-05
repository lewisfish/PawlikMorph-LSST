from typing import List, Tuple

from astropy.io import fits

import numpy as np
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
import scipy.ndimage as ndimage

from .asymmetry import minapix
from .imageutils import _calcSkybgr, maskstarsSEG
from .pixmap import pixelmap
from .apertures import distarr, aperpixmap
import scipy

__all__ = ["fitSersic"]


def fitSersic(image, centroid, fwhms, theta):

    log_stretch = LogStretch(a=10000.0)

    fit_p = fitting.LevMarLSQFitter()

    # amplitude => Surface brightness at r_eff
    # r_eff => Effective (half-light) radius
    # n => sersic index, guess 1.?
    # theta, rotation angle in radians, counterclockwise from +ive x axis
    # centroid, fwhms, theta = getshite(image, image.shape)

    ellip = (max(fwhms) - min(fwhms)) / max(fwhms)

    sersic_init = models.Sersic2D(amplitude=np.mean(image), r_eff=max(fwhms), n=1.5, x_0=centroid[0], y_0=centroid[1],
                                  ellip=ellip, theta=theta)
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]

    p = fit_p(sersic_init, x, y, image, maxiter=10000)

    # modelimage = models.Sersic2D.evaluate(x, y, p.amplitude, p.r_eff, p.n, p.x_0, p.y_0, p.ellip, p.theta)
    return p
# def getshite(img, imgsize):

#         sky, sky_err, fwhms, theta = _calcSkybgr(img, imgsize[0])

#         mask = pixelmap(img, sky + sky_err, 3)
#         plt.imshow(mask)
#         plt.show()
#         img -= sky

#         objectpix = np.nonzero(mask == 1)
#         cenpix = np.array([int(imgsize[0]/2) + 1, int(imgsize[0]/2) + 1])

#         distarray = distarr(imgsize[0], imgsize[0], cenpix)
#         objectdist = distarray[objectpix]
#         rmax = np.max(objectdist)
#         aperturepixmap = aperpixmap(imgsize[0], rmax, 9, 0.1)

#         starMask = np.ones_like(img)
#         apix = minapix(img, mask, aperturepixmap, starMask)
#         return apix, fwhms, theta
