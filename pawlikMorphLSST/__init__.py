''' A package to analyse images of galaxies in to determine various
    morphological properties.
'''

import numpy as _np


__all__ = ["gaussfitter", "pixmap", "apertures", "asymmetry",
           "imageutils", "objectMasker"]


def prepareimage(img: _np.ndarray):

    from .imageutils import skybgr, cleanimg
    from .pixmap import pixelmap
    from astropy.io import fits

    '''Helper function to prepare images

    Parameters
    ----------
        img : np.ndarray
            Image to be prepared.

    Returns
    -------

    img : np.ndarray
        Image which has been bgr subtracted and 'cleaned'.
    mask : np.ndarray
        Binary image of object.

    '''

    if img.shape[0] != img.shape[1]:
        print("ERROR! image not square")
        return

    img = img.byteswap().newbyteorder()

    sky, sky_err, flag = skybgr(img, img.shape[0])
    mask = pixelmap(img, sky + sky_err, 3)
    img -= sky

    img = cleanimg(img, mask)
    fits.writeto("clean.fits", img)

    return img, mask


def calculateMorphology(img: _np.ndarray, mask: _np.ndarray):
    '''Helper function to calculate all Asymmetry parameters.

    Parameters
    ----------

    img : np.ndarray
        Cleaned, bgr subtracted image.
    mask : np.ndarray
        Binary object mask

    Returns
    -------

    A : List[float]
        Asymmetry parameter and its background value
    As : List[float]
        Shape asymmetry parameter
    As90 : List[float]
        Shape 90 degrees Asymmetry parameter


    '''
    from .apertures import distarr, aperpixmap
    from .asymmetry import calcA, minapix

    imgsize = img.shape[0]
    objectpix = _np.nonzero(mask == 1)
    cenpix = _np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

    distarray = distarr(imgsize, imgsize, cenpix)
    objectdist = distarray[objectpix]
    r_max = _np.max(objectdist)
    aperturepixmap = aperpixmap(imgsize, r_max, 9, 0.1)

    apix = minapix(img, mask, aperturepixmap)
    angle = 180.

    A = calcA(img, mask, aperturepixmap, apix, angle, noisecorrect=True)
    As = calcA(mask, mask, aperturepixmap, apix, angle)
    As90 = calcA(mask, mask, aperturepixmap, apix, 90.)

    return A, As, As90
