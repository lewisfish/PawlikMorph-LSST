from typing import List

import numpy as np
from scipy import ndimage
from skimage import transform

from .apertures import distarr

__all__ = ["pixelmap", "calcMaskedFraction", "calcRmax", "checkPixelmapEdges"]


class _Error(Exception):
    '''Base class for custom exceptions'''
    pass


class _ImageError(_Error):
    '''Class of exception where the image contains no object or
       the object is not centred'''

    def __init__(self, value):
        print(value)
        raise AttributeError


def calcRmax(image: np.ndarray, segmap: np.ndarray) -> float:
    '''Function to calculate the maximum extent of a binary pixel map

    Parameters
    ----------

    image : float, 2d np.ndarray
        Image of galaxy.

    segmap : np.ndarray
        Binary pixel segmap

    Return
    ------

    rmax : float
        the maximum extent of the pixelmap

    '''

    MaskedImage = image*segmap

    objectpix = np.nonzero(segmap == 1)
    cenpix = np.array(np.unravel_index(MaskedImage.argmax(), image.shape))

    distarray = distarr(image.shape[0], image.shape[1], cenpix)
    objectdist = distarray[objectpix]
    rmax = np.max(objectdist)

    return rmax


def calcMaskedFraction(oldMask: np.ndarray, starMask: np.ndarray,
                       cenpix: List[float]) -> float:
    '''Calculates the fraction of pixels that are masked.

    Parameters
    ----------

    oldMask : np.ndarray
        Mask containing object of interest.

    starMask : np.ndarray
        Mask containing location of star

    cenpix : List[float]
        Centre of asymmetry in pixels.

    Returns
    -------

    Fraction of object pixels masked by stars : float

    '''

    starMaskCopy = starMask * transform.rotate(starMask, 180.,
                                               center=(cenpix[0], cenpix[1]),
                                               preserve_range=True, cval=1.)

    maskedFraction = np.sum(oldMask * starMaskCopy)
    objectFraction = np.sum(oldMask)

    return 1. - (maskedFraction / objectFraction)


def checkPixelmapEdges(segmap: np.ndarray) -> bool:
    '''Flag image if galaxy pixels are on edge of image.

    Parameters
    -------

    segmap : np.ndarray
        pixelmap to check

    Returns
    ----------

    flag : bool
        If true then the pixelmap hits the edge of the image

    '''

    flag = True

    summ = np.sum(segmap[0, :])
    if summ > 0:
        return flag

    summ = np.sum(segmap[segmap.shape[0]-1, :])
    if summ > 0:
        return flag

    summ = np.sum(segmap[:, 0])
    if summ > 0:
        return flag

    summ = np.sum(segmap[:, segmap.shape[0]-1])
    if summ > 0:
        return flag

    flag = False
    return flag


def pixelmap(image: np.ndarray, threshold: float, filterSize: int,
             starMask=None) -> np.ndarray:
    ''' Calculates an object binary mask.

        This is acheived using a mean filter, 8 connected
        pixels, and a given threshold.

    Parameters
    ----------

    image : np.ndarray
        Image from which the binary mask is calculated.
    threshold : float
        Threshold for calculating 8 connectedness
    filterSize : int
        Size of the mean filter. Must be odd
    starMask : optional, None or np.ndarray
        Mask that mask out nuisance stars

    Returns
    -------
    objectMask : np.ndrray
        Calculated binary object mask

    '''

    if starMask is None:
        starMask = np.full(image.shape, True)
    imageTmp = image * starMask

    if filterSize % 2 == 0:
        print("ERROR! Filter can not be of an even size.")
        sys.exit()

    imgsize = imageTmp.shape[0]

    # mean filter
    imageTmp = ndimage.uniform_filter(imageTmp, size=filterSize,
                                      mode="reflect")
    # resize image to match PawlikMorph
    # TODO leave this as an option?
    imageTmp = transform.resize(imageTmp, (int(imgsize / filterSize),
                                int(imgsize / filterSize)),
                                anti_aliasing=False, preserve_range=True)

    npix = imageTmp.shape[0]
    cenpix = np.array([int(npix/2), int(npix/2)])

    if imageTmp[cenpix[0], cenpix[1]] < threshold:
        raise _ImageError("ERROR! Central pixel too faint")

    # output binary image array
    objectMask = np.zeros_like(imageTmp)
    # set central pixel as this is always included
    objectMask[cenpix[0], cenpix[1]] = 1

    # start list with central pixel
    pixels = [cenpix]
    # order in which to view 8 connected pixels
    xvec = [1, 0, -1, -1, 0, 0, 1, 1]
    yvec = [0, -1, 0, 0, 1, 1, 0, 0]

    # loop over pixels in pixel array
    # check 8 connected pixels and add to array if above threshold
    # remove pixel from array when its been operated on
    while True:
        x, y = pixels.pop(0)
        xcur = x
        ycur = y
        for i in range(0, 8):
            xcur += xvec[i]
            ycur += yvec[i]
            if xcur >= npix or ycur >= npix or xcur < 0 or ycur < 0:
                continue
            if imageTmp[xcur, ycur] > threshold and objectMask[xcur, ycur] == 0:
                objectMask[xcur, ycur] = 1
                pixels.append([xcur, ycur])
        if len(pixels) == 0:
            break

    # resize binary image to original size
    objectMask = transform.resize(objectMask, (imgsize, imgsize), order=0,
                                  mode="edge")
    return objectMask
