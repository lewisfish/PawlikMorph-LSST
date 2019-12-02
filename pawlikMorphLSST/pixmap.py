import sys

import numpy as np
from scipy import ndimage
from skimage import transform

__all__ = ["pixelmap"]


def pixelmap(img: np.ndarray, thres: float, filtsize: int) -> np.ndarray:
    # from matplotlib import animation
    ''' Calculates an object binary mask using a mean filter and 8 connected
        pixels and a given threshold.

    Parameters
    ----------

    img : np.ndarray
        Image from which the binary mask is calculated.
    thres : float
        Threshold for calculating 8 connectedness
    filtsize : int
        Size of the mean filter. Must be odd

    Returns
    -------
    objmask : np.ndrray
        Calculated binary object mask

    '''

    if filtsize % 2 == 0:
        print("ERROR! Filter can not be of even size.")
        sys.exit()

    imgsize = img.shape[0]

    # mean filter
    img = ndimage.uniform_filter(img, size=filtsize, mode="reflect")
    # resize image to match PawlikMorph
    # TODO leave this as an option?
    img = transform.resize(img, (int(imgsize / filtsize), int(imgsize / filtsize)),
                           anti_aliasing=False, preserve_range=True)

    npix = img.shape[0]
    cenpix = np.array([int(npix/2), int(npix/2)])
    if img[cenpix[0], cenpix[1]] < thres:
        print("ERROR! Central pixel too faint")
        # sys.exit()

    # output binary image array
    objmask = np.zeros_like(img)
    # set central pixel as this is always included
    objmask[cenpix[0], cenpix[1]] = 1

    # start list with central pixel
    pixels = [cenpix]
    pixelsleft = True
    # order in which to view 8 connected pixels
    xvec = [1, 0, -1, -1, 0, 0, 1, 1]
    yvec = [0, -1, 0, 0, 1, 1, 0, 0]

    # loop over pixels in pixel array
    # check 8 connected pixels and add to array if above threshold
    # remove pixel from array when its been operated on
    while pixelsleft:
        x, y = pixels.pop(0)
        xcur = x
        ycur = y
        for i in range(0, 8):
            xcur += xvec[i]
            ycur += yvec[i]
            if xcur >= npix or ycur >= npix or xcur < 0 or ycur < 0:
                continue
            if img[xcur, ycur] > thres and objmask[xcur, ycur] == 0:
                objmask[xcur, ycur] = 1
                pixels.append([xcur, ycur])
        if len(pixels) == 0:
            pixelsleft = False
            break

    # resize binary image to original size
    objmask = transform.resize(objmask, (imgsize, imgsize), order=0, mode="edge")
    return objmask
