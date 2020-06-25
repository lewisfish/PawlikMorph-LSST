from typing import List

import numba as nb
import numpy as np
from astropy.io import fits

__all__ = ["makeaperpixmaps", "distarr", "subdistarr", "apercentre", "aperpixmap"]


def makeaperpixmaps(npix: int, folderpath=None) -> None:
    '''Writes the aperture binary masks out after calculation.

    Parameters
    ----------
    npix : int
        Width of aperture image.

    folderpath : Pathlib object
        Path to the folder where the aperture masks should be saved.

    Returns
    -------

    None
    '''

    cenpix = npix / 2. + 1.
    r_aper = np.arange(cenpix) + 1.

    numaper = len(r_aper)

    for i in range(0, numaper):

        aperturepixelmap = aperpixmap(npix, r_aper[i], 9, .1)
        if folderpath:
            fileout = folderpath / f"aperture{i}.fits"
        else:
            fileout = f"aperture{i}.fits"
        fits.writeto(fileout, aperturepixelmap, overwrite=True)


@nb.njit
def aperpixmap(npix: int, rad: float,  nsubpix: int, frac: float) -> np.ndarray:
    '''Calculate aperture binary mask.

       Calculates the aperture binary mask through pixel sampling knowing
       the aperture radius and number of subpixels.

       Near direct translation of IDL code.

    Parameters
    ----------

    npix : int
        Width of aperture image.

    rad : float
        Radius of the aperture.

    nsubpix : int
        Number of subpixels

    frac : float
        Fraction of something... Maybe due to Petrosian magnitude?

    Returns
    -------

    np.ndarry
        Numpy array that stores the mask.
    '''

    npix = int(npix)

    cenpix = np.array([int(npix/2) + 1, int(npix/2) + 1])

    mask = np.zeros((npix, npix))
    submasksize = (npix*nsubpix, npix*nsubpix)

    # create subdistance array
    subdist = subdistarr(npix, nsubpix, cenpix)

    xcoord = 0
    ycoord = 0

    # subpixel coordinates
    x_min = 0
    y_min = 0
    x_max = nsubpix - 1
    y_max = nsubpix - 1

    inds = np.arange(0, npix*npix)
    subpixels = np.zeros((npix*npix, nsubpix, nsubpix))

    i = 0
    for i in range(0, (npix*npix)):

        subpixels[i, :, :] = subdist[x_min:x_max+1, y_min:y_max+1]

        xcoord += 1

        x_min += nsubpix
        x_max += nsubpix

        if y_max > submasksize[1]:
            break
        if x_max > submasksize[0]:
            xcoord = 0
            ycoord += 1

            x_min = 0
            x_max = nsubpix - 1
            y_min += nsubpix
            y_max += nsubpix

    for i in range(0, (npix*npix)):

        # TODO. This is really inefficient...
        apersubpix = (subpixels[i, :, :].flatten()[::-1] <= rad).nonzero()[0]
        apersubpix_size = apersubpix.shape

        fraction = float(apersubpix_size[0]) / float(nsubpix**2)
        if fraction >= frac:
            x = int(inds[i] % npix)
            y = int(inds[i] // npix)
            mask[x, y] = 1

    return mask


@nb.njit(nb.float64[:, :](nb.int64, nb.int64, nb.int64[:]))
def distarr(npixx: int, npixy: int, cenpix: np.ndarray) -> np.ndarray:
    '''Creates an array of distances from given centre pixel.

    Near direct translation of IDL code.


    Parameters
    ----------
    npixx : int
        Number of x pixels in the aperture mask.
    npixy : int
        Number of y pixels in the aperture mask.
    cenpix : np.ndarray
        Location of central pixels.

    Returns
    -------
    np.ndarray
        array of distances.

    '''

    y1 = np.arange(npixy) - cenpix[1]
    y2 = np.ones_like(y1)

    x1 = np.arange(npixx) - cenpix[0]
    x2 = np.ones_like(x1)

    pixy = np.transpose(np.outer(y1, x2))
    pixx = np.transpose(np.outer(y2, x1))

    dist = np.sqrt(pixx**2 + pixy**2)

    return dist


@nb.njit(nb.float64[:, :](nb.int64, nb.int64, nb.int64[:]))
def subdistarr(npix: int, nsubpix: int, cenpix: List[int]) -> np.ndarray:
    '''Writes the aperture binary masks out after calculation.

    Near direct translation of IDL code.

    Parameters
    ----------
    npix : int
        Number of pixels in the aperture mask.
    nsubpix : int
        Number of subpixels.
    cenpix : List[int]
        Location of central pixels.

    Returns
    -------

    np.ndarray
        Array of sub-distances.
    '''

    xneg = np.arange(int(npix/2)*nsubpix) / nsubpix - cenpix[0]
    xpos = -xneg[::-1]
    zeros = np.zeros(nsubpix)
    x1 = xneg
    x1 = np.concatenate((x1, zeros))
    x1 = np.concatenate((x1, xpos))
    x2 = np.ones_like(x1)

    yneg = np.arange(int(npix/2)*nsubpix) / nsubpix - cenpix[1]
    ypos = -yneg[::-1]
    y1 = xneg
    y1 = np.concatenate((y1, zeros))
    y1 = np.concatenate((y1, ypos))

    subpix_x = np.outer(x1, x2)
    subpix_y = np.outer(x2, y1)

    subdist = np.sqrt(subpix_x**2 + subpix_y**2)

    return subdist


def apercentre(apermask: np.ndarray, pix: np.ndarray) -> np.ndarray:
    """Function that centers a precomputed aperture mask on a given pixel.

    Parameters
    ----------

    apermask : np.ndarray
        Aperture mask that is to be centred.

    pix: List[int]
        Central pixel indicies.

    Returns
    -------

    mask : np.ndarray
        Returns aperture mask centered on central pixel, pix.
    """

    npix = apermask.shape[0]
    cenpix = np.array([int(npix/2)+1, int(npix/2)+1])
    delta = pix - cenpix
    mask = apermask
    # moves each axis by delta[n], i.e translate image left/right/up/down by
    # desired amount
    mask = np.roll(mask, delta[0], axis=0)
    mask = np.roll(mask, delta[1], axis=1)

    return mask
