from typing import List

import numpy as np
from pkg_resources import parse_version
from scipy import ndimage
from skimage import transform

from .apertures import apercentre


def minapix(img: np.ndarray, mask: np.ndarray, apermask: np.ndarray) -> List[int]:
    """Funciton that finds the minimum asymmetry central pixel within the
       objects pixels of a given image.
       Selects a range of cnadidate centroids within the brightest region that
       comprimises og 20% of the total flux within object.
       Then measures the asymmetry of the image under rotation around that
       centroid.
       Then picks the centroid that yields the minimum A value.


    Parameters
    ----------

    img : np.ndarray
        Image that the minimum asymmetry pixel is to be found in.
    mask : np.ndarray
        Precomputed mask that describes where the object of interest is in the
        image
    apermask : np.ndarray
        Precomuted aperture mask

    Returns
    -------

    Centroid : List[int]
        The minimum asymmetery pixel position.
    """

    npix = img.shape[0]
    cenpix = np.array([int(npix / 2) + 1, int(npix / 2) + 1])

    imgravel = np.ravel(img)
    tmpmaskravel = np.ravel(img * mask)
    if np.sum(tmpmaskravel) == 0:
        return [0, 0]

    sortedind = np.argsort(tmpmaskravel)[::-1]
    ipix = tmpmaskravel[sortedind]

    itotal = np.sum(ipix)
    ii = 0
    isum = 0
    count = 0

    while ii < npix**2:
        if ipix[ii] > 0:
            count += 1
            isum += ipix[ii]
        if isum >= 0.2*itotal:
            ii = npix*npix
        ii += 1

    ii = 0
    jj = 0
    regionpix = np.zeros(count)
    regionpix_x = np.zeros(count)
    regionpix_y = np.zeros(count)

    isum = 0

    while ii < count:
        if ipix[ii] > 0:
            regionpix[jj] = sortedind[ii]

            if parse_version(np.__version__) >= parse_version("1.16.0"):
                regionpix_2d = np.unravel_index(sortedind[ii], shape=(npix, npix))
            else:
                regionpix_2d = np.unravel_index(sortedind[ii], dims=(npix, npix))

            regionpix_x[jj] = regionpix_2d[1]
            regionpix_y[jj] = regionpix_2d[0]

            isum += ipix[ii]
            jj += 1
        if isum >= 0.2*itotal:
            ii = count
        ii += 1

    regionpix_x = regionpix_x[0:count]
    regionpix_y = regionpix_y[0:count]
    a = np.zeros_like(regionpix)

    for i in range(0, regionpix.shape[0]):

        cenpix_x = int(regionpix_x[i])
        cenpix_y = int(regionpix_y[i])

        imgRot = transform.rotate(img, 180., center=(cenpix_y, cenpix_x), preserve_range=True)
        imgResid = np.abs(img - imgRot)
        imgResidravel = np.ravel(imgResid)

        regionmask = apercentre(apermask, [cenpix_y, cenpix_x])
        regionind = np.nonzero(np.ravel(regionmask) == 1)[0]
        region = imgravel[regionind]
        regionResid = imgResidravel[regionind]

        regionmask *= 0

        a[i] = np.sum(regionResid) / (2. * np.sum(np.abs(region)))

    a_min = np.min(a)
    sub = np.argmin(a)

    centroid_ind = int(regionpix[sub])
    if parse_version(np.__version__) >= parse_version("1.16.0"):
        centroid = np.unravel_index(centroid_ind, shape=(npix, npix))
    else:
        centroid = np.unravel_index(centroid_ind, dims=(npix, npix))

    return centroid


def calcA(img: np.ndarray, pixmap: np.ndarray, apermask: np.ndarray,
          centroid: List[int], angle: float, apermaskcut=None,
          noisecorrect=False) -> List[float]:
    """Function to calculate A, the asymmetery parameter. Near direct
       translation of IDL code.

    Parameters
    ----------

    img : np.ndarray
        Image to be analysed.

    pixmap : np.ndarray
        Mask that covers object of interest.

    apermask : np.ndarray
        Array of the aperature mask image.

    centroid : np.ndarray
        Pixel position of the centroid to be used for rotation.

    angle : float
        Angle to rotate object, in degrees.

    apermaskcut : np.ndarray, optional

    noisecorrect : bool, optional
        Default value False. If true corrects for background noise

    Returns
    -------

    A, Abgr : List(float)
        Returns the asymmetery value and its background value.

    """

    cenpix_x = centroid[0]
    cenpix_y = centroid[1]

    imgRot = transform.rotate(img, angle, center=(cenpix_x, cenpix_y),
                              preserve_range=True)
    imgResid = np.abs(img - imgRot)
    imgravel = np.ravel(img)

    if apermaskcut is None:
        netmask = apermask
    else:
        # TODO: implement
        print("ERROR! Not yet implmented!")
        sys.exit()

    imgResidravel = np.ravel(imgResid)
    regionind = np.nonzero(np.ravel(netmask) == 1)[0]
    region = imgravel[regionind]
    regionResid = imgResidravel[regionind]

    A = np.sum(regionResid) / (2. * np.sum(np.abs(region)))
    Abgr = 0
    if noisecorrect:

        # build "background noise" image using morphological dilation
        # https://en.wikipedia.org/wiki/Dilation_(morphology)
        bgrimg = np.zeros_like(img)
        bgrimg = np.ravel(bgrimg)
        element = np.ones((9, 9))

        mask = ndimage.morphology.binary_dilation(pixmap, structure=element)
        maskind = np.nonzero(np.ravel(mask) == 1)[0]
        bgrind = np.nonzero(np.ravel(mask) != 1)[0]
        bgrpix = imgravel[bgrind]

        if bgrind.shape[0] > (bgrind.shape[0] / 10.):
            if maskind.shape[0] > 1:
                if bgrind.shape[0] >= maskind.shape[0]:
                    maskpix = bgrpix[0:maskind.shape[0]]
                else:
                    pixfrac = maskind.shape[0] / bgrind.shape[0]
                    maskpix = bgrpix
                    if pixfrac == float(round(pixfrac)):
                        for p in range(1, int(pixfrac)):
                            maskpix = np.append(maskpix, bgrpix)
                    else:
                        for p in range(1, int(pixfrac)):
                            maskpix = np.append(maskpix, bgrpix)
                        diff = maskind.shape[0] - maskpix.shape[0]
                        maskpix = np.append(maskpix, bgrpix[0:diff])

                bgrimg[bgrind] = bgrpix
                bgrimg[maskind] = maskpix

                bgrimg = bgrimg.reshape((img.shape[0], img.shape[0]))
                bgrimgRot = transform.rotate(bgrimg, 180., center=(cenpix_y, cenpix_x), preserve_range=True)
                bgrimgResid = np.ravel(np.abs(bgrimg - bgrimgRot))

                bgrregionResid = bgrimgResid[regionind]

                Abgr = np.sum(bgrregionResid) / (2.*np.sum(np.abs(region)))
                A = A - Abgr
            else:
                Abgr = -99
        else:
            Abgr = -99

    return [A, Abgr]
