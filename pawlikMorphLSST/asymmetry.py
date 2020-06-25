from typing import List, Tuple

import numpy as np
from scipy import ndimage
from skimage import transform

from . apertures import apercentre, aperpixmap
from . pixmap import calcRmax

__all__ = ["minapix", "calcA", "calculateAsymmetries"]


def calculateAsymmetries(image: np.ndarray, pixelmap: np.ndarray) -> Tuple[float]:
    """helper function to calculate all asymmetries

    Parameters
    ----------

    image : np.ndarray, 2d float
        image of a galaxy for which the asymmetries should be calculated.

    pixelmap : np.ndarray, 2d uint8
        Pixel mask of the galaxy calculated from image.

    Returns
    -------

    A, As, As90 : Tuple, float
        The calculated asymmetry values.

    """

    Rmax = calcRmax(image, pixelmap)
    aperturepixmap = aperpixmap(image.shape[0], Rmax, 9, 0.1)

    starmask = np.ones_like(image)
    apix = minapix(image, pixelmap, aperturepixmap, starmask)
    angle = 180.

    A = calcA(image, pixelmap, aperturepixmap, apix, angle, starmask, noisecorrect=True)

    As = calcA(pixelmap, pixelmap, aperturepixmap, apix, angle, starmask)

    angle = 90.
    As90 = calcA(pixelmap, pixelmap, aperturepixmap, apix, angle, starmask)

    return A[0], As[0], As90[0]


def minapix(image: np.ndarray, mask: np.ndarray, apermask: np.ndarray,
            starMask=None) -> List[int]:
    """Find the pixel that minimises the asymmetry parameter, A

       Selects a range of candidate centroids within the brightest region that
       compromises of 20% of the total flux within object.
       Then measures the asymmetry of the image under rotation around that
       centroid.
       Then picks the centroid that yields the minimum A value.


    Parameters
    ----------

    image : np.ndarray
        Image that the minimum asymmetry pixel is to be found in.
    mask : np.ndarray
        Precomputed mask that describes where the object of interest is in the
        image
    apermask : np.ndarray
        Precomputed aperture mask
    starMask : np.ndarray
        Precomputed mask that masks stars that interfere with object
        measurement

    Returns
    -------

    Centroid : List[int]
        The minimum asymmetry pixel position.
    """

    if starMask is not None:
        # mask the image with object mask and star mask if provided
        imageMask = image * mask * starMask
    else:
        imageMask = image * mask

    # only want top 20% brightest pixels
    TWENTYPERCENT = 0.2

    # calculate flux percentage and sort pixels by flux value
    twentyPercentFlux = TWENTYPERCENT * np.sum(imageMask)
    imageMaskRavel = np.ravel(imageMask)
    sortedImageMask = np.sort(imageMaskRavel)[::-1]
    sortedIndices = np.argsort(imageMaskRavel)[::-1]

    count = 0
    fluxSum = 0
    centroidCandidates = []
    # Generate centroid candidates from brightest 20% pixels
    for j, pixel in enumerate(sortedImageMask):
        x, y = np.unravel_index(sortedIndices[j], shape=image.shape)
        if pixel > 0:
            count += 1
            fluxSum += pixel
            centroidCandidates.append([y, x])
        if fluxSum >= twentyPercentFlux:
            break

    a = np.zeros(count)

    # test centroid candidates for minima of asymmetry
    for i, point in enumerate(centroidCandidates):
        imageRotate = transform.rotate(imageMask, 180., center=point, preserve_range=True)
        imageResidual = np.abs(imageMask - imageRotate)
        imageResidualRavel = np.ravel(imageResidual)

        regionMask = apercentre(apermask, point)
        regionIndicies = np.nonzero(np.ravel(regionMask) == 1)[0]
        region = imageMaskRavel[regionIndicies]
        regionResidual = imageResidualRavel[regionIndicies]

        regionMask *= 0

        a[i] = np.sum(regionResidual) / (2. * np.sum(np.abs(region)))

    aMinimumIndex = np.argmin(a)

    return centroidCandidates[aMinimumIndex]


def calcA(img: np.ndarray, pixmap: np.ndarray, apermask: np.ndarray,
          centroid: List[int], angle: float, starMask=None,
          noisecorrect=False) -> List[float]:
    r"""Function to calculate A, the asymmetry parameter.

    .. math::
        A=\frac{\sum\left|I_0-I_{\theta}\right|}{2\sum I_0}-A_{bgr}

    Where :math:`I_0` is the original image, :math:`I_{\theta}` is the image rotated by :math:`\theta` degrees, and :math:`A_{bgr}` is the asymmerty of the sky.

    See `Conselice et al. <https://doi.org/10.1086/308300>`_ for full details.

    Near direct translation of IDL code.

    Parameters
    ----------

    img : np.ndarray
        Image to be analysed.

    pixmap : np.ndarray
        Mask that covers object of interest.

    apermask : np.ndarray
        Array of the aperture mask image.

    centroid : np.ndarray
        Pixel position of the centroid to be used for rotation.

    angle : float
        Angle to rotate object, in degrees.

    starMask : np.ndarray
        Precomputed mask that masks stars that interfere with object
        measurement

    noisecorrect : bool, optional
        Default value False. If true corrects for background noise

    Returns
    -------

    A, Abgr : List(float)
        Returns the asymmetry value and its background value.

    """

    cenpix_x = centroid[0]
    cenpix_y = centroid[1]

    if starMask is None:
        starMaskCopy = np.ones_like(img)
    else:
        # cast to float so that rotate is happy
        starMaskCopy = starMask.astype(np.float64)
        # rotate the star mask angle degrees, so that star does not interfere
        # with measurement
        starMaskCopy *= transform.rotate(starMaskCopy, angle,
                                         center=(cenpix_x, cenpix_y),
                                         preserve_range=True, cval=1.)

    # mask image
    imgCopy = img * starMaskCopy
    imgRot = transform.rotate(imgCopy, angle, center=(cenpix_x, cenpix_y),
                              preserve_range=True)

    imgResid = np.abs(imgCopy - imgRot)
    imgravel = np.ravel(imgCopy)

    netmask = apermask

    imgResidravel = np.ravel(imgResid)
    regionind = np.nonzero(np.ravel(netmask) == 1)[0]
    region = imgravel[regionind]
    regionResid = imgResidravel[regionind]

    A = np.sum(regionResid) / (2. * np.sum(np.abs(region)))
    Abgr = 0
    if noisecorrect:

        # build "background noise" image using morphological dilation
        # https://en.wikipedia.org/wiki/Dilation_(morphology)
        bgrimg = np.zeros_like(imgCopy)
        bgrimg = np.ravel(bgrimg)
        element = np.ones((9, 9))

        # mask pixel map
        pixmap *= starMaskCopy
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

                bgrimg = bgrimg.reshape((imgCopy.shape[0], imgCopy.shape[0]))
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
