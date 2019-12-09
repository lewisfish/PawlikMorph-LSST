from typing import List

import numpy as np
from scipy import ndimage
from skimage import transform

from .apertures import apercentre

__all__ = ["minapix", "calcA"]


def minapix(image: np.ndarray, mask: np.ndarray, apermask: np.ndarray,
            starMask: np.ndarray) -> List[int]:
    """Function that finds the minimum asymmetry central pixel within the
       objects pixels of a given image.
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

    # mask the image with object mask and star mask if provided
    imageMask = image * mask * starMask

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

    imageRavel = np.ravel(image)
    a = np.zeros(count)

    # test centroid candidates for minima of asymmetry
    for i, point in enumerate(centroidCandidates):
        imageRotate = transform.rotate(image, 180., center=point, preserve_range=True)
        imageResidual = np.abs(image - imageRotate)
        imageResidualRavel = np.ravel(imageResidual)

        regionMask = apercentre(apermask, point)
        regionIndicies = np.nonzero(np.ravel(regionMask) == 1)[0]
        region = imageRavel[regionIndicies]
        regionResidual = imageResidualRavel[regionIndicies]

        regionMask *= 0

        a[i] = np.sum(regionResidual) / (2. * np.sum(np.abs(region)))

    aMinimum = np.min(a)
    aMinimumIndex = np.argmin(a)

    return centroidCandidates[aMinimumIndex]


def calcA(img: np.ndarray, pixmap: np.ndarray, apermask: np.ndarray,
          centroid: List[int], angle: float, starMask: np.ndarray,
          noisecorrect=False) -> List[float]:
    """Function to calculate A, the asymmetry parameter. Near direct
       translation of IDL code.

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
        Precomputed mask that masks stars that interfere with object measurement

    noisecorrect : bool, optional
        Default value False. If true corrects for background noise

    Returns
    -------

    A, Abgr : List(float)
        Returns the asymmetry value and its background value.

    """

    cenpix_x = centroid[0]
    cenpix_y = centroid[1]

    # cast to float so that rotate is happy
    starMask = starMask.astype(np.float64)
    # rotate the star mask angle degrees, so that star does not interfere
    # with measurement
    starMask *= transform.rotate(starMask, angle, center=(cenpix_x, cenpix_y),
                                 preserve_range=True, cval=1.)

    # mask image
    img *= starMask
    imgRot = transform.rotate(img, angle, center=(cenpix_x, cenpix_y),
                              preserve_range=True)

    imgResid = np.abs(img - imgRot)
    imgravel = np.ravel(img)

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
        bgrimg = np.zeros_like(img)
        bgrimg = np.ravel(bgrimg)
        element = np.ones((9, 9))

        # mask pixel map
        pixmap *= starMask
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
