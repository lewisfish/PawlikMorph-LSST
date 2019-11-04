from typing import List, Tuple

import numba as nb
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy import optimize
from skimage import transform

from gaussfitter import twodgaussian, moments


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

    print("Creating aperture pixel maps.")

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
    '''Calculates the aperture binary mask through pixel sampling knowing
       aperture radius and number of subpixels.

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
    '''Writes the aperture binary masks out after calculation.

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

    x1 = np.arange(npixx) - cenpix[0]
    x2 = np.ones_like(x1)

    y1 = np.arange(npixy) - cenpix[1]
    y2 = np.ones_like(y1)

    pixx = np.transpose(np.outer(x1, y2))
    pixy = np.transpose(np.outer(x2, y1))

    dist = np.sqrt(pixx**2 + pixy**2)

    return dist


@nb.njit(nb.float64[:, :](nb.int64, nb.int64, nb.int64[:]))
def subdistarr(npix: int, nsubpix: int, cenpix: List[int]) -> np.ndarray:
    '''Writes the aperture binary masks out after calculation.

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


def skybgr(img: np.ndarray, imgsize: int) -> Tuple[float, float, int]:
    '''Function that calculates the sky background value.

    Parameters
    ----------
    img : np.ndarray
        Image data from which the background will be calculated.
    imgsize : int
        Size of the image.

    Returns
    -------

    sky, sky_err, and flag : float, float, int
        sky is the sky background measurement.
        sky_err is the error in that measurement.
        flag indicates that the image size was less than ideal.
    '''

    flag = 0
    npix = imgsize
    cenpix = np.array([int(npix/2) + 1, int(npix/2) + 1])
    distarrvar = distarr(npix, npix, cenpix)

    # Define skyregion by fitting a Gaussian to the galaxy and computing on all
    # outwith this

    yfit = gauss2dfit(img, imgsize)
    fact = 2 * np.sqrt(2 * np.log(2))
    fwhm_x = fact * np.abs(yfit[4])
    fwhm_y = fact * np.abs(yfit[5])
    r_in = 2. * max(fwhm_x, fwhm_y)

    skyind = np.nonzero(distarrvar > r_in)
    skyregion = img[skyind]

    if skyregion.shape[0] > 100:
        # Flag the measurement if sky region smaller than 20000 pixels
        # (Simard et al. 2011)
        if skyregion.shape[0] < 20000:
            flag = 1

        mean_sky = np.mean(skyregion)
        median_sky = np.median(skyregion)
        sigma_sky = np.std(skyregion)

        if mean_sky <= median_sky:
            # non crowded region. Use mean for background measurement
            sky = mean_sky
            sky_err = sigma_sky
        else:
            # crowded region. Use mode for background measurement
            mode_old = 3.*median_sky - 2.*mean_sky
            mode_new = 0.0
            w = 0
            clipsteps = skyregion.shape[0]

            # Begin sigma clipping until convergence
            while w < clipsteps:

                skyind = np.nonzero(np.abs(skyregion - mean_sky) < 3. * sigma_sky)
                skyregion = skyregion[skyind]

                mean_sky = np.mean(skyregion)
                median_sky = np.median(skyregion)
                sigma_sky = np.std(skyregion)
                mode_new = 3.*median_sky - 2.*mean_sky
                mode_diff = np.abs(mode_old - mode_new)

                if mode_diff < 0.01:
                    mode_sky = mode_new
                    w = clipsteps
                else:
                    w += 1
                mode_old = mode_new

            sky = mode_sky
            sky_err = sigma_sky
    else:
        sky = -99
        sky_err = -99
        flag = 1

    return sky, sky_err, flag


def gauss2dfit(img: np.ndarray, imgsize: int) -> List[float]:
    '''Function that fits a 2D Gaussian to image data.

    Parameters
    ----------

    img : np.ndarray
        Image array to be fitted to.
    imgsize : int
        Size of image in x and y directions.

    Returns
    -------

    popt : List[float]
        List of calculated parameters from curve_fit.

    '''

    x = np.linspace(0, imgsize-1, imgsize)
    y = np.linspace(0, imgsize-1, imgsize)
    X, Y = np.meshgrid(x, y)

    # guess intital parameters from calculation of moments
    initial_guess = moments(img)

    # convert image to 1D array for fitting purposes
    imgravel = img.ravel()

    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pconv = optimize.curve_fit(twodgaussian, (X, Y), imgravel, p0=initial_guess)

    return np.abs(popt)


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

    # TODO. move somewhere else?
    if img.shape[0] != img.shape[1]:
        print("ERROR! image must be square")
        sys.exit()

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

            regionpix_2d = np.unravel_index(sortedind[ii], shape=(npix, npix))
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
    centroid = np.unravel_index(centroid_ind, shape=(npix, npix))

    return centroid


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


def calcA(img: np.ndarray, pixmap: np.ndarray, apermask: np.ndarray,
          centroid: List[int], angle: float, apermaskcut=None,
          noisecorrect=False) -> List[float]:
    """Function to calculate A, the asymmetery parameter. Near direct translation
       of IDL code.

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
                            maskpix = [maskpix, bgrpix]  # these line currently cause the code to fail to executed
                    else:
                        for p in range(1, int(pixfrac)):
                            maskpix = [maskpix, bgrpix]  # these line currently cause the code to fail to executed
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


if __name__ == '__main__':
    import sys
    import time
    import warnings
    from argparse import ArgumentParser
    from pathlib import Path

    from astropy.io import fits
    from astropy.utils.exceptions import AstropyWarning

    parser = ArgumentParser(description="Prepare images for analysis")

    parser.add_argument("-f", "--file", type=str, help="Path to single image to be analysed")
    parser.add_argument("-fo", "--folder", type=str, help="Path to folder where images are to be analysed")
    parser.add_argument("-A", action="store_true", help="Calculate asymmetry parameter")
    # parser.add_argument("-Ao", action="store_true", help="Calculate outer asymmetry parameter")
    parser.add_argument("-As", action="store_true", help="Calculate shape asymmetry parameter")
    parser.add_argument("-Aall", action="store_true", help="Calculate all asymmetries parameters")
    parser.add_argument("-aperpixmap", action="store_true", help="Calculate aperature pixel maps")

    args = parser.parse_args()

    if not args.file and not args.folder:
        print("Script needs input images to work!!")
        sys.exit()

    # get image size. Assume all images the same size and are square
    # suppress warnings about unrecognised keywords
    warnings.simplefilter('ignore', category=AstropyWarning)
    # add files to a list
    files = []
    if args.file:
        files.append(Path(args.file))
    elif args.folder:
        # TODO change to a generator
        files = list(Path(args.folder).glob("sdss*.fits"))
    data = fits.getdata(files[0])
    imgsize = data.shape[0]

    # Generate binary aperture masks for computation of light profiles
    # Checks if they already exist, if so skips computation
    # TODO: probably could do this better
    if args.aperpixmap:
        cenpixtmp = (imgsize / 2.) + 1
        tmp = np.arange(cenpixtmp) + 1.
        aperpath = Path(args.folder).parents[0] / "aperpixmaps/"
        if len(list(aperpath.glob("aperture*.fits"))) == int(cenpixtmp) + 1:
            tmpdata = fits.getdata(aperpath / "aperture50.fits")
            if tmpdata.shape[0] != imgsize:
                makeaperpixmaps(imgsize)
        else:
            try:
                aperpath.mkdir()
            except FileExistsError:
                # folder already exists so pass
                pass
            makeaperpixmaps(imgsize, aperpath)

    for file in files:
        if not file.exists():
            print(f"Fits image:{file.name} does not exist!")
            continue
        print(file)
        data = fits.getdata(file)
        imgsize = data.shape[0]
        # The following is required as fits files are big endian and skimage
        # assumes little endian.
        # https://stackoverflow.com/a/30284033/6106938
        # https://en.wikipedia.org/wiki/Endianness
        # https://stackoverflow.com/a/30284033/6106938
        data = data.byteswap().newbyteorder()

        if not data.shape[0] == data.shape[1]:
            print("ERROR: wrong image size. Please preprocess data!")
            sys.exit()

        # get skybackground value and error
        sky, sky_err, flag = skybgr(data, imgsize)  # TODO: warn if flag is 1
        if flag == 1:
            print(f"ERROR! Skybgr not calculated for {file}")
        mask = pixelmap(data, sky + sky_err, 3)
        # mask = fits.getdata("target.fits")
        # fits.writeto("result.fits", mask, overwrite=True)
        data -= sky

        objectpix = np.nonzero(mask == 1)
        cenpix = np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

        distarray = distarr(imgsize, imgsize, cenpix)
        objectdist = distarray[objectpix]
        r_max = np.max(objectdist)
        aperturepixmap = aperpixmap(imgsize, r_max, 9, 0.1)

        apix = minapix(data, mask, aperturepixmap)
        angle = 180.

        if args.A or args.Aall:
            A = calcA(data, mask, aperturepixmap, apix, angle, noisecorrect=True)

        if args.As or args.Aall:
            As = calcA(mask, mask, aperturepixmap, apix, angle)
            if As[1] == 0:
                print(f"As_180={As[0]}")
            else:
                print("ERROR! Flag != 0 in calcA (180)")
            As90 = calcA(mask, mask, aperturepixmap, apix, 90.)
            if As90[1] == 0:
                print(f"As_90={As90[0]}")
            else:
                print("ERROR! Flag != 0 in calcA (90)")
