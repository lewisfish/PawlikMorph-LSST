import numpy as np
from astropy.io import fits
import numba as nb
from typing import List, Tuple
from scipy.optimize import curve_fit
from skimage.transform import resize
from scipy.ndimage import uniform_filter


def makeaperpixmaps(npix: int) -> None:
    '''Writes the aperture binary masks out after calculation.

    Parameters
    ----------
    npix : int
        Width of aperture image.

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
        fits.writeto("aperture" + str(i) + ".fits", aperturepixelmap, overwrite=True)


@nb.njit
def aperpixmap(npix: int, rad: float,  nsubpix: int, frac: float) -> np.ndarray:
    '''Calculates the aperture binary mask through pixel sampling knowing aperture radius and number of subpixels.

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

        apersubpix = (subpixels[i, :, :].flatten()[::-1] <= rad).nonzero()[0]  # This is really inefficient...
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

    npix = imgsize
    cenpix = np.array([int(npix/2) + 1, int(npix/2) + 1])
    distarrvar = distarr(npix, npix, cenpix)

    # Define skyregion by fitting a Gaussian to the galaxy and computing on all
    # outwith this

    yfit = gauss2dfit(img, imgsize)
    fact = 2 * np.sqrt(2 * np.log(2))
    fwhm_x = fact * np.abs(yfit[3])
    fwhm_y = fact * np.abs(yfit[4])
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

            # Begin sigma clipping until convergence/
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

    # convert image to 1D array for fitting purposes
    img = img.ravel()

    # Guess parameters for fit
    amp = np.amax(img)
    xo, yo = np.unravel_index(np.argmax(img), (imgsize, imgsize))
    sigx = np.std(img)
    theta = 0
    offset = 10
    initial_guess = [amp, xo, yo, sigx, sigx, theta, offset]
    xdata = np.vstack((X.ravel(), Y.ravel()))
    popt, pconv = curve_fit(Gaussian2D, (X, Y), img, p0=initial_guess, ftol=1e-4, xtol=1e-5, gtol=1e-2)

    return np.abs(popt)


@nb.njit
def Gaussian2D(xydata: List[float], amplitude: float,
               xo: float, yo: float, sigma_x: float, sigma_y: float,
               theta: float, offset: float) -> List[float]:
    '''Calculates a 2D Gaussian distribution.

    Parameters
    ----------
    xydata : List[float], List[float]
        Stack of x and y values values. xydata[0] is x and xydata[1] is y
    amplitude: float
        Amplitude of Gaussian.
    xo, yo: float
        Centre point of Gaussian distribution.
    sigma_x, sigma_y : float
        Standard deviation of Gaussian distribution in x and y directions.
    theta : float
        Angle of Gaussian distribution.
    offset : float
        Offset of the Gaussian distribution.
    Returns
    -------
    g.ravel() : List[float]
        1D array of computed Gaussian distribution. Array is 1D so that
        function is compatible with Scpiy's curve_fit.
        Parameters are: Amplitude, xo, yo, sigx, sigy, theta, offset
    '''

    x = xydata[0].reshape((141, 141))  # TODO fix this!!!!
    y = xydata[1].reshape((141, 141))

    a = (np.cos(theta)**2) / (2 * sigma_x**2) + \
        (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4 * sigma_x**2) + \
         (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + \
        (np.cos(theta)**2) / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a * ((x - xo)**2) +
                                    2 * b * (x - xo) * (y - yo) +
                                    c * ((y - yo)**2)))
    return g.ravel()


def pixelmap(img: np.ndarray, thres: float, filtsize: int) -> np.ndarray:
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
    img = uniform_filter(img, size=filtsize, mode="reflect")
    # resize image to match PawlikMorph
    # TODO leave this as an option?
    img = resize(img, (47, 47), anti_aliasing=False, preserve_range=True)

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
        x, y = pixels.pop()
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
    objmask = resize(objmask, (141, 141), order=0, mode="edge")
    return objmask


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from astropy.io import fits
    from astropy.utils.exceptions import AstropyWarning
    import sys
    import warnings

    parser = ArgumentParser(description="Prepare images for analysis")

    parser.add_argument("-f", "--file", type=str, help="Path to single image to be prepared")
    parser.add_argument("-fo", "--folder", type=str, help="Path to folder where images are saved")
    parser.add_argument("-A", action="store_true", help="Calculate asymmetry")
    parser.add_argument("-Ao", action="store_true", help="Calculate outer asymmetry")
    parser.add_argument("-As", action="store_true", help="Calculate shape asymmetry")
    parser.add_argument("-Aall", action="store_true", help="Calculate all asymmetries")
    parser.add_argument("-aperpixmap", action="store_true", help="Calculate aperature pixel maps")

    args = parser.parse_args()

    if not args.file and not args.folder:
        print("Script needs input images to work!!")
        sys.exit()

    # add files to a list #TODO more efficient way? i.e generator or something?
    files = []
    if args.file:
        files.append(Path(args.file))
    elif args.folder:
        print("Folders not supported yet!")
        sys.exit()

    # get image size. Assume all images the same size and are square
    warnings.simplefilter('ignore', category=AstropyWarning)  # suppress warnings about unrecognised keywords
    data = fits.getdata(files[0])
    imgsize = data.shape[0]

    # Generate binary aperture masks for computation of light profiles
    # Checks if they already exist, if so skips computation
    if args.aperpixmap:
        if Path("aperture32.fits").exists():
            tmpdata = fits.getdata(Path("aperture32.fits"))
            if tmpdata.shape[0] != imgsize:
                makeaperpixmaps(imgsize)
        else:
            makeaperpixmaps(imgsize)

    for file in files:
        if not file.exists():
            print(f"Fits image:{file.name} does not exist!")
            continue
        data = fits.getdata(file)
        if not data.shape[0] == data.shape[1]:
            print("ERROR: wrong image size. Please preprocess data!")
            sys.exit()
        sky, sky_err, flag = skybgr(data, imgsize)
        print(sky, sky_err, flag)
        mask = pixelmap(data, sky + sky_err, 3)
        fits.writeto("result.fits", mask, overwrite=True)
