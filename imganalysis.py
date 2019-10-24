import numpy as np
from astropy.io import fits
import numba as nb
from typing import List


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

    # add files to a list #TODO more efficient way? i.e generator or something?
    files = []
    if args.file:
        files.append(Path(args.file))
    elif args.folder:
        files = Path(args.folder)

    # get image size. Assume all images the same size and are square
    warnings.simplefilter('ignore', category=AstropyWarning)  # supress warnings about unrecognizsed keywords
    hdul = fits.open(files[0])
    imgsize = hdul[0].data.shape[0]

    # Generate binary aperture masks for compuation of light profiles
    # Checks if they already exist, if so skips compuation
    if args.aperpixmap:
        if Path("aperture32.fits").exists():
            tmphdul = fits.open(Path("aperture32.fits"))
            tmpdata = tmphdul[0].data
            if tmpdata.shape[0] != imgsize:
                makeaperpixmaps(imgsize)
        else:
            makeaperpixmaps(imgsize)

    for file in files:
        if not file.exists():
            print(f"Fits image:{file.name} does not exist!")
            continue
        hdul = fits.open(file)
        data = hdul[0].data
        if not data.shape[0] == data.shape[1]:
            print("ERROR: wrong image size. Please preprocess data!")
            sys.exit()
