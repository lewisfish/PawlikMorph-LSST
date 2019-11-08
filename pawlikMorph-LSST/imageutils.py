from typing import List, Tuple

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from astropy import units
from scipy import ndimage
from scipy import optimize
from skimage import transform

from apertures import distarr
from gaussfitter import twodgaussian, moments


def skybgr(img: np.ndarray, imgsize: int, smallimg=None) -> Tuple[float, float, int]:
    '''Function that calculates the sky background value.

    Parameters
    ----------
    img : np.ndarray
        Image data from which the background will be calculated.
    imgsize : int
        Size of the image.
    smallimg : np.ndarray, optional
        Smaller cutout of object. If provided Gaussian is fitted on this
        instead of larger cutout as the fitter sometimes fails on larger images.

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
    try:
        if smallimg is not None:
            yfit = gauss2dfit(smallimg, smallimg.shape[0])
        else:
            yfit = gauss2dfit(img, imgsize)
        fact = 2 * np.sqrt(2 * np.log(2))
        fwhm_x = fact * np.abs(yfit[4])
        fwhm_y = fact * np.abs(yfit[5])
        r_in = 2. * max(fwhm_x, fwhm_y)
    except RuntimeError:
        print("Running alt gaussgitter")
        yfit = altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)

    skyind = np.nonzero(distarrvar > r_in)
    skyregion = img[skyind]

    if skyregion.shape[0] < 300:
        print("Running alt gaussgitter")
        yfit = altgauss2dfit(img, imgsize)
        fwhm_x = yfit.x_fwhm
        fwhm_y = yfit.y_fwhm
        r_in = 2. * max(fwhm_x, fwhm_y)
        skyind = np.nonzero(distarrvar > r_in)
        skyregion = img[skyind]
        if skyregion.shape[0] < 100:
            return -99, -99, 1

    if skyregion.shape[0] > 100:
        # Flag the measurement if sky region smaller than 20000 pixels
        # (Simard et al. 2011)
        if skyregion.shape[0] < 20000:
            print(f"Warning! skyregion too small {skyregion.shape[0]}")

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


def altgauss2dfit(img: np.ndarray, imgsize: float) -> models:
    '''Alternative slower but more robust Gaussian fitter.

    img : np.ndarray
        Image array to be fitted to.
    imgsize : int
        Size of image in x and y directions.

    Returns
    -------

    g : astropy.modeling.functional_models.Gaussian2D class
        Model parameters of fitted Gaussian.

    '''

    imgold = np.copy(img)

    fit_w = fitting.LevMarLSQFitter()

    total = np.abs(imgold).sum()
    Y, X = np.indices(imgold.shape)  # python convention: reverse x,y np.indices
    mean = np.mean(imgold)
    xmid = int(imgsize / 2)
    ymid = int(imgsize / 2)

    # Mask out bright pixels not near object centre
    while True:
        x, y = np.unravel_index(np.argmax(imgold), shape=imgold.shape)
        if abs(x - xmid) > 20 or abs(y - ymid) > 20:
            imgold[x, y] = mean
        else:
            break

    # subtracting the mean sometimes helps
    mean = np.mean(imgold)
    imgold -= mean

    # make intial guess
    total = np.abs(imgold).sum()
    y0 = int(img.shape[0] / 2)
    x0 = int(img.shape[1] / 2)

    col = imgold[int(y0), :]
    sigmax = np.sqrt(np.abs((np.arange(col.size)-y0)**2*col).sum() / np.abs(col).sum())

    row = imgold[:, int(x0)]
    sigmay = np.sqrt(np.abs((np.arange(row.size)-x0)**2*row).sum() / np.abs(row).sum())

    height = np.median(imgold.ravel())
    amp = imgold.max() - height
    angle = 3.14 / 2.

    # fit model
    w = models.Gaussian2D(amp, x0, y0, sigmax, sigmay, angle)
    yi, xi = np.indices(imgold.shape)
    g = fit_w(w, xi, yi, imgold)

    return g


def cleanimg(img: np.ndarray, pixmap: np.ndarray) -> np.ndarray:
    '''Function that cleans the image of non overlapping sources, i.e outside
       the objects binary pixelmap


    Parameters
    ----------

    img : np.ndarray
        Input image to be cleaned.
    pixmap : np.ndarray
        Binary pixelmap of object.

    Returns
    -------

    imgclean : np.ndarray
        Cleaned image.

    '''

    imgsize = img.shape[0]
    imgravel = img.ravel()
    imgclean = imgravel

    # Dilate pixelmap
    element = np.ones((9, 9))
    mask = ndimage.morphology.binary_dilation(pixmap, structure=element)

    skyind = np.nonzero(mask.ravel() != 1)[0]

    if skyind.size > 10:

        # find a threshold for defining sky pixels
        meansky = np.mean(imgravel[skyind])
        mediansky = np.median(imgravel[skyind])

        if meansky <= mediansky:
            thres = meansky
        else:

            sigmasky = np.std(imgravel[skyind])

            mode_old = 3. * mediansky - 2.*meansky
            mode_new = 0.0
            w = 0
            clipsteps = imgravel.size

            while w < clipsteps:

                skyind = np.nonzero(np.abs(imgravel[skyind] - meansky) < 3.*sigmasky)
                meansky = np.mean(imgravel[skyind])
                mediansky = np.median(imgravel[skyind])
                sigmasky = np.std(imgravel[skyind])

                mode_new = 3.*mediansky - 2.*meansky
                mode_diff = abs(mode_old - mode_new)

                if mode_diff < 0.01:
                    modesky = mode_new
                    w = clipsteps
                else:
                    w += 1

                mode_old = mode_new

            thres = modesky

        # Mask out sources with random sky pixels
        # elemnt wise boolean AND
        skypix = np.nonzero((pixmap.ravel() != 1) & (imgravel > thres))

        skypixels = imgravel[skypix]
        allpixels = skypixels

        if skypixels.size >= imgravel.size:
            print("ERROR! No sources detected! Check image and pixel map.")
        else:
            pixfrac = imgravel.size / skypixels.size
            if pixfrac == float(round(pixfrac)):
                for p in range(1, int(pixfrac)):
                    allpixels = np.append(allpixels, skypixels)
            else:
                for p in range(1, int(pixfrac)):
                    allpixels = np.append(allpixels, skypixels)
                diff = imgravel.shape[0] - allpixels.shape[0]
                allpixels = np.append(allpixels, skypixels[0:diff])

        # shuffle sky pixels randomly
        np.random.shuffle(allpixels)

        imgclean = np.where((pixmap.ravel() != 1) & (imgravel >= thres), allpixels, imgravel)
    imgclean = imgclean.reshape((imgsize, imgsize))

    # convert imgclean from object dtype to float dtype
    imgclean[imgclean == None] = 0.
    imgclean = imgclean.astype(np.float)

    return imgclean


def calcRotation(cd: np.ndarray) -> float:
    # Copyright (c) 2011-2019, Ginga Maintainers

    # All rights reserved.

    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are
    # met:

    # * Redistributions of source code must retain the above copyright
    #   notice, this list of conditions and the following disclaimer.

    # * Redistributions in binary form must reproduce the above copyright
    #   notice, this list of conditions and the following disclaimer in the
    #   documentation and/or other materials provided with the
    #   distribution.

    # * Neither the name of Ginga Maintainers nor the names of its
    #   contributors may be used to endorse or promote products derived from
    #   this software without specific prior written permission.

    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    # IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    # TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    # PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    # HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    # TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    '''Function that calculates rotation from a cd matrix. Adapted from
       Ginga source code: https://ejeschke.github.io/ginga/.

    Parameters
    ----------

    cd : np.ndarray
        Matrix of cd values

    Returns
    -------

    xrot : float
        Angle of the image rotation.
    '''

    cd11 = cd[0, 0]
    cd12 = cd[0, 1]
    cd21 = cd[1, 0]
    cd22 = cd[1, 1]

    det = cd11*cd22 - cd12*cd21
    if det < 0:
        sgn = -1
    else:
        sgn = 1

    if cd21 == 0 or cd12 == 0:
        xrot = 0
        yrot = 0
        cdelt1 = cd11
        cdelt2 = cd22
    else:
        xrot = np.arctan2(sgn * cd12, sgn*cd11)
        yrot = np.arctan2(-cd21, cd22)

        cdelt1 = sgn * np.sqrt(cd11**2 + cd12**2)
        cdelt2 = np.sqrt(cd11**2 + cd21**2)

    xrot = np.degrees(xrot)+180.
    return xrot


def cutoutImg(file: str, ra: float, dec: float, stampsize: int, imgsource=None) -> np.ndarray:
    '''Function to cutout postage stamp of a given pixel size from larger image

    Parameters
    ----------

    file : str
        Name of file to be cutout.
    ra : float
        RA of object in large image.
    dec : float
        DEC of object in large image.
    stampsize:
        Size of image to cutout and return.
    imgsource: str, optional
        Source of image, i.e SDSS, LSST, HSC

    Returns
    -------

    cutout.data : np.ndarray
        Postage stamp image of given size.

    '''

    hdullist = fits.open(file)

    if imgsource:
        if imgsource.lower() == "sdss":
            var = 0
        elif imgsource.lower() == "hsc":
            var = 1
        else:
            print("ERROR! Source not supported!")
            sys.exit()
    else:
        var = 0

    w = wcs.WCS(hdullist[var].header)

    # hdullist[var].scale("int32", "old")  # scale images properly
    data = hdullist[var].data
    data = data.byteswap().newbyteorder()

    # get central pixel position in pixels
    pos = SkyCoord(ra*units.deg, dec*units.deg)
    pos = wcs.utils.skycoord_to_pixel(pos, wcs=w)
    x = pos[0]
    y = pos[1]

    # Calculate rotation of image from CD matrix
    cd = w.wcs.cd
    angle = -calcRotation(cd)

    # rotate image and cutout postage stamp
    img_rot = transform.rotate(data, angle, center=(x, y), preserve_range=True,
                               order=0)
    cutout = Cutout2D(img_rot, pos, (stampsize, stampsize), wcs=w,
                      mode="strict", copy=True)
    hdullist.close()

    return cutout.data
