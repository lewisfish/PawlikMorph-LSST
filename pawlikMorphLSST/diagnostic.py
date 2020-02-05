from typing import List
import warnings

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from astropy import units
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import LogStretch
from matplotlib.offsetbox import AnchoredText
from scipy.ndimage import gaussian_filter

__all__ = ["make_figure"]


def _normalise(image: np.ndarray):
    '''Function normalises an array s.t it is over a range[0., 1.]

    Parameters
    ----------

    image : np.ndarray
        Image to be normalised.

    Returns
    -------

    Normalised image: np.ndarray.
    '''

    m, M = np.min(image), np.max(image)
    return (image - m) / (M - m)


def _supressAxs(ax):
    '''Function that removes all labels and ticks from a figure

    Parameters
    ----------

    ax: matplotlib axis object

    Returns
    -------

    ax : matplotlib axis object
        Now with no ticks or labels

    '''

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return ax


def RADECtopixel(objList: List[List[float]], header) -> List[List[float]]:
    '''Function to convert RA DEC in objList to pixel coordinates using
       wcs in header of image

    Parameters
    ----------

    objList : List[List[float]]
        List of list of RA, DEC, object type and psfMag_r

    header :

    Returns
    -------

    occludingStars : List[List[float]]
        List of RA, DEC in pixel coordinates.
    '''

    occludingStars = []

    with warnings.catch_warnings():
        # ignore invalid card warnings
        warnings.simplefilter('ignore', category=AstropyWarning)
        w = wcs.WCS(header)

    RAS = [item[0] for item in objList]
    DECS = [item[1] for item in objList]

    for ra, dec in zip(RAS, DECS):
        skyCoordPos = SkyCoord(ra, dec, unit="deg")
        x, y = wcs.utils.skycoord_to_pixel(skyCoordPos, wcs=w)
        occludingStars.append([x, y])

    return occludingStars


def make_oneone(ax, img, result):
    '''Function plots the cleaned image

    Parameters
    ----------

    ax : matplotlip axis object

    img : np.ndarray
        image data to be plotted

    results : Result dataclass
        dataclass of calculated results for object

    Returns
    -------

    '''

    log_stretch = LogStretch(10000.)

    ax.imshow(log_stretch(_normalise(img)), origin="lower", aspect="auto")
    ax.scatter(result.apix[0], result.apix[1], label="Asym. centre")
    ax.set_xlim([-0.5, img.shape[0]+0.5])
    ax.set_title("Cleaned Image")

    text = f"Sky={result.sky:.2f}\n" fr"Sky $\sigma$={result.sky_err:.2f}"
    textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
    ax.add_artist(textbox)


def make_onetwo(ax, mask, result):
    '''Function plots the object map

    Parameters
    ----------

    ax : matplotlib axis object

    mask : np.ndarray
        object mask data to be plotted

    results : Result dataclass
        dataclass of calculated results for object

    Returns
    -------

    '''

    ax.imshow(mask, origin="lower", aspect="auto", cmap="gray")
    ax.scatter(result.apix[0], result.apix[1], label="Asym. centre")
    ax.set_xlim([-0.5, mask.shape[0]+0.5])
    ax.set_ylim([-0.5, mask.shape[1]+0.5])
    ax.set_title("Object mask")

    text = f"A={result.A[0]:.3f}\nA_bgr={result.A[1]:.3f}\n" rf"$A_s$={result.As[0]:.3f}"
    text += "\n" fr"$A_s90$={result.As90[0]:.3f}"
    if len(result.objList) > 0:
        text += f"\nmaskedFraction={result.maskedPixelFraction*100.:.1f}"
    textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
    ax.add_artist(textbox)

    circle = mpatches.Circle(((mask.shape[0]/2)+1, (mask.shape[1]/2)+1),
                             result.rmax, fill=False, label="Rmax", color="white")
    ax.add_patch(circle)


def make_twoone(ax, shape, result):
    '''Function plots the Sersic fit

    Parameters
    ----------

    ax : matplotlib axis object
        axis instance to plot to

    shape : Tuple[int]
        Shape of image

    results : Result dataclass
        dataclass of calculated results for object

    Returns
    -------

    modelimage : np.ndarray
        fitted model Sersic image


    '''

    log_stretch = LogStretch(10000.)

    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    modelimage = models.Sersic2D.evaluate(x, y, result.sersic_amplitude,
                                          result.sersic_r_eff, result.sersic_n,
                                          result.sersic_x_0, result.sersic_y_0,
                                          result.sersic_ellip, result.sersic_theta)

    modelimage += np.random.normal(result.sky, result.sky_err, size=shape)
    ax.imshow(log_stretch(_normalise(modelimage)), origin="lower", aspect="auto")
    ax.scatter(result.sersic_x_0, result.sersic_y_0, label="Sersic centre")
    ax.set_title("Sersic fit")

    text = f"Ellip.={result.sersic_ellip:.3f}\n"
    text += f"n={result.sersic_n:.3f}\n r_eff={result.sersic_r_eff:.3f}\n"
    text += f"Amplitude={result.sersic_amplitude:.3f}"
    textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
    ax.add_artist(textbox)

    a = result.sersic_r_eff
    b = a * np.abs(1. - result.sersic_ellip)
    x0 = result.sersic_x_0
    y0 = result.sersic_y_0
    theta = result.sersic_theta * 180./np.pi
    ellipse = mpatches.Ellipse(xy=(x0, y0), width=a, height=b, angle=theta, fill=False, label="Sersic half light", color="red")
    ax.add_patch(ellipse)

    return modelimage


def make_twotwo(ax, img, modelImage, listofStarstoPlot, result):
    ''' function plots sersic fit residual

    Parameters
    ----------

    ax : matplotlip axis object
        axis instance to plot to

    img : np.ndarray
        image data to be plotted

    modelImage : np.ndarray
        model sersic image

    listofStarstoPlot : List[List[float]]
        list of stars to that occlude the main object. [RA, DEC, name, psfMag_r]

    results : Result dataclass
        dataclasss of calculated results for object

    Returns
    -------

    '''

    if len(listofStarstoPlot) > 0:
        imageMask = np.where(result.starMask == 1, img, np.rot90(img))
        residual = (imageMask - modelImage)
        ax.imshow(residual, origin="lower", aspect="auto")
    else:
        residual = (img - modelImage)
        ax.imshow(residual, origin="lower", aspect="auto")

    text = f"Range={np.amin(residual):.3e} => {np.amax(residual):.3e}\n"
    textbox = AnchoredText(text, frameon=True, loc=3, pad=0.5)
    ax.add_artist(textbox)

    ax.set_title("Sersic fit residual")


def make_figure(result, folder, save=False, show=False):
    '''Function plots results from image analysis. Plots two or four images.
       Top row: original image  and object map with stars overplotted if any.
       bottom row: Sersic fit and residual with stars overplotted if any.

    Parameters
    ----------

    result : Results class
        Data class container of calculated results.

    folder : bool
        If True then adjusts path to read file from.

    save : bool, optional
        If true function saves generated figure.

    show: bool, optional
        If true open interactive matplotlib plot.

    Returns
    -------


    '''

    with warnings.catch_warnings():
        # ignore invalid card warnings
        warnings.simplefilter('ignore', category=AstropyWarning)
        try:
            img, header = fits.getdata(result.cleanImage, header=True)
        except ValueError:
            if folder:
                img, header = fits.getdata(result.outfolder.parent / ("data/" + result.file), header=True)
            else:
                img, header = fits.getdata(result.outfolder.parent / (result.file), header=True)

        try:
            mask = fits.getdata(result.pixelMapFile)
        except ValueError:
            mask = fits.getdata(result.outfolder / ("pixelmap_" + result.file))

    if result.sersic_r_eff != -99 and result.sky != -99:
        fig, axs = plt.subplots(2, 2)
        axs = axs.ravel()
        make_oneone(axs[0], img, result)
        make_onetwo(axs[1], mask, result)
        modelImage = make_twoone(axs[2], img.shape, result)
        make_twotwo(axs[3], img, modelImage, result.objList, result)
    else:
        fig, axs = plt.subplots(1, 2)
        make_oneone(axs[0], img, result)
        axs[0].set_ylim([-0.5, img.shape[1]+0.5])
        make_onetwo(axs[1], mask, result)
        axs[1].set_ylim([-0.5, mask.shape[1]+0.5])

    fig.set_figheight(11.25)
    fig.set_figwidth(20)

    if len(result.objList) > 0:
        occludingStars = RADECtopixel(result.objList, header)

    for i, ax in enumerate(axs):
        ax = _supressAxs(ax)
        if(len(result.objList) > 0):
            if i != 2:
                ax.scatter(*zip(*occludingStars), label="STAR", color="orange")
        if i != 3:
            ax.legend(loc=2)

    plt.subplots_adjust(top=0.975, bottom=0.005, left=0.003, right=0.997, hspace=0.050, wspace=0.006)

    if save:
        plt.savefig("results/result_" + result.file[11:-11] + ".png", dpi=96)
    if show:
        plt.show()
    plt.close()
