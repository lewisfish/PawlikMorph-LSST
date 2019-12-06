import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astropy.visualization import LogStretch
from astropy import units
from astropy import wcs

__all__ = ["make_figure"]


def _normalise(image: np.ndarray):
    '''Function normalises an array s.t it over a range[0., 1.]

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


def _getStarsOccludObject(file, header, outfolder, occludedFile):

    starcat = outfolder / occludedFile
    try:
        names, RAs, DECs, *rest = np.loadtxt(starcat, unpack=True, skiprows=1,
                                             delimiter=",", dtype=str)
    except ValueError:
        print(f"{starcat} file empty!")
        return []

    # If there is only one entry in csv file then loadtxt returns str for each
    # variable. Thefore we check if length of names and RAs match, if they dont
    # match put str in a list else just turn names into a list as other should
    # already be a list
    if len(RAs) != len(names):
        names = [names]
        RAs = [RAs]
        DECs = [DECs]
    else:
        names = list(names)
    occludingStars = []

    try:
        ind = names.index(file)
        w = wcs.WCS(header)

        skyCoordPos = SkyCoord(RAs[ind], DECs[ind], unit="deg")
        x, y = wcs.utils.skycoord_to_pixel(skyCoordPos, wcs=w)
        occludingStars.append([x, y])

        for i in range(1, len(names) - ind):
            if names[ind + i] == "":
                skyCoordPos = SkyCoord(RAs[ind+i], DECs[ind+i], unit="deg")
                x, y = wcs.utils.skycoord_to_pixel(skyCoordPos, wcs=w)
                occludingStars.append([x, y])
            else:
                break

    except ValueError:
        return occludingStars

    return occludingStars


def make_figure(result, save=False):
    '''Function plots results from image analysis. Plots two images
       Left: original image with stars overplotted if any
       Right: object pixel map and stars overplotted if any.

    Parameters
    ----------

    result : Results class
        Data class container of calculated results.

    save : bool, optional
        If true function saves generated figure.

    Returns
    -------


    '''

    log_stretch = LogStretch(10000.)

    img, header = fits.getdata(result.cleanImage, header=True)

    filemask = result.pixelMapFile
    mask = fits.getdata(filemask)

    if result.occludedFile != "":
        listofStarstoPlot = _getStarsOccludObject(result.file, header, result.outfolder, result.occludedFile)
    else:
        listofStarstoPlot = []

    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    fig.set_figheight(11.25)
    fig.set_figwidth(20)

    axs[0].imshow(log_stretch(_normalise(img)), origin="lower", aspect="auto")
    axs[0].scatter(result.apix[0], result.apix[1], label="Asym. centre")
    if(len(listofStarstoPlot) > 0):
        axs[0].scatter(*zip(*listofStarstoPlot), label="STAR")

    text = f"Sky={result.sky:.2f}\n" fr"Sky $\sigma$={result.sky_err:.2f}"

    axs[0].text(2, 13, text,
                horizontalalignment='left', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))

    axs[1].imshow(mask, origin="lower", aspect="auto", cmap="gray")
    axs[1].scatter(result.apix[0], result.apix[1], label="Asym. centre")
    if(len(listofStarstoPlot) > 0):
        axs[1].scatter(*zip(*listofStarstoPlot), label="STAR")

    text = f"A={result.A[0]:.3f}\nA_bgr={result.A[1]:.3f}\n" rf"$A_s$={result.As[0]:.3f}"
    text += "\n" fr"$A_s90$={result.As90[0]:.3f}"

    axs[1].text(2, 22, text,
                horizontalalignment='left', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    circle = mpatches.Circle(((img.shape[0]/2)+1, (img.shape[1]/2)+1), result.rmax, fill=False, label="Rmax", color="white")
    axs[1].add_patch(circle)

    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]
    modelimage = models.Sersic2D.evaluate(x, y, result.sersic_amplitude,
                                          result.sersic_r_eff, result.sersic_n,
                                          result.sersic_x_0, result.sersic_y_0,
                                          result.sersic_ellip, result.sersic_theta)

    modelimage += np.random.normal(result.sky, result.sky_err, size=img.shape)
    axs[2].imshow(log_stretch(_normalise(modelimage)), origin="lower", aspect="auto")
    axs[2].scatter(result.sersic_x_0, result.sersic_y_0, label="Sersic centre")

    text = f"Ellip.={result.sersic_ellip[0]:.3f}\n"
    text += f"n={result.sersic_n[0]:.3f}"
    axs[2].text(2, 22, text,
                horizontalalignment='left', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))

    a = result.sersic_r_eff
    b = a * np.abs(1. - result.sersic_ellip)
    x0 = result.sersic_x_0
    y0 = result.sersic_y_0
    theta = result.sersic_theta * 180./np.pi
    ellipse = mpatches.Ellipse(xy=(x0, y0), width=a, height=b, angle=theta, fill=False, label="Sersic half light", color="red")
    axs[2].add_patch(ellipse)

    axs[3].imshow(_normalise(modelimage - img), origin="lower", aspect="auto")

    for ax in axs:
        ax = _supressAxs(ax)

    plt.subplots_adjust(top=0.995, bottom=0.005, left=0.003, right=0.997, hspace=0.018, wspace=0.006)
    axs[0].legend()
    axs[1].legend()

    if save:
        plt.savefig("result_" + file[20:-9] + ".png", dpi=96)
    plt.show()
