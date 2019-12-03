import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from astropy import wcs

__all__ = ["make_figure"]


def _normalise(image):
    m, M = np.min(image), np.max(image)
    return (image - m) / (M - m)


def _supressAxs(ax):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _getStarsOccludObject(file, header):

    starcat = "exp/output/occluded-object-locations.csv"
    names, RAs, DECs, *rest = np.loadtxt(starcat, unpack=True, skiprows=1,
                                         delimiter=",", dtype=str)

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
        pass

    return occludingStars


def make_figure(result):

    log_stretch = LogStretch(10000.)

    file = str(result.file)
    img, header = fits.getdata(file, header=True)

    filemask = file.replace("data/sdssc", "output/pixelmap_sdssc")
    mask = fits.getdata(filemask)

    listofStarstoPlot = _getStarsOccludObject(file, header)
    if len(listofStarstoPlot) > 0:
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(11.25)
        fig.set_figwidth(20)

        axs[0].imshow(log_stretch(_normalise(img)), origin="lower", aspect="auto")
        axs[0].scatter(result.apix[0], result.apix[1], label="Asym. centre")
        if(len(listofStarstoPlot) > 0):
            axs[0].scatter(*zip(*listofStarstoPlot), label="STAR")

        text = f"Sky={result.sky:.2f}\n" fr"Sky $\sigma$={result.sky_err:.2f}"

        axs[0].text(2, 7, text,
                    horizontalalignment='left', verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))

        _supressAxs(axs[0])

        axs[1].imshow(mask, origin="lower", aspect="auto", cmap="gray")
        axs[1].scatter(result.apix[0], result.apix[1], label="Asym. centre")
        if(len(listofStarstoPlot) > 0):
            axs[1].scatter(*zip(*listofStarstoPlot), label="STAR")

        text = f"A={result.A[0]:.3f}\nA_bgr={result.A[1]:.3f}\n" rf"$A_s$={result.As[0]:.3f}"
        text += "\n" fr"$A_s90$={result.As90[0]:.3f}"

        axs[1].text(2, 13, text,
                    horizontalalignment='left', verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
        circle = mpatches.Circle(result.apix, result.rmax, fill=False, label="Rmax", color="white")
        axs[1].add_patch(circle)
        _supressAxs(axs[1])

        plt.subplots_adjust(top=0.985, bottom=0.015, left=0.008, right=0.992, hspace=0.2, wspace=0.016)
        plt.legend()
        # plt.savefig("result_" + file[20:-9] + "_old.png", dpi=96)
        plt.show()
