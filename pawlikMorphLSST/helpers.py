import csv
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from .apertures import aperpixmap
from .apertures import distarr
from .asymmetry import calcA
from .asymmetry import minapix
from .imageutils import maskstarsPSF
from .imageutils import maskstarsSEG
from .imageutils import skybgr
from .objectMasker import objectOccluded
from .sersic import fitSersic
from .pixmap import pixelmap

__all__ = ["checkFile", "getLocation", "calcMorphology"]


@dataclass
class Result:
    '''Class that stores the results of image analysis'''

    file: str
    outfolder: Any
    occludedFile: str
    pixelMapFile: Any = ""
    cleanImage: Any = ""
    starMask: Any = ""
    A: List[float] = field(default_factory=lambda: [-99., -99.])
    As: List[float] = field(default_factory=lambda: [-99., -99.])
    As90: List[float] = field(default_factory=lambda: [-99., -99.])
    rmax: float = -99
    apix: Tuple[float] = (-99., -99.)
    sky: float = -99.
    sky_err: float = 99.
    fwhms: List[float] = field(default_factory=lambda: [-99., -99.])
    theta: float = -99.
    sersic_amplitude: float = -99.
    sersic_r_eff: float = -99.
    sersic_n: float = -99.
    sersic_x_0: float = -99.
    sersic_y_0: float = -99.
    sersic_ellip: float = -99.
    sersic_theta: float = -99.
    time: float = 0.
    star_flag: bool = False

    def write(self, objectfile):

        objectfile.writerow([f"{self.file}", f"{self.apix}", f"{self.rmax}",
                             f"{self.sky}", f"{self.sky_err}", f"{self.A[0]}",
                             f"{self.A[1]}", f"{self.As[0]}", f"{self.As90[0]}",
                             f"{self.fwhms}", f"{self.theta}",
                             f"{self.sersic_amplitude}", f"{self.sersic_r_eff}",
                             f"{self.sersic_n}", f"{self.sersic_x_0}",
                             f"{self.sersic_y_0}", f"{self.sersic_ellip}",
                             f"{self.sersic_theta}", f"{self.time}",
                             f"{self.star_flag}"])


class _Error(Exception):
    """Base class for other exceptions"""
    pass


class _ImageSizeException(_Error):
    '''Image not square'''
    def __init__(self, value):
        print(f"Image from {value}, must be square!")
        raise AttributeError


class _WrongCmdLineArguments(_Error):
    '''Wrong, conflicting, or missing cmd line arguments'''
    def __init__(self, value):
        print(f"{value}")
        sys.exit()


def checkFile(filename):

    with warnings.catch_warnings():
        # ignore invalid card warnings
        warnings.simplefilter('ignore', category=AstropyWarning)

        img, header = fits.getdata(filename, header=True)
    # The following is required as fits files are big endian and skimage
    # assumes little endian. https://stackoverflow.com/a/30284033/6106938
    # https://en.wikipedia.org/wiki/Endianness
    img = img.byteswap().newbyteorder()
    if img.shape[0] != img.shape[1]:
        raise _ImageSizeException(filename)

    imgsize = img.shape[0]

    return img, header, imgsize


def getFiles(imgSource, file=None, folder=None):
    '''Function to get files for analysis

    Parameters
    ----------
    imageSource : str
        Source of image, i.e which telescope took the image
    file : str, optional
        string that contains location of file to be analysed

    folder : str, optional
        string that contains location of folder of files to be analysed

    Return
    ------

    Returns a generator which iterates over the image files.

    '''

    if folder:
        # Get all relevant files in folder
        return Path(folder).glob(f"{imgSource}cutout*.fits")
    else:
        # just single file so place in a generator manually
        return (Path(file) for i in range(1))


def getLocation(file=None, folder=None):
    '''Function to determine the outfolder, and current folder of file/set
       of files provided via command line

    Parameters
    ----------
    file : str, optional
        String that points to file to be analysed
    folder : str, optional
        String that points to folder of files to be analysed

    Returns
    -------
    outfolder: Tuple(Path, Path)
        path to folder where data from analysis will be saved.
    '''

    if file and folder:
        raise _WrongCmdLineArguments("Script cant use both folder and file input!!")

    if not file and not folder:
        raise _WrongCmdLineArguments("Script needs input images to work!!")

    if folder:
        curfolder = Path(folder)
        outfolder = curfolder.parents[0] / "output"
    else:
        curfolder = Path(file).parents[0]
        outfolder = curfolder / "output"

    if not outfolder.exists():
        outfolder.mkdir()

    return curfolder, outfolder


def calcMorphology(files, outfolder, asymmetry=False, shapeAsymmetry=False,
                   allAsymmetry=True, calculateSersic=False, savePixelMap=True,
                   saveCleanImage=True, imageSource=None, catalogue=None,
                   largeImage=False, paramsaveFile="parameters.csv",
                   occludedSaveFile="occluded-object-locations.csv"):
    '''
    Calculates various morphological parameters of galaxies from an image.

    Parameters
    ----------

    files : List[str] or List[Pathobjects] or generator object
        files to iterate over
    outfolder : Path object or str
        path to folder where data from analysis will be saved
    asymmetry : bool, optional
        Default false. If true calculates asymmetry value
    shapeAsymmetry : bool, optional
        Default false. If true calculates shape asymmetry value
    allAsymmetry : bool, optional
        Default True. If true calculates all asymmetry values.
    imageSource : str
        Contains the source of the image, i.e which telescope the image was
        captured by
    catalogue : str or Path, optional
        Catalogue of objects nearby object of interest. Is used to mask out
        objects that interfere with object of interest
    largeImage : bool, optional
        Default False. If true, a larger image is used to calculate the sky
        background
    paramsaveFile: str or Path object
        Name of file where calculated files are to be written
    occludedSaveFile: str or Path object
        Name of file where objects that occlude the object of interest are saved

    Returns
    -------

    results : Result data class
        Container of all calculated results

    '''

    outfile = outfolder / paramsaveFile
    csvfile = open(outfile, mode="w")
    paramwriter = csv.writer(csvfile, delimiter=",")
    paramwriter.writerow(["file", "apix", "r_max", "sky", "sky_err", "A", "Abgr",
                          "As", "As90", "fwhms", "theta", "sersic_amplitude",
                          "sersic_r_eff", "sersic_n", "sersic_x_0", "sersic_y_0",
                          "sersic_ellip", "sersic_theta", "time", "star_flag"])

    if catalogue:
        outfile = outfolder / occludedSaveFile
        objcsvfile = open(outfile, mode="w")
        objwriter = csv.writer(objcsvfile, delimiter=",")
        objwriter.writerow(["file", "ra", "dec", "type"])

    results = []

    for file in files:

        print(file)
        try:
            img, header, imgsize = checkFile(file)
        except IOError:
            print(f"File {file}, does not exist!")
            continue
        except AttributeError as e:
            continue

        # convert image data type to float64 so that later calculations do not
        # raise exceptions
        img = img.astype(np.float64)

        if catalogue is None:
            occludedSaveFile = ""

        newResult = Result(file, outfolder, occludedSaveFile)

        s = time.time()

        # get sky background value and error
        try:
            newResult.sky, newResult.sky_err, newResult.fwhms, newResult.theta = skybgr(img, imgsize, file, largeImage, imageSource)
        except AttributeError:
            # TODO can fail silently if some other attribute error is raised!
            newResult.write(paramwriter)
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')
            print(" ")
            continue

        tmpmask = pixelmap(img, newResult.sky + newResult.sky_err, 3)
        objlist = []
        if catalogue:
            newResult.star_flag, objlist = objectOccluded(tmpmask, file.name, catalogue, header)
            if newResult.star_flag:
                for i, obj in enumerate(objlist):
                    if i == 0:
                        objwriter.writerow([f"{file}", obj[0], obj[1], obj[2]])
                    else:
                        objwriter.writerow(["", obj[0], obj[1], obj[2]])

        starMask = maskstarsPSF(img, objlist, header, newResult.sky)
        newResult.starMask = starMask
        mask = pixelmap(img, newResult.sky + newResult.sky_err, 3, starMask)

        img -= newResult.sky

        # clean image of external sources
        img = maskstarsSEG(img)

        if saveCleanImage:
            filename = file.name
            filename = "clean_" + filename
            outfile = outfolder / filename
            newResult.cleanImage = outfile
            hdu = fits.PrimaryHDU(data=img, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        if savePixelMap:
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            newResult.pixelMapFile = outfile
            hdu = fits.PrimaryHDU(data=mask, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        objectpix = np.nonzero(mask == 1)
        cenpix = np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

        distarray = distarr(imgsize, imgsize, cenpix)
        objectdist = distarray[objectpix]
        newResult.rmax = np.max(objectdist)
        aperturepixmap = aperpixmap(imgsize, newResult.rmax, 9, 0.1)

        newResult.apix = minapix(img, mask, aperturepixmap, starMask)
        angle = 180.

        if asymmetry or allAsymmetry:
            newResult.A = calcA(img, mask, aperturepixmap, newResult.apix, angle, starMask, noisecorrect=True)

        if shapeAsymmetry or allAsymmetry:
            newResult.As = calcA(mask, mask, aperturepixmap, newResult.apix, angle, starMask)
            newResult.As90 = calcA(mask, mask, aperturepixmap, newResult.apix, 90., starMask)

        if calculateSersic:
            p = fitSersic(img, newResult.apix, newResult.fwhms, newResult.theta)
            newResult.sersic_amplitude = p.amplitude.value
            newResult.sersic_r_eff = p.r_eff.value
            newResult.sersic_ellip = p.ellip.value
            newResult.sersic_n = p.n.value
            newResult.sersic_theta = p.theta.value
            newResult.sersic_x_0 = p.x_0.value
            newResult.sersic_y_0 = p.y_0.value

        f = time.time()
        timetaken = f - s
        newResult.time = timetaken
        newResult.write(paramwriter)
        results.append(newResult)

    if catalogue:
        objcsvfile.close()
    csvfile.close()
    return results
