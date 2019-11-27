import csv
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

import numpy as np

from .apertures import aperpixmap
from .apertures import distarr
from .asymmetry import calcA
from .asymmetry import minapix
from .imageutils import maskstarsSEG
from .imageutils import skybgr
from .objectMasker import objectOccluded
from .pixmap import pixelmap

__all__ = ["checkFile", "getLocation", "prepareimage", "calcMorphology"]


@dataclass
class Result:
    '''Class that stores the results of image analysis'''

    file: str
    A: List[float] = field(default_factory=lambda: [-99., -99.])
    As: List[float] = field(default_factory=lambda: [-99., -99.])
    As90: List[float] = field(default_factory=lambda: [-99., -99.])
    rmax: float = -99
    apix: Tuple[float] = (-99., -99.)
    sky: float = -99.
    sky_err: float = 99.
    time: float = 0.
    star_flag: bool = False

    def write(self, objectfile):

        objectfile.writerow([f"{self.file}", f"{self.apix}", f"{self.rmax}",
                             f"{self.sky}", f"{self.sky_err}", f"{self.A[0]}",
                             f"{self.A[1]}", f"{self.As[0]}", f"{self.As90[0]}",
                             f"{self.time}", f"{self.star_flag}"])


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

    img, header = fits.getdata(filename, header=True)
    # The following is required as fits files are big endian and skimage
    # assumes little endian. https://stackoverflow.com/a/30284033/6106938
    # https://en.wikipedia.org/wiki/Endianness
    img = img.byteswap().newbyteorder()
    if img.shape[0] != img.shape[1]:
        raise _ImageSizeException(filename)

    imgsize = img.shape[0]

    return img, header, imgsize


def getFiles(args):
    '''Function to get files for analysis

    Parameters
    ----------

    args: argparseobject
        comand line arguments.

    Return
    ------

    Returns a generator which iterates over the image files.

    '''

    if args.folder:
        return Path(args.folder).glob(f"{args.imgsource}cutout*.fits")
    else:
        # just single file so place in a generator manually
        return (Path(args.file) for i in range(1))


def getLocation(args):
    '''Function to determine the outfolder, and current folder of file/set
       of files provided via commandline

    Parameters
    ----------
    args: argparseobject
        comand line arguments.

    Returns
    -------
    outfolder: Tuple(Path, Path)
        path to folder where data from analysis will be saved.
    '''

    if args.file and args.folder:
        raise _WrongCmdLineArguments("Script cant use both folder and file input!!")

    if not args.file and not args.folder:
        raise _WrongCmdLineArguments("Script needs input images to work!!")

    if args.folder:
        curfolder = Path(args.folder)
        outfolder = curfolder.parents[0] / "output"
    else:
        curfolder = Path(args.file).parents[0]
        outfolder = curfolder / "output"

    if not outfolder.exists():
        outfolder.mkdir()

    return curfolder, outfolder


def calcMorphology(files, outfolder, args, paramsaveFile="parameters.csv",
                   occludedsaveFile="occluded-object-locations.csv"):
    '''
    Calculates various morphological parameters of galaxies from an image.

    Parameters
    ----------

    files: List[str] or List[Pathobjects] or generator object
        files to iterate over
    outfolder: Path object or str
        path to folder where data from analysis will be saved
    args: argpare object
        cmd line arguments
    paramsaveFile: str or Path object
        name of file where calculated files are to be written
    occludedsaveFile: str or Path object
        name of file where objects that occlude the object of interest are saved

    Returns
    -------

    '''

    # suppress warnings about unrecognised keywords
    warnings.simplefilter('ignore', category=AstropyWarning)

    outfile = outfolder / paramsaveFile
    csvfile = open(outfile, mode="w")
    paramwriter = csv.writer(csvfile, delimiter=",")
    paramwriter.writerow(["file", "apix", "r_max", "sky", "sky_err", "A", "Abgr",
                          "As", "As90", "time", "star_flag"])

    if args.catalogue:
        outfile = outfolder / occludedsaveFile
        objcsvfile = open(outfile, mode="w")
        objwriter = csv.writer(objcsvfile, delimiter=",")
        objwriter.writerow(["file", "ra", "dec", "type"])

    for file in files:

        try:
            img, header, imgsize = checkFile(file)
        except IOError:
            print(f"File {file}, does not exist!")
            continue
        except AttributeError as e:
            continue

        # set default values for calculated parameters
        newResult = Result(file)

        s = time.time()

        print(file)

        # get sky background value and error
        try:
            newResult.sky, newResult.sky_err = skybgr(img, imgsize, file, args)
        except AttributeError:
            newResult.write(paramwriter)
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')
            print(" ")
            continue

        mask = pixelmap(img, newResult.sky + newResult.sky_err, 3)

        if args.catalogue:
            newResult.star_flag, objlist = objectOccluded(mask, file.name, args.catalogue, header, galaxy=True, cosmicray=True, unknown=True)
            if newResult.star_flag:
                for i, obj in enumerate(objlist):
                    if i == 0:
                        objwriter.writerow([f"{file}", obj[0], obj[1], obj[2]])
                    else:
                        objwriter.writerow(["", obj[0], obj[1], obj[2]])

        img -= newResult.sky

        # clean image of external sources
        img = maskstarsSEG(img)

        if args.savecleanimg:
            filename = file.name
            filename = "clean_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=img, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        if args.savepixmap:
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=mask, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        objectpix = np.nonzero(mask == 1)
        cenpix = np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

        distarray = distarr(imgsize, imgsize, cenpix)
        objectdist = distarray[objectpix]
        newResult.rmax = np.max(objectdist)
        aperturepixmap = aperpixmap(imgsize, newResult.rmax, 9, 0.1)

        newResult.apix = minapix(img, mask, aperturepixmap)
        angle = 180.

        if args.A or args.Aall:
            newResult.A = calcA(img, mask, aperturepixmap, newResult.apix, angle, noisecorrect=True)

        if args.As or args.Aall:
            newResult.As = calcA(mask, mask, aperturepixmap, newResult.apix, angle)
            newResult.As90 = calcA(mask, mask, aperturepixmap, newResult.apix, 90.)

        f = time.time()
        timetaken = f - s
        newResult.time = timetaken
        newResult.write(paramwriter)

    if args.catalogue:
        objcsvfile.close()
    csvfile.close()


def prepareimage(img: np.ndarray):

    '''Helper function to prepare images

    Parameters
    ----------
        img : np.ndarray
            Image to be prepared.

    Returns
    -------

    img : np.ndarray
        Image which has been bgr subtracted and 'cleaned'.
    mask : np.ndarray
        Binary image of object.

    '''

    if img.shape[0] != img.shape[1]:
        print("ERROR! image not square")
        return

    img = img.byteswap().newbyteorder()

    sky, sky_err, flag = skybgr(img, img.shape[0])
    mask = pixelmap(img, sky + sky_err, 3)
    img -= sky

    img = cleanimg(img, mask)

    return img, mask
