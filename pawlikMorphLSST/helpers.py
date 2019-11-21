from pathlib import Path as _Path
import sys as _sys

from astropy.io import fits as _fits
import numpy as _np
import glob as gb
from .imageutils import cleanimg as _cleanimg
from .imageutils import skybgr as _skybgr
from .pixmap import pixelmap as _pixelmap

__all__ = ["checkFile", "getLocation", "prepareimage"]


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
        _sys.exit()


def checkFile(filename):

    img, header = _fits.getdata(filename, header=True)
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
        return _Path(args.folder).glob(f"{args.imgsource}cutout*.fits")
    else:
        # just single file so place in a generator manually
        return (_Path(args.file) for i in range(1))


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
        curfolder = _Path(args.folder)
        outfolder = curfolder.parents[0] / "output"
    else:
        curfolder = _Path(args.file).parents[0]
        outfolder = curfolder / "output"

    if not outfolder.exists():
        outfolder.mkdir()

    return curfolder, outfolder


def prepareimage(img: _np.ndarray):

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

    sky, sky_err, flag = _skybgr(img, img.shape[0])
    mask = _pixelmap(img, sky + sky_err, 3)
    img -= sky

    img = _cleanimg(img, mask)

    return img, mask
