import sys
import warnings
from pathlib import Path

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

__all__ = ["checkFile", "getLocation"]


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
    '''Function that checks file exists, is right size, and right source

    Parameters
    ----------

    filename : str or Path object
        Name of file to check

    Returns
    -------

    img : np.ndarray
        image data

    header : object
        header from fits file

    imgsize : int
        Size of image in 1 dimension. Image is square

    '''

    with warnings.catch_warnings():
        # ignore invalid card warnings
        warnings.simplefilter('ignore', category=AstropyWarning)
        img, header = fits.getdata(filename, header=True)

    # The following is required as fits files are big endian and skimage
    # assumes little endian. https://stackoverflow.com/a/30284033/6106938
    # https://en.wikipedia.org/wiki/Endianness
    img = img.byteswap().newbyteorder()

    # check image is square
    # TODO add support of rectangular images?
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
        if folder.find(".fits") != -1:
            raise _WrongCmdLineArguments("Specified folder option but provided single file!!!")

        # Get all relevant files in folder
        if imgSource == "none":
            return Path(folder).glob(f"cutout*.fits")
        else:
            return Path(folder).glob(f"{imgSource}cutout*.fits")
    else:
        # just single file so place in a generator manually
        return (Path(file) for i in range(1))


def getLocation(file=None, folder=None):
    '''Determines the output folder and current folder of files.

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
