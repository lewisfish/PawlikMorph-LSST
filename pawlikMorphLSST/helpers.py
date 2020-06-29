from pathlib import Path
import sys
from typing import List, Union
import warnings

from astropy.io import fits
from astropy.nddata import PartialOverlapError
from astropy.utils.exceptions import AstropyWarning

__all__ = ["getLocation", "analyseImage"]


def analyseImage(info: List[Union[float, str]], *args) -> List[Union[float, str]]:
    """Helper function that calculates CASGM including As and AS90

    Parameters
    ----------

    info : List[str, float, float]
        List of filename, RA, and DEC

    Returns
    -------

    Tuple[float, float, float, float, float, float, float, str, float, float]
        A, As, AS90, C, S, gini, M20, filename , RA, DEC

    """

    from .asymmetry import calculateAsymmetries
    from .casgm import calculateCSGM
    from .Image import readImage
    from .imageutils import maskstarsSEG
    from .pixmap import pixelmap
    from .skyBackground import skybgr

    filename = info[0]
    ra, dec = info[1], info[2]

    try:
        img = readImage(filename, ra, dec)
    except (AttributeError, PartialOverlapError) as e:
        # if image load fails return array of -99
        return [-99 for i in range(0, 8)]

    # preprocess image
    img = maskstarsSEG(img)

    # estimate skybackground
    try:
        skybgr, skybgr_err, *_ = skybgr(img)

        # create image where the only bright pixels are the pixels that belong to the galaxy
        mask = pixelmap(img, skybgr + skybgr_err, 3)
        img -= skybgr
        A, As, As90 = calculateAsymmetries(img, mask)
        C, S, gini, m20 = calculateCSGM(img, mask, skybgr)

    except (AttributeError, RuntimeError, MemoryError) as e:
        # if sky background estimation fails return array of -99
        return [-99 for i in range(0, 8)]
    return [A, As, As90, C, S, gini, m20, filename, ra, dec]


def getFiles(file):
    from astropy.table import Table

    '''Function to read csv file, and return a generator object which yields filenames, and ra decs.

    Parameters
    ----------

    file : str, optional
        string that contains location of file to be analysed

    Return
    ------

    Returns a generator which iterates over the image files.

    '''

    data = Table.read(file)
    filenames = data["filename"]
    ras = data["ra"]
    decs = data["dec"]

    for filename, ra, dec in zip(filenames, ras, decs):
        yield filename, ra, dec


def getLocation():
    '''Determines the output folder and current folder of files.

    Parameters
    ----------

    Returns
    -------
    outfolder: Tuple(Path, Path)
        Path to folder where data from analysis will be saved.
    '''

    curfolder = Path().cwd()
    outfolder = curfolder / "output"

    if not outfolder.exists():
        outfolder.mkdir()

    return outfolder
