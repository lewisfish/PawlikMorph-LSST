from pathlib import Path
from typing import List, Union

from astropy.nddata import PartialOverlapError

__all__ = ["getLocation", "analyseImage", "getFiles", "getFilesLSST"]


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
        segmap = pixelmap(img, skybgr + skybgr_err, 3)
        img -= skybgr
        A, As, As90 = calculateAsymmetries(img, segmap)
        C, S, gini, m20 = calculateCSGM(img, segmap, skybgr)

    except (AttributeError, RuntimeError, MemoryError) as e:
        # if sky background estimation fails return array of -99
        return [-99 for i in range(0, 8)]
    return [A, As, As90, C, S, gini, m20, filename, ra, dec]


def getFiles(file: str):
    from astropy.table import Table

    '''Function to read CSV file, and return a generator object which yields
       filenames, and RA DECs.

    Parameters
    ----------

    file : str
        String that filename of the CSV file to be read.

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


def getFilesLSST(file: str, folder: str):
    from astropy.table import Table

    '''Function to read CSV file, and return a generator object which yields
       filenames, and RA DECs, camcol, run, and field.

    Parameters
    ----------

    file : str
        String that filename of the CSV file to be read.

    Return
    ------

    Returns a generator which iterates over the image files.

    '''

    data = Table.read(file)

    ras = data["ra"]
    decs = data["dec"]
    camCols = data["camCol"]
    runs = data["run"]
    fields = data["field"]

    for ra, dec, camCol, run, field in zip(ras, decs, camCols, runs, fields):
        yield folder, ra, dec, camCol, run, field


def getLocation(folder):
    '''Helper function that returns a Path object containg the output folder

    Parameters
    ----------

    folder : str
        Path to a folder for output files.

    Returns
    -------
    outfolder: Path object
        Path to folder where data from analysis will be saved.
    '''

    if folder:
        outfolder = Path(folder)
    else:
        curfolder = Path().cwd()
        outfolder = curfolder / "output"

    if not outfolder.exists():
        outfolder.mkdir()

    return outfolder
