from pathlib import Path
from typing import List, Union

from astropy.nddata import PartialOverlapError

__all__ = ["getLocation", "analyseImage", "getFiles", "getFilesLSST"]




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
