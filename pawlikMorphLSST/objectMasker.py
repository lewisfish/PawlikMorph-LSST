import re
from typing import Tuple, List

import pandas as _pd
from astropy import wcs as _wcs
from astropy.coordinates import SkyCoord as _SkyCoord
from astropy import units as _units
import numpy as _np

__all__ = ["objectOccluded"]


def objectOccluded(mask: _np.ndarray, name: str, catalogue: str, wcs,
                   galaxy=False, cosmicray=False, unknown=False) -> Tuple[bool, List[float]]:

    '''

    Parameters
    ----------
    mask: np.ndarray
        Object mask
    name: str
        Name of file to get ra, dec from
    catalogue: str
        Name of object catalogue to check against
    wcs: WCSobject
        WCS information from fits image header
    galaxy: bool, optional
        Option to include galaxy objects
    cosmicray: bool, optional
        Option to include cosmic rays
    unknown: bool, optional
        Option to include unkown objects

    Returns
    -------
    Tuple[bool, List[float]]
        Returns true alongside list of objects that occulde object mask.
        Otherwise returns flase and an empty list
    '''

    ra = float(re.search(r"\d{2,3}\.\d{3,}", name).group())
    dec = float(re.search(r"[\+\-]\d{,3}\.\d{,10}", name).group())

    listofobjs = findStars(catalogue, ra, dec, galaxy, cosmicray, unknown)
    listofOccludedObjs = []

    for obj in listofobjs:
        pos = _SkyCoord(obj[0]*_units.deg, obj[1]*_units.deg)
        pos = _wcs.utils.skycoord_to_pixel(pos, wcs=wcs)
        bools = _np.nonzero(mask[int(pos[1])-1:int(pos[1])+1, int(pos[0])-1:int(pos[0])+1] == 1)
        if _np.any(bools):
            listofOccludedObjs.append(obj)

    if len(listofOccludedObjs) > 0:
        return True, listofOccludedObjs
    return False, []


def findStars(catalogue: str, ra: float, dec: float,
              galaxy: bool, cosmicray: bool, unknown: bool,
              size=0.008333333333333333,) -> List[float]:

    '''Function to find postion of stars from a catalogue,
       within a square box of size around a given object.

    Parameters
    ----------

    catalogue: str
        filename of catalogue to be searched.
    ra: float
        RA of object.
    dec: float
        DEC of object.
    size: float, optional
        size in degrees of box half width.
    galaxy: bool
        Option to include galaxy objects
    cosmicray: bool
        Option to include cosmic rays
    unknown: bool
        Option to include unkown objects

    Returns
    -------

    objs: List[float]
        list of position of nearby objects.
    '''

    # read in catalogue of nearby objects
    df = _pd.read_csv(catalogue, dtype={"objID": "float", "ra": "float", "dec": "float", "type": "str"})

    am = size

    # get bounding box of image
    xmin = ra - am
    xmax = ra + am
    ymin = dec - am
    ymax = dec + am

    # split up objects
    stars = df.loc[df['type'] == "STAR"]
    ras = list(stars["ra"])
    decs = list(stars["dec"])
    types = ["STAR" for i in range(0, len(ras))]

    if galaxy:
        galaxies = df.loc[df['type'] == "GALAXY"]
        rag = list(galaxies["ra"])
        decg = list(galaxies["dec"])
        ras += rag
        decs += decg
        types += ["GALAXY" for i in range(0, len(ras))]

    if cosmicray:
        cr = df.loc[df['type'] == "COSMIC_RAY"]
        racr = list(cr["ra"])
        deccr = list(cr["dec"])
        ras += racr
        decs += deccr
        types += ["COSMIC_RAY" for i in range(0, len(ras))]

    if unknown:
        unk = df.loc[df['type'] == "UNKNOWN"]
        rau = list(unk["ra"])
        decu = list(unk["dec"])
        ras += rau
        decs += decu
        types += ["UNKNOWN" for i in range(0, len(ras))]

    objs = getObject(ras, decs, types, xmin, xmax, ymin, ymax)
    return objs


def getObject(ra: List[float], dec: List[float], types: List[str], xmin: float, xmax: float,
              ymin: float, ymax: float) -> List[float]:

    '''Function that check that objects is nearby object of interest.

    Parameters
    ----------
    ra: List[float]
        List of RAs of potential nearby objects.
    dec: List[float]
        List of DECs of potential nearby objects.
    types: List[str]
        List of types for potential nearby objects.
    xmin: float
        minimum in x of bounding box (degrees).
    xmax: float
        maximum in x of bounding box (degrees).
    ymin: float
        minimum in y of bounding box (degrees).
    ymax: float
        maximum in y of bounding box (degrees).

    Returns
    -------

    pos: List[float]
        list of position of nearby objects.

    '''

    pos = []
    for i, j, k in zip(ra, dec, types):
        if i > xmin and i < xmax:
            if j > ymin and j < ymax:
                pos.append([i, j, k])

    return pos
