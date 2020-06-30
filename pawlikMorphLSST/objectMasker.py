import sys
from typing import Tuple, List
import warnings

import numpy as np
import pandas as pd

from astropy import wcs
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning

__all__ = ["objectOccluded"]


def objectOccluded(mask: np.ndarray, radec: Tuple[float, float],
                   catalogue: str, header, galaxy=False, cosmicray=False,
                   unknown=False) -> Tuple[bool, List[float]]:

    '''Function gets list of objects near the object of interest, and
       determines if that objects light occludeds the object of interest light.

    Parameters
    ----------

    mask: np.ndarray
        Object mask

    radec: Tuple[float, float]
        Tuple of ra, dec

    catalogue: str
        Name of object catalogue to check against
        Expected format is objID: float, ra: float, dec: float, type: str

    header: ?
        fits image header

    galaxy: bool, optional
        Option to include galaxy objects

    cosmicray: bool, optional
        Option to include cosmic rays

    unknown: bool, optional
        Option to include unknown objects

    Returns
    -------
    Tuple[bool, List[float]]
        Returns true alongside list of objects that occlude object mask.
        Otherwise returns false and an empty list
    '''

    ra, dec = radec

    listofobjs = _findStars(catalogue, ra, dec, galaxy, cosmicray, unknown)
    listofOccludedObjs = []

    with warnings.catch_warnings():
        # ignore invalid card warnings
        warnings.simplefilter('ignore', category=AstropyWarning)
        w = wcs.WCS(header)

    for obj in listofobjs:
        pos = SkyCoord(obj[0]*units.deg, obj[1]*units.deg)
        pos = wcs.utils.skycoord_to_pixel(pos, wcs=w)
        # check for occlusion
        bools = np.nonzero(mask[int(pos[1])-1:int(pos[1])+1, int(pos[0])-1:int(pos[0])+1] == 1)
        if np.any(bools):
            listofOccludedObjs.append(obj)

    if len(listofOccludedObjs) > 0:
        return True, listofOccludedObjs
    return False, []


def _findStars(catalogue: str, ra: float, dec: float,
               galaxy: bool, cosmicray: bool, unknown: bool,
               size=0.008333333333333333) -> List[float]:

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
        size in degrees of box half width. DEfault is 0.5 arcmins

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
    df = pd.read_csv(catalogue, dtype={"objID": "float", "ra": "float", "dec": "float", "psfMag_r": "float", "type": "str"})

    am = size

    # get bounding box of image
    xmin = ra - am
    xmax = ra + am
    ymin = dec - am
    ymax = dec + am

    # split up objects
    try:
        stars = df.loc[df['type'] == "STAR"]
        ras = list(stars["ra"])
        decs = list(stars["dec"])
        psfMags = list(stars["psfMag_r"])
    except KeyError:
        print("Supplied catalogue has wrong format. Expected: objId, RA, DEC, psfMag_r, type")
        sys.exit()
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

    objs = _getObject(ras, decs, psfMags, types, xmin, xmax, ymin, ymax)
    return objs


def _getObject(ra: List[float], dec: List[float], psfMags: List[float],
               types: List[str], xmin: float, xmax: float,
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
    for i, j, k, p in zip(ra, dec, types, psfMags):
        if i > xmin and i < xmax:
            if j > ymin and j < ymax:
                pos.append([i, j, k, p])

    return pos
