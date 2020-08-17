"""
This package can be extended by sub classing Image and implementing the required
methods, and _IMAGE_TYPE.
"""

from abc import ABC, abstractmethod
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy import wcs
from astropy import units
from astropy.utils.exceptions import AstropyWarning
import numpy as np

try:
    import lsst.daf.persistence as dafPersist
    import lsst.afw.geom as afwGeom
    import lsst.afw.image as afwImage
    lsst = True
except ImportError:
    lsst = None


__all__ = ["Image", "readImage", "sdssImage", "lsstImage"]

# ignore invalid card warnings when reading FITS files
warnings.simplefilter('ignore', category=AstropyWarning)


def readImage(filename: str, ra: float, dec: float, npix=128, header=False):
    """Helper function that can be used to read images directly without need
        to manually create Image class.

    Parameters
    ----------

    filename : sty
        location of image to read

    ra : float
        RA, right ascension of object of interest in image

    dec : float
        DEC, declination of object of interest in image

    npix: int, optional
        Size of cutout to return from larger image. Default is 128

    header: bool, optional
        If true return header information as well. Default is False

    Returns
    -------

    img : np.ndarray, 2D, float
        Cutout image.
    If header=True then also returns the header from the FITS file.

    """

    imgObj = Image("SDSS", filename=filename)
    imgObj.setView(ra=ra, dec=dec, npix=npix)
    img = imgObj.getImage()
    if header:
        header = imgObj.getHeader()
        return img, header
    else:
        return img


class Image(ABC):
    """Abstract base class for images"""
    _IMAGE_TYPE = ""

    def __init__(self, filename=None):
        super(Image, self).__init__()

    @abstractmethod
    def getImage(self):
        pass

    @abstractmethod
    def getHeader(self):
        pass

    @abstractmethod
    def _make_cutout(self, ra, dec, npix):
        pass

    def __new__(cls, _IMAGE_TYPE, **kwargs):
        subclass_map = {subclass._IMAGE_TYPE: subclass for subclass in cls.__subclasses__()}
        try:
            subclass = subclass_map[_IMAGE_TYPE]
        except KeyError as e:
            raise ValueError("Invalid image class!") from e
        instance = super(Image, subclass).__new__(subclass)
        return instance


class sdssImage(Image):
    """Class for SDSS images, as ingested by standard 'method'"""
    _IMAGE_TYPE = "SDSS"

    def __init__(self, *args, **kwargs):
        super(sdssImage, self).__init__()
        self.filename = kwargs["filename"]
        self.image = None

    def setView(self, ra=None, dec=None, npix=128):
        """ Get the correct view in larger image, and create the cutout on the
            correct view

        Parameters
        ----------

        ra: float
            RA position

        dec: float
            DEC position

        npix : int
            Size to make cutout

        Returns
        -------

        """

        img, self.header = fits.getdata(self.filename, header=True)

        self.largeImage = img.byteswap().newbyteorder()
        if self.largeImage.shape[0] >= npix and ra and dec:
            self.cutout = self._make_cutout(ra, dec, npix)
        else:
            self.cutout = self.largeImage

    def getImage(self) -> np.ndarray:
        """ Returns the cutout image

        Parameters
        ----------

        Returns
        -------

        self.image: np.ndarray, float (2D)
            The cutout image centered on the view provided in setview.

        """

        self.image = self.cutout
        self.image = self.image.astype(np.float64)

        return self.image

    def getHeader(self) -> fits.Header:
        """ Returns the image header

        Parameters
        ----------

        Returns
        -------

        self.header: astropy.io.fits.Header
            The header for the cutout.

        """

        return self.header

    def _make_cutout(self, ra, dec, npix):
        w = wcs.WCS(self.header)
        position = SkyCoord(ra*units.deg, dec*units.deg)
        position = wcs.utils.skycoord_to_pixel(position, wcs=w)
        # There is a bug in Astropy that does not deal with reversed ctype
        # headers in FITS files.
        # see https://github.com/astropy/astropy/issues/10468
        if w.wcs.ctype[0][0] == "D":
            position = position[::-1]
        stamp = Cutout2D(self.largeImage.data, position=position, size=(npix, npix), wcs=w, mode="strict")

        return stamp.data


class lsstImage(Image):
    """Class for SDSS images ingested via LSST dataButler.

        Some metadata not available, this includes wcs, and pixel value
        conversion information (bscale, bzero etc).
        Shouldn't really recreate butler on each image call...

        This code is far from the optimal way to read images.
        https://github.com/LSSTScienceCollaborations/StackClub

        The above source maybe of help for future developer.

    """

    _IMAGE_TYPE = "LSST"

    def __init__(self, *args, **kwargs):
        super(lsstImage, self).__init__()
        if lsst is None:
            raise ImportError("LSST stack not installed!")
        self.butler = dafPersist.Butler(str(kwargs["filename"]))
        self.image = None

    def setView(self, ra, dec, run, camCol, field, filter="r", npix=128):
        self.largeImage = self.butler.get("fpC", run=run, camcol=camCol, field=field, filter=filter)
        self.cutout = self._make_cutout(ra, dec, npix)

    def getImage(self):
        self.image = self.cutout.image.array
        return self.image

    def getHeader(self):
        self.header = self.cutout.getMetadata()
        return self.header

    def _make_cutout(self, ra, dec, npix):
        # Point2I, Extent2I, and Box2I are going to be deprecated in a future version...
        lsstwcs = self.largeImage.getWcs()
        radec = afwGeom.SpherePoint(ra, dec, afwGeom.degrees)
        x, y = afwGeom.PointI(lsstwcs.skyToPixel(radec))
        corner = afwGeom.Point2I(int(x - npix // 2), int(y - npix // 2))
        bbox = afwGeom.Box2I(afwGeom.Point2I(corner), afwGeom.Extent2I(npix, npix))
        stamp = self.largeImage.Factory(self.largeImage, bbox, afwImage.PARENT, False)

        return stamp
