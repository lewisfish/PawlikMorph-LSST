from abc import ABC, abstractmethod
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy import units
from astropy.utils.exceptions import AstropyWarning

try:
    import lsst.daf.persistence as dafPersist
    import lsst.afw.geom as afwGeom
    import lsst.afw.image as afwImage
except ImportError:
    lsst = None


__all__ = ["Image", "readImage"]


def readImage(imgType: str, filename: str, ra: float, dec: float, header=False):
    imgObj = Image(imgType, filename=filename)
    imgObj.setView(ra=ra, dec=dec)
    img = imgObj.getImage()
    if header:
        header = imgObj.getHeader()
        return img, header
    else:
        return img


class Image(ABC):
    """Abstract base class for images"""
    def __init__(self):
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

    # TODO
    # new function that gets the correct implmentationn based upoin user passed string
    # probably needs exception handling...
    def __new__(cls, _IMAGE_TYPE, **kwargs):
        subclass_map = {subclass._IMAGE_TYPE: subclass for subclass in cls.__subclasses__()}
        subclass = subclass_map[_IMAGE_TYPE]
        instance = super(Image, subclass).__new__(subclass)
        return instance


class sdssImage(Image):
    """Class for SDSS images, as ingested by standard 'method'"""
    _IMAGE_TYPE = "sdss"

    def __init__(self, *args, **kwargs):
        super(sdssImage, self).__init__()
        self.filename = kwargs["filename"]
        self.image = None

    def setView(self, ra=None, dec=None, npix=128):
        with warnings.catch_warnings():
            # ignore invalid card warnings
            warnings.simplefilter('ignore', category=AstropyWarning)
            img, self.header = fits.getdata(self.filename, header=True)

        self.largeImage = img.byteswap().newbyteorder()
        if self.largeImage.shape[0] >= npix and ra and dec:
            self.cutout = self._make_cutout(ra, dec, npix)
        else:
            self.cutout = self.largeImage

    def getImage(self):
        self.image = self.cutout
        return self.image

    def getHeader(self):
        return self.header

    def _make_cutout(self, ra, dec, npix):
        wcs = WCS(self.header)
        position = SkyCoord(ra*units.deg, dec*units.deg)
        stamp = Cutout2D(self.largeImage.data, position=position, size=(npix, npix), wcs=wcs, mode="strict")

        return stamp.data


class lsstImage(Image):
    """Class for SDSS images ingested via LSST dataButler
        some metadata not available
        this includes wcs, and pixel value conversion information (bscal, bzero etc)
    """
    _IMAGE_TYPE = "lsst"

    def __init__(self, *args, **kwargs):
        super(lsstImage, self).__init__()
        if lsst is None:
            raise ImportError("LSST stack not installed!")
        self.butler = dafPersist.Butler(kwargs["filename"])
        self.image = None

    def setView(self, ra, dec, run, camCol, field, filter, npix=128):
        self.largeImage = self.butler.get("fpC", run=run, camcol=camCol, field=field, filter=filter)
        self.cutout = self._make_cutout(ra, dec, npix)

    def getImage(self):
        self.image = self.cutout.image.array
        return self.image

    def getHeader(self):
        self.header = self.cutout.getMetadata()
        return self.header

    def _make_cutout(self, ra, dec, npix):
        lsstwcs = self.largeImage.getWcs()
        radec = afwGeom.SpherePoint(ra, dec, afwGeom.degrees)
        x, y = afwGeom.PointI(lsstwcs.skyToPixel(radec))
        corner = afwGeom.Point2I(int(x - npix // 2), int(y - npix // 2))
        bbox = afwGeom.Box2I(afwGeom.Point2I(corner), afwGeom.Extent2I(npix, npix))
        stamp = self.largeImage.Factory(self.largeImage, bbox, afwImage.PARENT, False)

        return stamp
