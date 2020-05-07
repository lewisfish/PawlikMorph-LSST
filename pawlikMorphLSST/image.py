from abc import ABC, abstractmethod

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy import units

try:
    import lsst.daf.persistence as dafPersist
    import lsst.afw.geom as afwGeom
    import lsst.afw.image as afwImage
except ImportError:
    lsst = None


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


class sdssImage(Image):
    """Class for SDSS images, as ingested by standard 'method'"""
    def __init__(self, filename):
        super(sdssImage, self).__init__()
        self.filename = filename
        self.image = None

    def setView(self, ra, dec, npix):
        img, self.header = fits.getdata(self.filename, header=True)
        self.largeImage = img.byteswap().newbyteorder()
        self.cutout = self._make_cutout(ra, dec, npix)

    def getImage(self):
        self.image = self.cutout
        return self.image

    def getHeader(self):
        return self.header

    def _make_cutout(self, ra, dec, npix):
        wcs = WCS(self.header)
        position = SkyCoord(ra*units.deg, dec*units.deg)
        stamp = Cutout2D(self.largeImage.data, position=position, size=(npix, npix), wcs=wcs)

        return stamp.data


class lsstImage(Image):
    """Class for SDSS images ingested via LSST dataButler
        some metadata not available
        this includes wcs, and pixel value conversion information (bscal, bzero etc)
    """
    def __init__(self, filename):
        super(lsstImage, self).__init__()
        self.butler = dafPersist.Butler(filename)
        self.image = None
        if lsst is none:
            raise ImportError:
                print("LSST stack not installed!")

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
