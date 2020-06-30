from dataclasses import dataclass, field
from typing import List, Tuple, Any

__all__ = ["Result"]


@dataclass
class Result:
    '''Data class that stores the results of image analysis.'''

    #: Filename of image
    file: str
    #: Output folder for saving data
    outfolder: Any
    #: Filename of output data for objects that occlude with segmentation map.
    occludedFile: str
    #: Path to segmentation map
    pixelMapFile: Any = ""
    #: path to clean image.
    cleanImage: Any = ""
    #: path to star mask
    starMask: Any = ""
    #: List of objects RA, DECs that occlude objects segmentation map.
    objList: Any = field(default_factory=lambda: [])
    #: Calculated asymmetry value, format [A, A_error]
    A: List[float] = field(default_factory=lambda: [-99., -99.])
    #: Calculated shape asymmetry value, format [As, As_error]
    As: List[float] = field(default_factory=lambda: [-99., -99.])
    #: Calculated shape asymmetry 90 value, format [As90, As90_error]
    As90: List[float] = field(default_factory=lambda: [-99., -99.])
    #: Maxmimum radius of the segmentation map
    rmax: float = -99
    #: Asymmetry (A) minimised central pixel
    apix: Tuple[float] = (-99., -99.)
    #: Sky background value.
    sky: float = -99.
    #: Sky background error.
    sky_err: float = 99.
    #: FWHM's of the fitted 2D Gaussian
    fwhms: List[float] = field(default_factory=lambda: [-99., -99.])
    #: Theta of the fitted 2D Gaussian
    theta: float = -99.
    #: Radius in which 20% of total light flux is contained
    r20: float = -99.
    #: Radius in which 80% of total light flux is contained
    r80: float = -99.
    #: Concentraion value
    C: float = -99.
    #: Gini index
    gini: float = -99.
    #: M20 value
    m20: float = -99.
    #: Smoothness value.
    S: float = -99.
    #: Sersic amplitude.
    sersic_amplitude: float = -99.
    #: Sersic effective radius
    sersic_r_eff: float = -99.
    #: Sersic index.
    sersic_n: float = -99.
    #: Sersic x centre
    sersic_x_0: float = -99.
    #: Sersic y centre
    sersic_y_0: float = -99.
    #: Sersic ellipticity.
    sersic_ellip: float = -99.
    #: Sersic rotation.
    sersic_theta: float = -99.
    #: Time taken to analyse image.
    time: float = 0.
    #: If true means that there is a star in the catalogue occluding the objects segmentation map
    star_flag: bool = False
    #: Fraction of pixels masked due to occluding star/object.
    maskedPixelFraction: float = -99.
    #: If true then the segmentation map extends to an edge of the image.
    objectEdge: bool = False

    def write(self, objectfile):
        '''Write out result as a row to a csv file'''

        objectfile.writerow([f"{self.file}", f"{self.apix}", f"{self.rmax}",
                             f"{self.r20}", f"{self.r80}",
                             f"{self.sky}", f"{self.sky_err}", f"{self.C}",
                             f"{self.A[0]}", f"{self.A[1]}", f"{self.S}",
                             f"{self.As[0]}", f"{self.As90[0]}", f"{self.gini}",
                             f"{self.m20}", f"{self.fwhms}", f"{self.theta}",
                             f"{self.sersic_amplitude}", f"{self.sersic_r_eff}",
                             f"{self.sersic_n}", f"{self.sersic_x_0}",
                             f"{self.sersic_y_0}", f"{self.sersic_ellip}",
                             f"{self.sersic_theta}", f"{self.time}",
                             f"{self.star_flag}", f"{self.maskedPixelFraction}",
                             f"{self.objectEdge}"])
