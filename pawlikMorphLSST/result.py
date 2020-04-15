from dataclasses import dataclass, field
from typing import List, Tuple, Any


@dataclass
class Result:
    '''Class that stores the results of image analysis'''

    # names of files and folders used or generated in analysis
    file: str
    outfolder: Any
    occludedFile: str
    pixelMapFile: Any = ""
    cleanImage: Any = ""
    starMask: Any = ""
    objList: Any = field(default_factory=lambda: [])
    # Calculated asymmetry values
    A: List[float] = field(default_factory=lambda: [-99., -99.])
    As: List[float] = field(default_factory=lambda: [-99., -99.])
    As90: List[float] = field(default_factory=lambda: [-99., -99.])
    # Misc calculated values
    rmax: float = -99
    apix: Tuple[float] = (-99., -99.)
    sky: float = -99.
    sky_err: float = 99.
    fwhms: List[float] = field(default_factory=lambda: [-99., -99.])
    theta: float = -99.
    # CAS
    r20: float = -99.
    r80: float = -99.
    C: float = -99.    # Concentration
    gini: float = -99.
    m20: float = -99.
    S: float = -99.    # clumpiness/smoothness
    # Sersic fit values
    sersic_amplitude: float = -99.
    sersic_r_eff: float = -99.
    sersic_n: float = -99.
    sersic_x_0: float = -99.
    sersic_y_0: float = -99.
    sersic_ellip: float = -99.
    sersic_theta: float = -99.
    time: float = 0.
    # Flags that signify data may not be useful
    star_flag: bool = False
    maskedPixelFraction: float = -99.
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
