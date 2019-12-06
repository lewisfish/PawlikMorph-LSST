import csv
import time

import numpy as np

from astropy.io import fits

from .apertures import aperpixmap
from .apertures import distarr
from .asymmetry import calcA
from .asymmetry import minapix
from .helpers import checkFile
from .imageutils import maskstarsPSF
from .imageutils import maskstarsSEG
from .imageutils import skybgr
from .objectMasker import objectOccluded
from .result import Result
from .sersic import fitSersic
from .pixmap import pixelmap

__all__ = ["calcMorphology"]


def calcMorphology(files, outfolder, asymmetry=False, shapeAsymmetry=False,
                   allAsymmetry=True, calculateSersic=False, savePixelMap=True,
                   saveCleanImage=True, imageSource=None, catalogue=None,
                   largeImage=False, paramsaveFile="parameters.csv",
                   occludedSaveFile="occluded-object-locations.csv"):
    '''
    Calculates various morphological parameters of galaxies from an image.

    Parameters
    ----------

    files : List[str] or List[Pathobjects] or generator object
        files to iterate over
    outfolder : Path object or str
        path to folder where data from analysis will be saved
    asymmetry : bool, optional
        Default false. If true calculates asymmetry value
    shapeAsymmetry : bool, optional
        Default false. If true calculates shape asymmetry value
    allAsymmetry : bool, optional
        Default True. If true calculates all asymmetry values.
    imageSource : str
        Contains the source of the image, i.e which telescope the image was
        captured by
    catalogue : str or Path, optional
        Catalogue of objects nearby object of interest. Is used to mask out
        objects that interfere with object of interest
    largeImage : bool, optional
        Default False. If true, a larger image is used to calculate the sky
        background
    paramsaveFile: str or Path object
        Name of file where calculated files are to be written
    occludedSaveFile: str or Path object
        Name of file where objects that occlude the object of interest are saved

    Returns
    -------

    results : Result data class
        Container of all calculated results

    '''

    outfile = outfolder / paramsaveFile
    csvfile = open(outfile, mode="w")
    paramwriter = csv.writer(csvfile, delimiter=",")
    paramwriter.writerow(["file", "apix", "r_max", "sky", "sky_err", "A", "Abgr",
                          "As", "As90", "fwhms", "theta", "sersic_amplitude",
                          "sersic_r_eff", "sersic_n", "sersic_x_0", "sersic_y_0",
                          "sersic_ellip", "sersic_theta", "time", "star_flag"])

    if catalogue:
        outfile = outfolder / occludedSaveFile
        objcsvfile = open(outfile, mode="w")
        objwriter = csv.writer(objcsvfile, delimiter=",")
        objwriter.writerow(["file", "ra", "dec", "type"])

    results = []

    for file in files:

        print(file)
        try:
            img, header, imgsize = checkFile(file)
        except IOError:
            print(f"File {file}, does not exist!")
            continue
        except AttributeError as e:
            continue

        # convert image data type to float64 so that later calculations do not
        # raise exceptions
        img = img.astype(np.float64)

        if catalogue is None:
            occludedSaveFile = ""

        newResult = Result(file, outfolder, occludedSaveFile)

        s = time.time()

        # get sky background value and error
        try:
            newResult.sky, newResult.sky_err, newResult.fwhms, newResult.theta = skybgr(img, imgsize, file, largeImage, imageSource)
        except AttributeError:
            # TODO can fail silently if some other attribute error is raised!
            newResult.write(paramwriter)
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')
            print(" ")
            continue

        tmpmask = pixelmap(img, newResult.sky + newResult.sky_err, 3)
        objlist = []
        if catalogue:
            newResult.star_flag, objlist = objectOccluded(tmpmask, file.name, catalogue, header)
            if newResult.star_flag:
                for i, obj in enumerate(objlist):
                    if i == 0:
                        objwriter.writerow([f"{file}", obj[0], obj[1], obj[2]])
                    else:
                        objwriter.writerow(["", obj[0], obj[1], obj[2]])

        starMask = maskstarsPSF(img, objlist, header, newResult.sky)
        newResult.starMask = starMask
        mask = pixelmap(img, newResult.sky + newResult.sky_err, 3, starMask)

        img -= newResult.sky

        # clean image of external sources
        img = maskstarsSEG(img)

        if saveCleanImage:
            filename = file.name
            filename = "clean_" + filename
            outfile = outfolder / filename
            newResult.cleanImage = outfile
            hdu = fits.PrimaryHDU(data=img, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        if savePixelMap:
            filename = file.name
            filename = "pixelmap_" + filename
            outfile = outfolder / filename
            newResult.pixelMapFile = outfile
            hdu = fits.PrimaryHDU(data=mask, header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')

        objectpix = np.nonzero(mask == 1)
        cenpix = np.array([int(imgsize/2) + 1, int(imgsize/2) + 1])

        distarray = distarr(imgsize, imgsize, cenpix)
        objectdist = distarray[objectpix]
        newResult.rmax = np.max(objectdist)
        aperturepixmap = aperpixmap(imgsize, newResult.rmax, 9, 0.1)

        newResult.apix = minapix(img, mask, aperturepixmap, starMask)
        angle = 180.

        if asymmetry or allAsymmetry:
            newResult.A = calcA(img, mask, aperturepixmap, newResult.apix, angle, starMask, noisecorrect=True)

        if shapeAsymmetry or allAsymmetry:
            newResult.As = calcA(mask, mask, aperturepixmap, newResult.apix, angle, starMask)
            newResult.As90 = calcA(mask, mask, aperturepixmap, newResult.apix, 90., starMask)

        if calculateSersic:
            p = fitSersic(img, newResult.apix, newResult.fwhms, newResult.theta)
            newResult.sersic_amplitude = p.amplitude.value
            newResult.sersic_r_eff = p.r_eff.value
            newResult.sersic_ellip = p.ellip.value
            newResult.sersic_n = p.n.value
            newResult.sersic_theta = p.theta.value
            newResult.sersic_x_0 = p.x_0.value
            newResult.sersic_y_0 = p.y_0.value

        f = time.time()
        timetaken = f - s
        newResult.time = timetaken
        newResult.write(paramwriter)
        results.append(newResult)

    if catalogue:
        objcsvfile.close()
    csvfile.close()
    return results
