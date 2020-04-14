import csv
import time
import warnings
from multiprocessing import Pool

import numpy as np
import parsl
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from parsl.app.app import python_app
from parsl.configs.local_threads import config

from .apertures import aperpixmap
from .apertures import distarr
from .asymmetry import calcA
from .asymmetry import minapix
from .helpers import checkFile
from .imageutils import maskstarsPSF
from .imageutils import maskstarsSEG
from .casgm import gini, m20, concentration, calcR20_R80, smoothness
from .objectMasker import objectOccluded
from .pixmap import calcMaskedFraction
from .pixmap import calcRmax
from .pixmap import pixelmap, checkPixelmapEdges
from .result import Result
from .sersic import fitSersic
from .skyBackground import skybgr

__all__ = ["calcMorphology"]


class Engine(object):
    '''Class existences so that Pool method can be used on _analyseImage.
       Basically a way to pass the function arguments that are he same with
       one variable argument, i.e the file name'''

    def __init__(self, parameters):
        '''This sets the arguments for the function passed to pool via
           engine'''
        self.parameters = parameters

    def __call__(self, filename):
        '''This calls the function when engine is called on pool'''
        return _analyseImage(filename, *self.parameters)


def calcMorphology(files, outfolder, filterSize, parallelLibrary: str, cores: int,
                   numberSigmas: float, asymmetry=False, shapeAsymmetry=False,
                   allAsymmetry=True, calculateSersic=False, savePixelMap=True,
                   saveCleanImage=True, imageSource=None, catalogue=None,
                   largeImage=False, paramsaveFile="parameters.csv",
                   occludedSaveFile="occluded-object-locations.csv",
                   mask=False, CAS=True):
    '''
    Calculates various morphological parameters of galaxies from an image.

    Parameters
    ----------

    files : List[str] or List[Pathobjects] or generator object
        files to iterate over
    outfolder : Path object or str
        path to folder where data from analysis will be saved
    filterSize : int
        Size of mean filter to use on object pixelmap
    parallelLibrary : str
        Chooses which parallel library to use
    cores: int
        Number of cores/processes to use for multiprocessing.
    numberSigmas : float
        The extent of which to mask out stars if a catalogue is provided.
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
        Name of file where objects that occlude the object of interest are
        saved
    CAS : bool
        If Tur, calculates CAS parameters

    Returns
    -------

    results : Result data class
        Container of all calculated results

    '''

    outfile = outfolder / paramsaveFile
    csvfile = open(outfile, mode="w")
    paramwriter = csv.writer(csvfile, delimiter=",")
    paramwriter.writerow(["file", "apix", "r_max", "sky", "sky_err", "A",
                          "Abgr", "As", "As90", "fwhms", "theta",
                          "sersic_amplitude", "sersic_r_eff", "sersic_n",
                          "sersic_x_0", "sersic_y_0", "sersic_ellip",
                          "sersic_theta", "time", "star_flag",
                          "Masked pixel fraction", "Object on image edge?"])

    if catalogue:
        outfile = outfolder / occludedSaveFile
        objcsvfile = open(outfile, mode="w")
        objwriter = csv.writer(objcsvfile, delimiter=",")
        objwriter.writerow(["file", "ra", "dec", "type"])

    if parallelLibrary == "multi":
        # https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop
        pool = Pool(cores)
        engine = Engine([outfolder, filterSize, asymmetry,
                         shapeAsymmetry, allAsymmetry,
                         calculateSersic, savePixelMap,
                         saveCleanImage, imageSource, catalogue,
                         largeImage, paramsaveFile, occludedSaveFile,
                         numberSigmas, mask, CAS])
        results = pool.map(engine, files)
        pool.close()
        pool.join()
    elif parallelLibrary == "parsl":
        parsl.load(config)
        outputs = []
        for file in files:
            output = _analyseImageParsl(file, outfolder, filterSize, asymmetry,
                                        shapeAsymmetry, allAsymmetry,
                                        calculateSersic, savePixelMap,
                                        saveCleanImage, imageSource, catalogue,
                                        largeImage, paramsaveFile, occludedSaveFile,
                                        numberSigmas, mask, CAS)
            outputs.append(output)
        results = [i.result() for i in outputs]
    else:
        results = []
        for file in files:
            result = _analyseImage(file, outfolder, filterSize, asymmetry,
                                   shapeAsymmetry, allAsymmetry,
                                   calculateSersic, savePixelMap,
                                   saveCleanImage, imageSource, catalogue,
                                   largeImage, paramsaveFile, occludedSaveFile,
                                   numberSigmas, mask, CAS)
            results.append(result)

    # write out results
    for result in results:
        result.write(paramwriter)
        if result.star_flag:
            for i, obj in enumerate(result.objList):
                if i == 0:
                    # obj[0] = RA, obj[1] = DEC, obj[2] = TYPE, obj[3] = psfMag_r
                    objwriter.writerow([f"{result.file}", obj[0], obj[1], obj[2]])
                else:
                    objwriter.writerow(["", obj[0], obj[1], obj[2]])

    if catalogue:
        objcsvfile.close()
    csvfile.close()
    return results


@python_app
def _analyseImageParsl(file, outfolder, filterSize, asymmetry,
                       shapeAsymmetry, allAsymmetry,
                       calculateSersic, savePixelMap,
                       saveCleanImage, imageSource, catalogue,
                       largeImage, paramsaveFile, occludedSaveFile,
                       numberSigmas, mask, CAS):
    '''Helper function so that Parsl can run

    Parameters
    ----------

    See _analyseImage

    Returns
    -------

    See _analyseImage

    '''

    return _analyseImage(file, outfolder, filterSize, asymmetry,
                         shapeAsymmetry, allAsymmetry,
                         calculateSersic, savePixelMap,
                         saveCleanImage, imageSource, catalogue,
                         largeImage, paramsaveFile, occludedSaveFile,
                         numberSigmas, mask, CAS)


def _analyseImage(file, outfolder, filterSize, asymmetry,
                  shapeAsymmetry, allAsymmetry,
                  calculateSersic, savePixelMap,
                  saveCleanImage, imageSource, catalogue,
                  largeImage, paramsaveFile, occludedSaveFile, numberSigmas,
                  mask, CAS):
    '''The main function that calls all the underlying scientific analysis code

    Parameters
    ----------

    file : Path object
        name of file to analyse

    outfolder : str or Path object
        folder where output files are to be saved

    filterSize : int
        Size of mean filter

    asymmetry : bool
        If true calculate asymmetry

    shapeAsymmetry : bool
        If true calculate shape asymmetry

    allAsymmetry : bool
        If true calculate all asymmetries

    calculateSersic : bool
        If true calculate Sersic profile

    savePixelMap : bool
        If true save pixelmap

    saveCleanImage : bool
        If true save cleaned image

    imageSource : str
        Contains the source of the image, i.e which telescope took the image

    catalogue : str or Path object
        If provided, contains the name of the file to be used as a star catalogue

    largeImage : bool
        If true use a larger image to estimate sky background value

    paramsaveFile : str
        Contains the name to save the parameters to

    occludedSaveFile : str
        Contains the name of the file to save the occluded objects to

    numberSigmas : float
        Number of sigmas to mask stars to.

    mask : bool
        If true then use "pixelmap_" + file as pixelmap, and don't
        calculate pixelmap.

    CAS : bool
        If True, then calculate gini, clumpiness/smoothness, M20, and
        concentration morphology parameters.

    Returns
    -------

    newResult : Result object
        Data class containing calculated parameters and flags

    '''

    print(file)
    if catalogue is None:
        occludedSaveFile = ""
    newResult = Result(file.name, outfolder, occludedSaveFile)

    try:
        img, header, imgsize = checkFile(file)
    except IOError:
        print(f"File {file}, does not exist!")
        return newResult
    except AttributeError as e:
        # Skip file if error is raised
        return newResult

    # convert image data type to float64 so that later calculations do not
    # raise exceptions
    img = img.astype(np.float64)

    s = time.time()

    # get sky background value and error
    try:
        newResult.sky, newResult.sky_err, newResult.fwhms, newResult.theta = skybgr(img, file=file, largeImage=largeImage, imageSource=imageSource)
    except AttributeError:
        # TODO can fail silently if some other attribute error is raised!
        filename = file.name
        filename = "pixelmap_" + filename
        outfile = outfolder / filename
        hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
        hdu.writeto(outfile, overwrite=True, output_verify='ignore')
        print(" ")
        return newResult
    except MemoryError:
        filename = file.name
        filename = "pixelmap_" + filename
        outfile = outfolder / filename
        hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
        hdu.writeto(outfile, overwrite=True, output_verify='ignore')
        print(" ")
        return newResult

    if not mask:
        if catalogue:
            # if a star catalogue is provided calculate pixelmap and then see
            # if any star in catalogue overlaps pixelmap
            try:
                tmpmask = pixelmap(img, newResult.sky + newResult.sky_err,
                                   filterSize)
            except AttributeError:
                return newResult

            newResult.star_flag, newResult.objList = objectOccluded(tmpmask, file.name,
                                                                    catalogue, header)

            # remove star using images PSF to estimate stars radius
            starMask = maskstarsPSF(img, newResult.objList, header, newResult.sky, numberSigmas, adaptive=False)#, sky_err=newResult.sky_err)
            newResult.starMask = starMask
            mask = pixelmap(img, newResult.sky + newResult.sky_err, filterSize, starMask)
        else:
            try:
                mask = pixelmap(img, newResult.sky + newResult.sky_err, filterSize)
                starMask = np.ones_like(img)
            except AttributeError:
                return newResult

        # check if pixelmap touch edge and flag if it does.
        newResult.objectEdge = checkPixelmapEdges(mask)
    else:
        with warnings.catch_warnings():
            # ignore invalid card warnings
            warnings.simplefilter('ignore', category=AstropyWarning)
            mask = fits.getdata("pixelmap_" + file.name)
        starMask = np.ones_like(img)

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

    newResult.rmax = calcRmax(img, mask)
    aperturepixmap = aperpixmap(imgsize, newResult.rmax, 9, 0.1)

    # get centre of asymmetry
    newResult.apix = minapix(img, mask, aperturepixmap, starMask)
    angle = 180.

    if asymmetry or allAsymmetry:
        newResult.A = calcA(img, mask, aperturepixmap, newResult.apix, angle, starMask, noisecorrect=True)
        if catalogue:
            newResult.maskedPixelFraction = calcMaskedFraction(tmpmask, starMask, newResult.apix)

    if shapeAsymmetry or allAsymmetry:
        newResult.As = calcA(mask, mask, aperturepixmap, newResult.apix, angle, starMask)
        angle = 90.
        newResult.As90 = calcA(mask, mask, aperturepixmap, newResult.apix, angle, starMask)

    if calculateSersic:
        p = fitSersic(img, newResult.apix, newResult.fwhms, newResult.theta, starMask)
        newResult.sersic_amplitude = p.amplitude.value
        newResult.sersic_r_eff = p.r_eff.value
        newResult.sersic_ellip = p.ellip.value
        newResult.sersic_n = p.n.value
        newResult.sersic_theta = p.theta.value
        newResult.sersic_x_0 = p.x_0.value
        newResult.sersic_y_0 = p.y_0.value

    if CAS:
        newResult.r20, newResult.r80 = calcR20_R80(img, newResult.apix, newResult.rmax)
        newResult.C = concentration(newResult.r20, newResult.r80)
        newResult.gini = gini(img, mask)
        newResult.S = smoothness(img, mask, newResult.apix, newResult.rmax, newResult.r20, newResult.sky)
        newResult.m20 = casgm.m20(img, mask)
        # pr, meanFluxatPr = calcPetrosianRadius(img, newResult.apix, newResult.fwhms, newResult.theta)

    f = time.time()
    timetaken = f - s
    newResult.time = timetaken

    return newResult
