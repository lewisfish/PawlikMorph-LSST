import csv
from multiprocessing import Pool
from pathlib import Path
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.nddata.utils import PartialOverlapError
try:
    import parsl
    from parsl.app.app import python_app
    from parsl.configs.local_threads import config
except ModuleNotFoundError:
    parsl = None

from .apertures import aperpixmap
from .asymmetry import calcA
from .asymmetry import minapix
from .engines import multiprocEngine
from .Image import Image
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


def calcMorphology(imageInfos, outfolder, largeImgFactor: float, npix: float, filterSize: int,
                   parallelLibrary: str, cores: int, numberSigmas: float,
                   asymmetry=False, shapeAsymmetry=False,
                   allAsymmetry=True, calculateSersic=False, saveSegMap=False,
                   saveCleanImage=True, imageSource=None, catalogue=None,
                   largeImage=False, paramsaveFile="parameters.csv",
                   occludedSaveFile="occluded-object-locations.csv",
                   segmap=False, starMask=False, CAS=True):
    '''
    Calculates various morphological parameters of galaxies from an image.

    Parameters
    ----------

    imageInfos : Generator
        Generator which yields Tuple[str, float, float].
        This maps to Tuple[filename, RA, DEC]

    outfolder : Path object or str
        path to folder where data from analysis will be saved

    largeImgFactor: float
        Factor to multiply cutout size by to create larger image for better
        skybackground estimation.

    npix : float
        Size to make cutout of larger image

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
    paramwriter.writerow(["file", "apix", "r_max", "r20", "r80", "sky",
                          "sky_err", "C", "A", "Abgr", "S", "As", "As90",
                          "Gini index", "M20", "fwhms", "theta",
                          "sersic_amplitude", "sersic_r_eff", "sersic_n",
                          "sersic_x_0", "sersic_y_0", "sersic_ellip",
                          "sersic_theta", "time", "star_flag",
                          "Masked pixel fraction", "Object on image edge?"])

    if parsl is None and parallelLibrary == "parsl":
        raise ImportError("Parsl not installed!")

    if catalogue:
        outfile = outfolder / occludedSaveFile
        objcsvfile = open(outfile, mode="w")
        objwriter = csv.writer(objcsvfile, delimiter=",")
        objwriter.writerow(["file", "ra", "dec", "type"])

    if parallelLibrary == "multi":
        # https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop
        pool = Pool(cores)
        engine = multiprocEngine(_analyseImage,
                                 [outfolder, filterSize, asymmetry,
                                  shapeAsymmetry, allAsymmetry,
                                  calculateSersic, saveSegMap,
                                  saveCleanImage, imageSource, catalogue,
                                  largeImage, paramsaveFile, occludedSaveFile,
                                  numberSigmas, segmap, starMask, CAS, npix, largeImgFactor])
        results = pool.map(engine, imageInfos)
        pool.close()
        pool.join()
    elif parallelLibrary == "parsl":
        parsl.load(config)
        outputs = []
        for imageInfo in imageInfos:
            output = _analyseImageParsl(imageInfo, outfolder, filterSize, asymmetry,
                                        shapeAsymmetry, allAsymmetry,
                                        calculateSersic, saveSegMap,
                                        saveCleanImage, imageSource, catalogue,
                                        largeImage, paramsaveFile, occludedSaveFile,
                                        numberSigmas, segmap, starMask, CAS, npix, largeImgFactor)
            outputs.append(output)
        results = [i.result() for i in outputs]
    else:
        results = []
        for imageInfo in imageInfos:
            result = _analyseImage(imageInfo, outfolder, filterSize, asymmetry,
                                   shapeAsymmetry, allAsymmetry,
                                   calculateSersic, saveSegMap,
                                   saveCleanImage, imageSource, catalogue,
                                   largeImage, paramsaveFile, occludedSaveFile,
                                   numberSigmas, segmap, starMask, CAS, npix, largeImgFactor)
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
def _analyseImageParsl(imageInfo, outfolder, filterSize, asymmetry,
                       shapeAsymmetry, allAsymmetry,
                       calculateSersic, saveSegMap,
                       saveCleanImage, imageSource, catalogue,
                       largeImage, paramsaveFile, occludedSaveFile,
                       numberSigmas, segmap, starMask, CAS, npix, largeImgFactor):
    '''Helper function so that Parsl can run

    Parameters
    ----------

    See _analyseImage

    Returns
    -------

    See _analyseImage

    '''

    return _analyseImage(imageInfo, outfolder, filterSize, asymmetry,
                         shapeAsymmetry, allAsymmetry,
                         calculateSersic, saveSegMap,
                         saveCleanImage, imageSource, catalogue,
                         largeImage, paramsaveFile, occludedSaveFile,
                         numberSigmas, segmap, starMask, CAS, npix, largeImgFactor)


def _analyseImage(imageInfo, outfolder, filterSize, asymmetry: bool,
                  shapeAsymmetry: bool, allAsymmetry: bool,
                  calculateSersic: bool, saveSegMap: bool,
                  saveCleanImage: bool, imageSource: str, catalogue,
                  largeImage: bool, paramsaveFile, occludedSaveFile, numberSigmas: float,
                  segmap: np.ndarray, starMask: np.ndarray, CAS: bool, npix: float, largeImgFactor: float):
    '''The main function that calls all the underlying scientific analysis code

    Parameters
    ----------

    imageInfo : Tuple[str, float, float]
        Tuple contains the filename, ra , and dec of image to be analysed

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

    saveSegMap : bool
        If true save segmentation map

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

    segmap : bool
        If true then use "segmap_" + file as segmap, and don't
        calculate segmap.

    starMask : bool
        If true then use "starmask_" + file as starmask.
        
    CAS : bool
        If True, then calculate gini, clumpiness/smoothness, M20, and
        concentration morphology parameters.

    npix : int
        Size to make the cutout image.

    largeImgFactor : int
        Factor to multiply cutout image size by to create larger image.

    Returns
    -------

    newResult : Result object
        Data class containing calculated parameters and flags

    '''

    file = Path(imageInfo[0])
    ra = imageInfo[1]
    dec = imageInfo[2]
    print(file)

    if catalogue is None:
        occludedSaveFile = ""
    newResult = Result(file.name, outfolder, occludedSaveFile)

    try:
        imgObj = Image(imageSource, filename=file)
        if imageSource == "LSST":
            camCol = imageInfo[3]
            run = imageInfo[4]
            field = imageInfo[5]
            imgObj.setView(ra, dec, run, camCol, field, npix=npix)
            img = imgObj.getImage()
            header = imgObj.getHeader()
            if largeImage:
                imgObj.setView(ra=ra, dec=dec, npix=npix * largeImgFactor)
                imgLarge = imgObj.getImage()
        else:
            imgObj.setView(ra=ra, dec=dec, npix=npix)
            img = imgObj.getImage()
            header = imgObj.getHeader()
            if largeImage:
                imgObj.setView(ra=ra, dec=dec, npix=npix * largeImgFactor)
                imgLarge = imgObj.getImage()
        if not largeImage:
            imgLarge = None

        imgsize = img.shape[0]
    except IOError:
        print(f"File {file}, does not exist!")
        return newResult
    except AttributeError as e:
        # Skip file if error is raised
        return newResult
    except PartialOverlapError as e:
        print(f"File {file.name} cannot be cutout at size {npix}!")
        return newResult

    # convert image data type to float64 so that later calculations do not
    # raise exceptions
    img = img.astype(np.float64)

    s = time.time()

    # get sky background value and error
    try:
        newResult.sky, newResult.sky_err, newResult.fwhms, newResult.theta = skybgr(img, file=file, largeImage=imgLarge, imageSource=imageSource)
    except AttributeError:
        # TODO can fail silently if some other attribute error is raised!
        if saveSegMap:
            filename = file.name
            filename = "segmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')
        print(" ")
        return newResult
    except MemoryError:
        if saveSegMap:
            filename = file.name
            filename = "segmap_" + filename
            outfile = outfolder / filename
            hdu = fits.PrimaryHDU(data=np.zeros_like(img), header=header)
            hdu.writeto(outfile, overwrite=True, output_verify='ignore')
        print(" ")
        return newResult

    if not segmap:
        # Create Object Mask using 8-connected snail shell if none provided 

        if catalogue:
            # if a star catalogue is provided calculate pixelmap and then see
            # if any star in catalogue overlaps pixelmap
            try:
                tmpmask = pixelmap(img, newResult.sky + newResult.sky_err,
                                   filterSize)
            except AttributeError:
                return newResult

            newResult.star_flag, newResult.objList = objectOccluded(tmpmask, (ra, dec),
                                                                    catalogue, header)

            # remove star using images PSF to estimate stars radius
            try:
                starMask = maskstarsPSF(img, newResult.objList, header, newResult.sky, numberSigmas, adaptive=False)#, sky_err=newResult.sky_err)
                newResult.starMask = starMask
            except KeyError as e:
                print(e, ", so not using star catalogue to mask stars!")
                starMask = np.ones_like(img)
                
            segmap = pixelmap(img, newResult.sky + newResult.sky_err, filterSize, starMask)

        if starMask:
            # if a star mask is provided calculate pixelmap and then see
            # if any masked pixels overlaps pixelmap
            try:
                tmpmask = pixelmap(img, newResult.sky + newResult.sky_err,
                                   filterSize)
            except AttributeError:
                return newResult
            
            # read in starMask file
            with warnings.catch_warnings():
                # ignore invalid card warnings
                warnings.simplefilter('ignore', category=AstropyWarning)
                filename = file.name
                filename = "starmask_" + filename
                print('reading starmask file:',filename)
                starMaskpath = outfolder / filename
                starMask = fits.getdata(starMaskpath)
                
            segmap = pixelmap(img, newResult.sky + newResult.sky_err, filterSize, starMask)
            
        else:
            # no info provided on stars
            print('no star info provided, assuming no stars in field')

            try:
                segmap = pixelmap(img, newResult.sky + newResult.sky_err, filterSize)
                starMask = np.ones_like(img)
            except AttributeError:
                return newResult

        # check if pixelmap touch edge and flag if it does.
        newResult.objectEdge = checkPixelmapEdges(segmap)

    else:
        # read in segmap file
        with warnings.catch_warnings():
            # ignore invalid card warnings
            warnings.simplefilter('ignore', category=AstropyWarning)
            filename = file.name
            filename = "segmap_" + filename
            print('reading segmap file:',filename)
            segmappath = outfolder / filename
            segmap = fits.getdata(segmappath)
        # set starmask array as 1's as not using this if segmap provided
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

    if saveSegMap:
        filename = file.name
        filename = "segmap_" + filename
        outfile = outfolder / filename
        newResult.pixelMapFile = outfile
        hdu = fits.PrimaryHDU(data=segmap, header=header)
        hdu.writeto(outfile, overwrite=True, output_verify='ignore')

    newResult.rmax = calcRmax(img, segmap)
    aperturepixmap = aperpixmap(imgsize, newResult.rmax, 9, 0.1)

    # get centre of asymmetry
    newResult.apix = minapix(img, segmap, aperturepixmap, starMask)
    angle = 180.

    if asymmetry or allAsymmetry or CAS:
        newResult.A = calcA(img, segmap, aperturepixmap, newResult.apix, angle, starMask, noisecorrect=True)
        if catalogue:
            newResult.maskedPixelFraction = calcMaskedFraction(tmpmask, starMask, newResult.apix)

    if shapeAsymmetry or allAsymmetry:
        newResult.As = calcA(segmap, segmap, aperturepixmap, newResult.apix, angle, starMask)
        angle = 90.
        newResult.As90 = calcA(segmap, segmap, aperturepixmap, newResult.apix, angle, starMask)

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
        newResult.gini = gini(img, segmap)
        newResult.S = smoothness(img, segmap, newResult.apix, newResult.rmax, newResult.r20, newResult.sky)
        newResult.m20 = m20(img, segmap)

    f = time.time()
    timetaken = f - s
    newResult.time = timetaken

    return newResult
